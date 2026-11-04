# P4 — Faster Computation (Linux W7900)

**Date:** 2026-11-03 · **Runs:** `p34-F1-F1`, `p34-F2-F2`, `p34-LF1-LF1`, `p34-LF2-LF2`, `p34-LF3-LF3`
All independent A/Bs vs the frozen B0 (40-step bf16) or L1 (turbo 8-step) references; fresh server per variant; 3 prompts × 3 seeds.

## F1 — Attention backend (ck vs pytorch SDPA default)

Functional backends on this stack: **pytorch SDPA** (ComfyUI's aotriton probe
passes on Linux → flash + mem-efficient + math) and **Comfy Kitchen attention**
(`--use-ck-attention`, `comfy-kitchen` 0.2.40 HIP backend, `sol_attn`).
CUDA SageAttention2 is irrelevant on AMD. Backend verified from server logs
("Using Comfy Kitchen attention" vs "Using pytorch attention") — evidence
recorded in every run record.

| Metric | B0 (pytorch) | F1 (ck) | Δ |
|---|---|---|---|
| Warm total | 55.21 s | **45.09 s** | **1.22×** |
| DiT denoise | 53.88 s | 43.15 s | 1.25× |
| Steady step time | 1.345 s | 1.077 s | 1.25× |
| Peak device VRAM | 37.7 GB | 37.7 GB | — |
| Cold start | 74.2 s | 84.6 s | slower first run (backend warm-up) |

Cross-check vs P2 kernel trace: attention was 32.4% of GPU busy; the measured
10.7 s DiT saving ≈ attention cost dropping from ~17.2 s to ~6.5 s (~2.6×
faster attention kernels) — consistent with the Amdahl bound (≤1.46×).

**Then against Turbo 8-step (LF1):** L1 12.13 s → LF1 (turbo+ck) **10.03 s**
(1.21× on the shortened pipeline — attention's *relative* share grows after
the step cut, so the backend gain persists).

## F2 — INT8 ConvRot DiT (40 steps)

| Metric | B0 (bf16) | F2 (int8) | Δ |
|---|---|---|---|
| Warm total | 55.21 s | **51.09 s** | **1.08×** |
| DiT denoise | 53.88 s | 49.09 s | 1.10× |
| Steady step time | 1.345 s | 1.220 s | 1.10× |
| Weight file | 14.23 GB | **7.26 GB** | −49% |
| Peak device VRAM | 37.7 GB | **31.0 GB** | −6.7 GB |
| LPIPS fidelity vs B0 (p1/p2/p3) | — | 0.021 / 0.009 / 0.005 | **near-lossless band** |

**INT8 is a memory-efficiency improvement first** (half the weights, −6.7 GB
runtime VRAM) with a modest 1.08× end-to-end gain at 1024² — the dequantize-
on-use path keeps most GEMMs bf16-bound. Recorded as measured; no speedup was
assumed. (Fidelity LPIPS 0.005–0.031 matches the historical Windows band
0.004–0.060.)

## F3 — Combination (measured, never multiplied)

Turbo 8-step + INT8 DiT + best validated attention, same work conditions:

| ID | Config | Warm total | Step time | vs B0 | Peak VRAM | LPIPS p1/p2/p3 |
|---|---|---|---|---|---|---|
| L1 | turbo bf16, 8 st | 12.13 s | 1.337 | 4.55× | 37.7 GB | 0.298/0.187/0.212 |
| LF1 | turbo bf16 + ck | 10.03 s | 1.067 | 5.50× | 37.7 GB | 0.299/0.182/0.209 |
| LF2 | turbo int8 | **9.02 s** | 0.916 | **6.12×** | 31.0 GB | 0.302/0.190/0.209 |
| LF3 | turbo int8 + ck | **9.02 s** | 0.906 | **6.12×** | 31.0 GB | 0.302/0.183/0.210 |

Independent ratios would have "predicted" 4.55 × 1.08 × 1.25 ≈ 6.1× — the
measured compound is 6.12×, but the composition is *not* the product of
independent gains: ck attention adds **nothing** on the int8 turbo pipeline
(LF2 = LF3 = 9.02 s; the int8 attention path already runs the faster kernels),
while int8 adds more to turbo (1.35× on the 8-step DiT) than to the 40-step
base (1.10×). **Bottleneck migration:** after L1, the denoise drops to 10.8 s
and VAE (0.9 s) + TE (0.5 s) + fixed costs (~0.8 s) become 18% of the total;
after LF2, DiT is 7.4 s of 9.0 s (82%) — the next lever remains DiT compute,
but attention-specific optimizations are exhausted on the int8 path.

## F4 — torch.compile: NOT_RUN (rejected without measurement, with reason)

P2 profiling shows elementwise+norm = 6.0% of GPU busy and GPU is 97.8% busy —
the theoretical ceiling of any host-side/fusion improvement is ~6% of DiT time
(< 0.4 s e2e on B0, less on turbo). Triton exists on this Linux stack (unlike
the Windows blocker) but compile cost + shape specialization + numerics risk
are not justified by a <6% ceiling. Rejected at the decision matrix stage;
recorded as NOT_RUN with this evidence, per the "no marketing optimizations"
discipline.

## Reproduce

```bash
python -m lab run --experiment F1 --date <tag>   # or F2 / LF1 / LF2 / LF3
```
