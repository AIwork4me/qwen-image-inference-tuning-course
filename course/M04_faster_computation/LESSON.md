# M4 — Faster Computation

**Goal:** accelerate the remaining math — attention kernels and GEMM
precision — as independent A/B experiments, then recombine *by measurement*.

## Lesson

### F1: attention backend (software choice, same hardware)

Options on gfx1100 + this stack: ComfyUI's default **pytorch SDPA**
(AOTriton flash works on Linux ROCm 7.2.1) vs **Comfy Kitchen (ck)
attention** (`--use-ck-attention`). Never assume a CUDA-only backend
(SageAttention) exists on AMD.

| Metric | B0 (pytorch) | F1 (ck) |
|---|---|---|
| Warm total | 55.21 s | **45.09 s (1.22×)** |
| Steady step | 1.345 s | 1.077 s (1.25×) |
| Peak VRAM | 37.7 GB | 37.7 GB |

Cross-check with M2: attention was 32.4% of busy ≈ 17.2 s; the DiT saved
10.7 s ⇒ attention now ≈ 6.5 s (ck's attention ≈ 2.6× faster). The Amdahl
ceiling was 1.46×; measured 1.22×. **The backend actually in use is proven
from server logs** ("Using Comfy Kitchen attention"), recorded per run —
flags alone are not evidence.

### F2: INT8 weights (precision choice)

| Metric | B0 (bf16) | F2 (int8 convrot) |
|---|---|---|
| Warm total | 55.21 s | 51.09 s (**1.08×**) |
| Weight file | 14.23 GB | **7.26 GB (−49%)** |
| Peak VRAM | 37.7 GB | **31.0 GB (−6.7 GB)** |
| LPIPS fidelity vs B0 | — | **0.005–0.031 (near-lossless)** |

**INT8 here is a memory optimization first.** "Theoretical 2× int8
throughput" did not materialize at 1024² — the dequantize-on-use path keeps
most GEMMs bf16-bound. This gap between hardware peak and measured latency
is the module's central lesson. On the 24 GB Windows card, this same memory
saving was the *only* reason 2048² fit at all.

### F3: combinations — measured, never multiplied

| Config | Warm | vs B0 |
|---|---|---|
| L1 turbo | 12.13 s | 4.55× |
| LF1 turbo+ck | 10.03 s | 5.50× |
| LF2 turbo int8 | **9.02 s** | **6.12×** |
| LF3 turbo int8+ck | **9.02 s** | **6.12×** |

The "product of parts" (4.55 × 1.08 × 1.25 ≈ 6.1×) happens to land near the
measured 6.12× — but the *composition* is different: **ck adds exactly
nothing on the int8 turbo pipeline** (LF2 = LF3). After int8, the attention
path already runs the fast kernels; after the step cut, the remaining
attention time is too small to matter. Levers interact — re-measure every
combination, and re-profile: the bottleneck migrates.

### The rejected experiment

torch.compile was **NOT_RUN — rejected with evidence**: elementwise+norm is
6.0% of busy and GPU is 97.8% busy, so the ceiling is < 0.4 s end-to-end,
against real compile cost + shape-specialization risk. Saying "no" with a
number is an optimization decision too.

## Exercise

1. `python -m lab run --experiment F1 --date <tag>` and `--experiment F2`.
2. From your F1 records: quote the `attention_evidence` log lines. From F2:
   report both VRAM peaks and the fidelity LPIPS (`quality` command).
3. Compute what fraction of F1's DiT saving is explained by the M2 attention
   share. What's the residual?
4. Essay (150 words): your manager wants INT8 "because it's 2× on paper".
   Using F2 data, write the reply.
5. Run `--experiment LF2` and `--experiment LF3`. Verify they tie. Explain
   why in terms of where time goes after turbo+int8.

**Acceptance:** three A/B tables from your records + the LF2=LF3 explanation.
