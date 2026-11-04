# P2 — Baseline Profiling & Bottleneck Analysis (Linux W7900)

**Date:** 2026-11-03 · **Run:** `evidence/results/p2-profile-profile`
**Gate P2: PASS** — every proposed optimization below references an observed bottleneck.

## Layer A — Pipeline-level profiling (from the frozen B0 records, n=18 warm)

| Stage | Median | Share of warm total (55.21 s) | Evidence |
|---|---|---|---|
| `T_dit_denoise` (KSampler, 40 steps) | **53.88 s** | **97.6%** | ws node timing, records |
| `T_vae_decode` | 0.90 s | 1.6% | records |
| `T_text_encoder` (fresh prompt) | 0.56 s | ~1.0% (fresh only; warm = cache hit) | fresh-prompt arm |
| `T_other` (queue, save, client) | 0.43 s | 0.8% | derived |
| Steady step time (median) | 1.345 s | — | protocol v1.3 (step[0] excluded) |
| Full cold start | 74.17 s | — | cold record (boot 7.0 s separate) |

## Layer B — GPU kernel profiling (rocprofv3, real ComfyUI server process)

Method: server launched under `rocprofv3 --runtime-trace --output-format csv -P 230:90:1`
(bounded 90 s window); a driver client runs cold (model load) + warmup, then aligns one
warm generation inside the window. Shutdown via SIGINT to the rocprofv3 wrapper (it
finalizes and writes CSVs; SIGTERM leaves unconverted `.dat` buffers — a tooling lesson
recorded here for reproduction).

**Captured: exactly 40.0 denoise steps (95,600 kernel dispatches) + VAE decode** —
i.e. one complete warm generation, B0 configuration.

| Kernel class | Dispatches | GPU time | Share of GPU-busy |
|---|---|---|---|
| **GEMM** (`Cijk_Alik_Bljk…` hipBLASLt composable, 96×96×32 tiles) | 8,021 | **32.54 s** | **61.3%** |
| **Attention** (`attn_fwd`, AOTriton flash) | 1,248 | **17.23 s** | **32.4%** |
| Elementwise/activation (rope, cat, silu-path, 68,940 tiny 6.5 µs launches) | 83,067 | 2.49 s | 4.7% |
| Norm (comfy `adaln_kernel` hip_backend) | 2,573 | 0.71 s | 1.3% |
| Other (vol2col/im2col = VAE conv) | 535 | 0.15 s | 0.3% |
| **GPU busy** | — | **53.11 s of 54.31 s span** | **97.8%** |

Top kernels: `Cijk_Alik_Bljk…MT96x96x32` 32.3 s (n=7,718, mean 4.19 ms) ·
`attn_fwd` 17.2 s (n=1,248, mean 13.8 ms) · `adaln` 0.70 s (n=2,535).

**Host/HIP API view (same window):** 95,365 kernel launches, **197 memcpy API calls
(~156 tiny 36 µs H2D uploads — per-step sigmas/params), 1 sync call**. Models stay
resident: no PCIe weight shuffling (the Windows diffusers cpu_offload pathology does
not exist here). 633k `hipGetDevice/SetDevice` calls indicate host-side device-guard
churn, but with 97.8% GPU busy it does not bottleneck the GPU.

**Known limitation (recorded, not hidden):** the bounded 90 s window opened ~1 s after
the fresh-prompt generation began, so the ~0.5 s TE phase fell outside the captured
window in the "te" session. TE cost is measured at Layer A (0.31–0.60 s fresh); its
kernel composition is characterized by the dedicated H1 probe instead.

## Bottleneck table

| Component | Measured time | Share | Evidence | Optimization candidate |
|---|---|---|---|---|
| DiT GEMM (bf16) | 32.54 s | 61.3% of GPU busy | kernel trace | INT8 ConvRot DiT (F2), fewer steps (L1) |
| DiT attention (flash) | 17.23 s | 32.4% of GPU busy | kernel trace | attention backend A/B (F1: ck vs pytorch SDPA) |
| DiT elementwise+norm | 3.20 s | 6.0% | kernel trace | (fusion — torch.compile, blocked: no proven benefit yet) |
| VAE decode | 0.90 s | 1.6% of e2e | pipeline + trace | negligible at 1024² |
| Text encoder (fresh) | 0.31–0.60 s | ~1% | fresh-prompt arm | CPU offload for VRAM (H1) — not for speed |
| Transfers / idle | <0.1 s copies; 1.2 s idle | ~0% | trace | nothing to schedule around |

## Amdahl upper bounds (theoretical estimates — NOT results)

With DiT = 97.6% of warm e2e and GPU-busy 97.8%:
- Attention → 0 instantly: e2e ≤ 55.2 × (1 − 0.324×0.976) ≈ **1.46×**
- GEMM → 2× faster: e2e ≤ 55.2 / (1 − 0.613×0.976×0.5) ≈ **1.43×**
- Steps 40 → 8 with equal step cost: ≈ **4.6×** (step-external costs cap it below 5×)
- Both attention and GEMM improvements compound sub-multiplicatively.

## Optimization decision matrix

| Rank | Candidate | Hotspot ref | Expected e2e | gfx1100 compat | Effort | Quality risk |
|---|---|---|---|---|---|---|
| 1 | Turbo 8-step (L1) | removes 80% of ALL denoise | ~4–5× | proven (upstream ckpt) | low (weights ready) | different schedule — human grid review |
| 2 | INT8 DiT (F2) | GEMM 61% + weights 14.2→7.3 GB | ≤1.4× or memory win | proven (comfy-kitchen hip backend available) | low | near-lossless band (0.004–0.060 historical) |
| 3 | ck attention (F1) | attention 32% | ≤1.2–1.3× | available (`comfy_kitchen` hip `sol_attn`) | low | numerics path change — grid review |
| 4 | Combinations (LF*) | post-L1 the DiT share shrinks; attention share GROWS | measure | — | low | cumulative review |
| 5 | torch.compile (F4, optional) | elementwise 6% | small | Triton present on Linux (unlike Windows) | high (compile cost, shape specialization) | numerics — deferred unless justified |
| — | CPU TE (H1) | TE is only ~1% of time | none for speed; VRAM −~15 GB | ComfyUI `CLIPLoader device=cpu` | low | conditioning parity check |

Profiling verdict: **the W7900 baseline is GPU-compute-bound inside the DiT
(GEMM+attention = 93.7% of GPU busy; host/transfer overhead ≈ 0)** — so "less
computation" (steps) is the dominant lever, followed by GEMM cost (int8) and
attention cost (backend), exactly the Phase 3–4 experiment order.

## Reproduction

```bash
python -m lab profile --date <tag>   # pipeline + kernel layers
# artifacts: pipeline_breakdown.json, rocprof-{dit,te}/analysis.json,
#            *_kernel_trace.csv (~85 MB each, NOT committed), client events
```
