# P1 — Frozen Baseline (Linux W7900)

**Date:** 2026-11-03 · **Branch:** `feat/linux-w7900-optimization-lab`
**Runs:** `evidence/results/p1-baseline-B0` (r1) and `p1-baseline-r2-B0` (r2, full instrumentation incl. rocm-smi device VRAM + GPU util)
**Gate P1: PASS** — physically validated, reproducible baseline before any optimization.

## Configuration (frozen)

| Field | Value |
|---|---|
| Diffusion model | `qwen_image_2.1_bf16.safetensors` (14.23 GB, sha256-pinned) |
| Text encoder | `qwen3vl_8b_bf16.safetensors` (17.53 GB) |
| VAE | `qwen_image_2.1_vae_bf16.safetensors` |
| Resolution | 1024 × 1024 · Steps 40 · CFG 1.0 · euler / simple |
| GPU | single Radeon PRO W7900 (gfx1100, 48 GB), ROCm 7.2.1, torch 2.9.1+rocm7.2.1 |
| ComfyUI | v0.43.0 @ pinned commit `b5a4321` |
| Attention | **normal functional Linux backend** = ComfyUI aotriton probe → "Using pytorch attention" (flash+mem-efficient+math enabled). Server-log evidence captured in every record. **No math-forcing applied** (the Windows `QIR_SDP=math` workaround is NOT used — it was a Windows-wheel AOTriton bug workaround and would artificially slow the Linux baseline). |
| Optimization | none (no smart-memory flags, no quantization, no turbo) |

Protocol: frozen prompt set (zh scene / en scene / zh typography poster) × seeds
[42, 1337, 20261030]; cold seed +1M, warmup seed +2M (cache-defeating, protocol v1.3);
fresh-prompt arm (" ." suffix) to expose real TE cost. 16 runs per session, two
independent sessions (r1/r2) for reproducibility.

## Results (r2 primary; r1 confirms within noise)

| Metric | Value | Detail |
|---|---|---|
| **Warm generation (median, n=9)** | **55.11 s** (r1: 55.24) | spread 55.10–55.28 (0.3%) |
| DiT denoise (KSampler, median) | 53.72 s | 40 steps |
| Steady step time (median) | 1.345 s | min 1.336 / max 1.367 (step[0] excluded per protocol) |
| VAE decode | 0.88 s | |
| Text encoder (fresh prompt) | 0.31–0.60 s | warm runs: TE node-cache hit (0 s) |
| Fresh-prompt total (median) | 56.10 s | warm + real TE encode |
| Full cold start (launch→1st PNG, first gen) | 74.17 s | server boot 7.0 s separate; UNETLoader ~19 s + CLIPLoader ~15 s dominate the delta |
| Peak VRAM (torch-sampled, 0.5 s) | 32.1–33.2 GB | lower bound (sampled) |
| Peak VRAM (rocm-smi device, 1 s) | 34.2–37.7 GB | includes non-torch allocations; also sampled |
| GPU utilization (rocm-smi 1 s samples, p50) | ~99–100 % during KSampler | see records |
| Host RAM | 55 GB cgroup, no swap activity observed | |

Latency definitions are kept strictly separate (warm / fresh / cold / boot) and are
never mixed (REAL_WORLD_LATENCY vocabulary).

## Reproducibility evidence

- Two independent full sessions (r1: 16 records, r2: 16 records) agree:
  warm medians 55.24 vs 55.11 s (0.2% apart), step medians 1.345 vs 1.345 s.
- All 32 records schema-valid (`bench/schema/result.schema.json`).
- Every record carries: run id, commit, env snapshot ref, seeds, sampler/scheduler,
  attention evidence (server log), image sha256 + path, VRAM samples, host mem.
- Images: `evidence/results/raw/<run_id>/` (3 prompts × 3 seeds measured
  + cold/warmup/fresh artifacts).

## Attention backend (actual, not assumed)

Server log evidence (in records): `Using pytorch attention`, plus
`Using AOTriton backend for Efficient Attention forward` warnings from `comfy/ops.py`
— i.e., PyTorch SDPA with AOTriton flash/mem-efficient kernels on gfx1100.
A separately-labeled Math-Attention diagnostic baseline was NOT needed on Linux
(the normal path is functional); the Windows `math` baseline is historical context only.

## Comparison with the historical Windows reference (background only, not a claim)

Windows RX 7900 XTX (24 GB, math-forced attention): ~45–51 s warm at the same
protocol. The Linux W7900 number (55.1 s) is a *different GPU + different attention
path* measurement — it is the Linux baseline, not a Windows comparison claim.
Notable: the W7900/Linux TE is fast (0.3–0.6 s fresh) because pytorch attention works
(Windows paid ~3–7 s with math-forced SDPA).

## Reproduction

```bash
python -m lab doctor                      # must PASS
python -m lab run --experiment B0 --date <tag>
```

**Gate P1 passed — optimization work may begin (Phase 2 profiling first).**
