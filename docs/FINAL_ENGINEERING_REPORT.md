# Final Engineering Report — Linux W7900 Optimization Lab

**Project:** Qwen-Image-2.1 inference optimization on a single AMD Radeon PRO
W7900 (gfx1100, 48 GB) under Linux ROCm · **Period:** 2026-11-03
**Branch:** `feat/linux-w7900-optimization-lab` · **Base:** upstream `main`
@ `acc318fc` (byte-verified mirror import; historical results untouched)

## Executive summary

**What worked**

- A complete, reproducible Linux lab on the physical W7900: frozen baseline →
  two-layer profiling → 10 measured experiment configurations → combined
  BEST → independent rerun, all schema-valid records with attention-path
  evidence, image hashes, dual-source VRAM peaks.
- **Baseline 55.21 s (warm, 1024², bf16-40-step) → BEST 9.02 s = 6.12×**,
  reproduced in a fresh process (9.020/9.023/9.020 s).
- Profiling did its job: the measured hotspots (GEMM 61%, attention 32%,
  GPU-busy 97.8%) correctly predicted which levers would pay (steps ≫ int8 >
  attention) and the Amdahl ceilings were confirmed within ~5%.
- INT8 DiT: −49% weights, −6.7 GB runtime VRAM, near-lossless fidelity
  (LPIPS 0.005–0.031) — a *memory* optimization first (1.08× speed).
- Honest negative results with proof: CPU TE (152–391× slower, parity
  cosine ≥ 0.999999) and CPU–GPU pipelining (14× throughput loss despite
  50.4 s of proven overlap + measured 1.5× GPU contention slowdown).

**What did not work / was rejected**

- ck attention on the int8 turbo pipeline: zero gain (LF2 = LF3 = 9.02 s) —
  rejected as a BEST component on simplicity grounds.
- torch.compile: NOT_RUN, rejected at the decision matrix (elementwise+norm
  ceiling < 6% of DiT; GPU 97.8% busy leaves nothing for host-side wins).
- 2048² resolution: not attempted — gated behind the pending 1024² quality
  verdict per the execution discipline.
- GitHub push / Draft PR: **BLOCKED** (egress proxy blocks github.com git
  endpoints; no credentials in this environment). The branch exists locally,
  commit-by-commit, ready to push by the owner.

**Main measured bottleneck:** the DiT denoise — 97.6% of warm end-to-end;
within it GEMM kernels 61.3% of GPU-busy, attention 32.4%.

**Highest-value optimization:** Turbo 8-step (4.55× alone) — "less
computation" dominates everything else, exactly as the profile predicted.

**Best valid end-to-end speedup:** **6.12×** (Turbo INT8 + default
attention; 55.21 → 9.02 s warm, same environment, paired protocol).

**Quality limitations:** the turbo schedule legitimately changes outputs
(LPIPS distance 0.18–0.34); the binding human three-tier verdict
(无感/略降/不可接受) on the frozen grids — including native-res Chinese
typography — is **PENDING_HUMAN_REVIEW**. No quality PASS is claimed.
If turbo is rejected for typography, the fastest fidelity-approved config
is INT8-40-step (51.1 s, near-lossless).

**Open research opportunities:** attention-kernel deep-dive on the int8 path,
per-shape INT8 GEMM analysis, memory-scheduling flags on 48 GB, batching for
single-GPU throughput, 2048² after the quality verdict (../course/M07_student_challenge/CHALLENGES.md).

## Experiment table (verdicts)

| ID | Configuration | Status | Warm (s) | VRAM dev (GB) | Quality | Evidence |
|---|---|---|---|---|---|---|
| B0 | bf16 40-step baseline | **PASS** | 55.21 | 37.7 | reference | `p1-baseline-{B0,r2-B0}` |
| L1 | Turbo bf16 8-step | **PASS** | 12.13 | 37.7 | distance 0.19–0.30 · review pending | `p34-L1-L1`, `quality-L1` |
| L1O | Turbo, official σ schedule | **PASS** | 12.07 | 37.7 | distance 0.23–0.34 · review pending | `p3-L1O-L1O`, `quality-L1O` |
| F1 | ck attention (40-step) | **PASS** | 45.09 | 37.7 | distance 0.004–0.065 · review pending | `p34-F1-F1`, `quality-F1` |
| F2 | INT8 DiT (40-step) | **PASS** | 51.09 | 31.0 | fidelity 0.005–0.031 (near-lossless) | `p34-F2-F2`, `quality-F2` |
| LF1 | Turbo+ck | **PASS** | 10.03 | 37.7 | distance · review pending | `p34-LF1-LF1`, `quality-LF1` |
| LF2 | Turbo INT8 | **PASS** | 9.02 | 31.0 | distance · review pending | `p34-LF2-LF2`, `quality-LF2` |
| LF3 | Turbo INT8+ck | **PASS** | 9.02 | 31.0 | distance · review pending | `p34-LF3-LF3`, `quality-LF3` |
| H1 | CPU text encoder | **PASS** (negative) | TE 137–207 s vs 0.35–1.17 | −16.8 (TE) | cond parity ✓ | `p5-H1-r2-H1-cpu-te` |
| H2 | 4-req pipeline | **PASS** (negative) | 737.1 vs 51.1 s | — | — | `p5-H2-H2-pipeline` |
| F4 | torch.compile | **NOT_RUN** (rejected w/ evidence) | — | — | — | P4 §F4 |
| BEST | Turbo INT8 + default attn | **PASS · PENDING_HUMAN_REVIEW** | **9.02** (rerun 9.02) | 31.0 | grids committed | `p6-BEST-rerun-BEST`, `quality-BEST` |

Every run: fresh server, health gate, frozen 3 prompts × 3 seeds, cold/warmup/
measured/fresh arms separated, schema validation, image SHA256, attention
evidence from server logs, env snapshots (`env/snapshot-*.json`).

## Final recommendation

**Which combination should an AMD Radeon developer use, and why?**

On a 48 GB W7900 under Linux ROCm with this stack (torch 2.9.1+rocm, ComfyUI
0.43.0): **Turbo INT8 ConvRot DiT + bf16 TE + default pytorch SDPA attention**
— 9.02 s/image warm (6.12×), 32.4 s cold, 31 GB peak, 25.5 GB of weights —
*conditional on the human quality verdict*; it is the measured fastest and
the simplest (no attention flags needed on Linux: the AOTriton flash path
that broke on Windows ROCm 7.14.1 works here). If the verdict rejects the
turbo schedule (especially for Chinese typography), use **INT8 40-step**:
near-lossless fidelity, 51.1 s, and the INT8 memory savings are the
enabler that made 2048² possible on 24 GB cards in this project's Windows
lab — on 48 GB it buys headroom for batching (untested, next step).

Key AMD-specific findings worth carrying forward: (1) ComfyUI's aotriton
flash probe passes on Linux ROCm 7.2.1 — no math-forcing needed, unlike the
Windows 7.14.1 wheels; (2) `comfy-kitchen`'s HIP backend (ck attention,
int8 convrot) is functional and fast on gfx1100; (3) sampled VRAM is a
lower bound — record torch AND device-level (rocm-smi) peaks; (4) CPU-GPU
pipelining for this workload class loses by 2–3 orders of magnitude on
server-class EPYC — keep the 8B encoder on the GPU.

**What should students optimize next?** Ranked by expected benefit/cost:
(1) batching (>1 images per denoise) for single-GPU throughput — the one
parallelism direction this lab did not test; (2) attention kernel deep-dive
on the int8 path (explain LF2=LF3, maybe find the next 10%); (3) per-shape
INT8 GEMM analysis (why 1.08× at 40 steps but 1.35× on turbo); (4) memory
scheduling flags on the 48 GB card; (5) 2048² once quality is approved.
All specified with acceptance criteria in [../course/M07_student_challenge/CHALLENGES.md](../course/M07_student_challenge/CHALLENGES.md).

## Owner actions required

See the mission response §10 — the two blocking items are the human quality
verdict (grids committed at `evidence/results/quality-*/`) and the
git push / Draft PR (branch ready; egress + credentials blocked here).
