# LECTURE DEMO — Making Generative AI Faster (Linux W7900 Lab)

**Course:** graduate-level enterprise AI engineering, Sun Yat-sen University
**Date:** 2026-11-19 · **Theme:** *Making Generative AI Faster: Less
Computation, Faster Computation, and Parallel Computing*
**Hardware story:** one AMD Radeon PRO W7900 (gfx1100, 48 GB), Linux ROCm
7.2.1, torch 2.9.1+rocm — every number below was measured on this GPU
2026-11-03 and resolves to a committed run record (see
[CLAIM_EVIDENCE_MATRIX.md](CLAIM_EVIDENCE_MATRIX.md)).

---

## Slide 1 — The real problem

**How long does one 1024×1024 image take, unoptimized?**

> **55.2 seconds** (warm, median of 9; the spread is 0.3%)
> — B0 frozen baseline: Qwen-Image-2.1 BF16, 40 steps, euler/simple, CFG 1.

Breakdown (measured): DiT denoise **53.9 s (97.6%)** · VAE decode 0.9 s ·
text encode 0.56 s (fresh prompt) · everything else 0.4 s.

Talking point: before any optimization, know where the 55 seconds live —
almost entirely inside the diffusion transformer.

## Slide 2 — Profile before optimizing

Two profiling layers, both on the real server process:

Pipeline layer (websocket node timing) → the table above.
Kernel layer (rocprofv3, bounded 90 s window, one full generation):

| Kernel class | share of GPU-busy (97.8%) |
|---|---|
| **GEMM** (hipBLASLt `Cijk_…`) | **61.3%** |
| **Attention** (`attn_fwd` flash) | **32.4%** |
| elementwise + norm | 6.0% |
| copies / syncs | ≈ 0 (197 tiny H2D, 1 sync) |

Talking point: the GPU is busy 97.8% of the time — this is a *compute*
problem, not a scheduling problem. Amdahl: even infinite-speed attention
caps us at ~1.46×; cutting steps is the big lever.

## Slide 3 — Less computation

Turbo 8-step distilled DiT (same TE/VAE/prompts/seeds):

> 55.2 s → **12.1 s = 4.55×** · per-step time *unchanged* (1.345 → 1.337 s)

The gain is *exactly* the removed compute: 32 fewer steps. And the reason it
isn't 5×: Amdahl — VAE + TE + fixed costs (~1.4 s) cap it at ≈4.6×. Measured
4.55×. Also: images legitimately DIFFER (different schedule; LPIPS distance
0.19–0.30 vs baseline) — "less computation" here is a quality tradeoff that
a human must judge (verdict pending), not a free lunch.

## Slide 4 — Faster computation

Two independent levers, A/B against the 40-step baseline:

- **Attention backend (F1):** Comfy Kitchen (ck) attention vs pytorch SDPA →
  step 1.345 → 1.077 s, **45.1 s = 1.22×**. Cross-check with slide 2:
  attention was 32% of busy time; measured saving 10.7 s ≈ attention cost
  17.2 → ~6.5 s. Theory bound 1.46× — measured 1.22×.
- **INT8 DiT (F2):** weights 14.2 → 7.3 GB, runtime VRAM 37.7 → 31.0 GB,
  quality near-lossless (LPIPS 0.005–0.031) — but only **1.08×** faster at
  1024². Theoretical peak throughput ≠ measured end-to-end: dequantize-on-use
  keeps most GEMMs bf16-bound.

Talking point: hardware-usage percentages are not latencies; only the paired
measurement counts.

## Slide 5 — Parallel computing (one GPU + one CPU)

Can CPU work overlap the GPU? We measured it (4 requests, turbo):

| Design | 4-image time | imgs/min |
|---|---|---|
| A sequential (GPU TE) | **51.1 s** | 4.69 |
| B pipelined (CPU TE overlapped) | 737.1 s | 0.33 |

B's overlap is *real* — 50.4 s of proven CPU∩GPU concurrency (timestamps in
`H2_RESULTS.json`) — yet B loses 14×: CPU TE costs 160–221 s per prompt
(8B encoder; 152–391× slower than the GPU's 0.3–0.6 s), AND the CPU matmuls
starve the GPU's launch threads (gens slowed 12→19 s).

Talking points: (1) throughput ≠ single-image latency — neither design
changes one image's 9–12 s; (2) parallelism has a price (contention);
(3) a negative result with proof is worth more than an optimistic guess.

## Slide 6 — Combined optimization (measured, never multiplied)

| Stage | warm | vs B0 |
|---|---|---|
| B0 baseline | 55.21 s | 1.00× |
| L1 turbo | 12.13 s | 4.55× |
| LF1 turbo+ck | 10.03 s | 5.50× |
| **LF2/BEST turbo+INT8** | **9.02 s** | **6.12×** |

The compound 6.12× is measured end-to-end — and it *breaks the product of
parts*: 4.55 × 1.08 × 1.25 would "predict" 6.14×, but ck attention adds
**zero** on the int8 turbo pipeline (LF2 = LF3 = 9.02 s). After the step cut
and int8, attention is no longer the binding constraint. Compose, then
re-measure.

## Slide 7 — Tradeoffs (the fastest is not automatically "the best")

- BEST (turbo-int8, 9.02 s): different schedule ⇒ **human verdict still
  pending** on the frozen grids (typography at native resolution is the
  acceptance axis). If the owner rejects turbo for text rendering, the
  fastest *quality-approved* config in our data is INT8-40-step (51.1 s,
  near-lossless fidelity, −6.7 GB VRAM).
- INT8's real win is memory (−49% weights): on a 24 GB card that's the
  difference between 2048² fitting or not (the Windows lab's experience).
- The quality gate is binding: LPIPS 0.21 distance ≠ "21% worse"; only eyes
  on grids decide.

## Slide 8 — Open-source challenge

Everything above is reproducible on a single Radeon GPU:
`doctor → baseline → profile → run --experiment L1|F1|F2|BEST → report`.
Six concrete student tasks (attention kernel deep-dive, INT8 GEMM analysis,
memory scheduling, CPU–GPU break-even derivation, auto charts, quality
tooling) with acceptance criteria: [OPEN_ISSUES.md](OPEN_ISSUES.md).
Bring a paired measurement, not a speedup story.

---

## Interactive classroom Q&A (answers backed by this lab's data)

**Q1 — Which operation consumes the most inference time, and how do we know?**
The DiT denoise: 53.9 of 55.2 s (97.6%) at the pipeline layer; inside it,
GEMM kernels take 61.3% of GPU-busy and attention 32.4% at the kernel layer.
We know from two independent instruments: websocket per-node/per-step timing
on the live server, and a rocprofv3 trace of the server process capturing
exactly 40.0 steps (95,600 kernel dispatches). Speaker note: stress that both
layers agreed before any optimization was attempted — that's what makes the
later attributions trustworthy.

**Q2 — If steps go 40 → 8, should generation automatically become 5× faster?**
No: measured 4.55×. The 32 removed steps were 97.6% of the workload, but the
remaining ~1.4 s (VAE + TE + queue/save) is now fixed cost — Amdahl's law in
one line: 55.2 / (55.2×0.976×(8/40) + 1.4) ≈ 4.6× ceiling, 4.55× measured.
Also per-step cost is unchanged (1.337 vs 1.345 s) — proof the gain is purely
"less computation", not hidden execution changes. Speaker note: ask students
to compute the ceiling live before revealing the measurement.

**Q3 — Why might an attention optimization help the baseline but barely move
an already-optimized pipeline?**
Measured here: ck attention gives bf16-40-step a 1.25× DiT speedup (attention
was 32.4% of busy), but on turbo-int8 it adds exactly nothing (LF2 = LF3 to
the millisecond). Two reasons: (1) the int8 path already routes attention
through the faster implementation, and (2) after cutting steps 5×, the same
0.3 s attention saving is a much smaller fraction of a 9 s total. Lesson:
re-profile after every big change; hotspots migrate. Speaker note: this is
also why we never multiply independent ratios — the levers interact.

**Q4 — With only one GPU, how can CPU–GPU scheduling improve multiple
requests?**
Honest answer from our measurements: on this host it can't — but we proved
why with numbers. True overlap exists (50.4 s of concurrent CPU-TE + GPU-denoise),
yet the CPU's 8B text encoder is 152–391× slower than the GPU's, and its
128-thread matmuls contend with kernel-launch threads, slowing the GPU 1.5×.
Net: 4.69 → 0.33 img/min. The break-even is computable: pipelining wins only
when CPU-encode time < per-request GPU time (here: never, 160 s vs 12 s).
Where CPU–GPU scheduling DOES pay on one GPU: overlapping *cheap* host work
(png save, queueing) — which ComfyUI already does — or batching multiple
images into one denoise (untested; open challenge). Speaker note: negative
results with timestamps are first-class science; invite students to find the
batching experiment in OPEN_ISSUES.

---

**Demo fallback (if live hardware is unavailable):** every figure above has a
committed evidence file (CLAIM_EVIDENCE_MATRIX) — present the grids
(`quality-BEST/quality_grid.png`), the kernel-class pie from
`rocprof-dit/analysis.json`, and the H2 timeline JSON. Do not invent charts:
this lab's discipline is that every pixel on a slide traces to a run record.
