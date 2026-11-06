# PR #2 — Phase P4: Statistical and System Performance Audit

New physical measurements (this audit, unique run IDs, frozen campaign untouched):

- CPU A/B sessions: `runs/pr2-p4-cpuab-v2-114711/` (3 independent paired sessions; the
  failed first attempt `runs/pr2-p4-cpuab-114444/` is retained as honest failure evidence —
  it exposed the `lab/profile.py` stdlib-shadow bug fixed in P1-B).
- Balanced warm A/C pipeline: `runs/pr2-p4-pipewarm-120822/pipeline_warm/pipeline_warm.json`.

Host = the campaign host (EPYC 9334, 16-CPU cgroup, gfx1100 48 GB, ROCm 7.2.1, OV 2026.4.1).
**Deviation**: an unrelated ~1-core python workload (`yolo26-rocm-day1-labs`) ran throughout;
it consumes <7% of the 16-CPU quota and both A/B arms of every pair shared it (paired design).

## P1-D. CPU encoder A/B — additional paired sessions

Design: 3 independent process pairs, each: C0 = `m51_baseline.py --device cpu` (cold + 1 warm,
16 threads, frozen prompts/token IDs, same checkpoint lineage), C1b = deployment IR
(FP16-stored, BF16 execution) cold + 2 warm. Bounded to p2 (51 tok) + empty-negative (22→8 tok)
to bound CPU-time; p1/p3 single-session values remain from the campaign.

| Prompt | Arm | warm samples (s) | median | campaign value |
|---|---|---|---|---|
| p2-en-scene | C0 | 136.509 / 136.478 / 136.673 | **136.51** | 136.81 (Δ 0.2%) |
| p2-en-scene | C1b | 0.299 0.284 · 0.283 0.296 · 0.280 0.284 (n=6) | **0.284** | 0.285 |
| negative | C0 | 61.257 / 61.206 / 61.008 | **61.21** | 61.51 (Δ 0.5%) |
| negative | C1b | 0.198 0.206 · 0.193 0.198 · 0.204 0.202 (n=6) | **0.201** | 0.187 |

- **Speedup, p2: 136.51 / 0.284 = 480.7×** (campaign: 480.4×) — reproduced across 3 sessions;
  C0 session spread is 0.14% (136.478–136.673), so the ratio's uncertainty is dominated by
  C1b's ±3% — the order-of-magnitude (and even the 2-significant-digit ratio for p2) is now
  supported by n=3/n=6 paired samples.
- Negative: 61.21 / 0.201 = **304×** (campaign single-session: 328×) — same order, honest
  spread; the "331–480×" claim range refers to the positive prompts and remains accurate.
- Baseline naming (mandatory): **ComfyUI PyTorch CPU, BF16-stored weights with FP32
  manual-cast compute**, 16 threads — not "optimized PyTorch CPU". No universal claim beyond
  this stack/host.
- C0 RSS reproduced (34.5 GB); C1b deployment RSS 16.8 GB in these sessions (30.0 GB in the
  campaign's first-touch session — page-cache state differs; both recorded).

## P1-E. Balanced warm A/C pipeline — steady-state measured directly

Design: three independent warm server sessions — A (GPU-native TE; warmup then 2×4 frozen
requests), C (precomputed-cond only: **no CLIPLoader ever instantiated**, warmup itself
precomputed; persistent OV BF16 worker, next-request encoding overlapped), A2 (order-drift
control, 1×4). All requests back-to-back, models resident.

| Metric | A (GPU-native) | C (OV BF16 precomputed) |
|---|---|---|
| steady per-request SERVICE latency (median) | **13.031 s** (n=8; 13.029–13.033) | **12.060 s** (n=8; 12.027–12.198) |
| service-latency ratio | **C is 1.081× faster (8.05%)** — measured, non-overlapping spreads | |
| inter-request overhead (submission/monitoring/cond staging) | ≈0.54 s/request (gap totals 2.0–2.3 s per 4-request batch) | ≈1.10 s/request (4.4–4.6 s per batch; gen-level gaps are 0.0 — requests are back-to-back, the overhead is client-side orchestration) |
| ACTUAL warm batch throughput | **4.42 img/min** (batch walls 54.42/54.14 s) | **4.55 img/min** (batch walls 52.77/52.71 s) |
| actual batch-throughput ratio | **C is 1.029× faster (≈2.9%)** — the honest batch-level number | |
| VRAM peak | 34.17 GB | 17.98 GB (**−16.18 GB**) |
| CPU encode per request (hidden) | 0.60 s GPU TE (inside wall) | 0.37–0.67 s OV CPU (overlap Σ 3.11 s) |
| worker / server RSS | — / 1.92 GB | 16.8 GB / 1.82 GB |

**Service latency vs actual batch throughput (corrected distinction, PR2 final polish).**
Recomputed from committed raw timestamps
(`scripts/release-closure/latency_vs_throughput.py` →
`PR2_LATENCY_VS_THROUGHPUT.json`): removing the GPU TE improves what a single
requester waits by 8.05% (13.031→12.060 s), but the precomputed arm pays
≈0.56 s/request MORE client-side orchestration overhead (workflow submission +
cond-file staging + worker bookkeeping), so the **actual four-request batch
throughput improves only ≈2.9% (4.42→4.55 img/min)**. The earlier "4.60 vs
4.97 img/min" figures in this file's first edition were naive sum-of-walls
upper bounds and are superseded. The campaign's original conservative
"≈1.04× steady" batch-level estimate was therefore closer to the batch truth
than the 8% service-latency figure; both are now directly measured and
explicitly labeled. Campaign cold-inclusive decomposition from raw timestamps:
A sum-of-walls 65.162 s + inter-request gaps 0.71 s + pre-batch 0.74 s =
66.615 s recorded total (fully reconciled).

- **The campaign's "≈1.04× steady" was a conservative estimate; the directly measured warm
  steady gain is ≈8.1%** (0.97 s/request ≈ the removed GPU-TE encode + its overhead).
  Attribution is clean: identical DiT/VAE/seed/latents; C differs only by absent GPU TE.
- Cold-inclusive 1.30–1.33× (campaign) is confirmed as a **session-start effect**: both the
  campaign and this audit observe a recurring ~26–30 s first-request penalty in GPU-native
  sessions (this audit additionally caught one 30.06 s outlier on an A2 batch-first request;
  steady values unaffected — remaining A2 requests are 13.033 s exactly).
- GPU contention: none measurable — C's generation (12.03–12.20 s) matches the campaign's
  no-contention values while 16-thread CPU encoding runs concurrently (hidden under GPU denoise).

## P1-F. Speed vs memory objectives — validated inventory

| Objective | Value | Status |
|---|---|---|
| Encoder-only speedup (deployment BF16 vs C0) | 331–480× positives (n=3/n=6 now) | PASS (order + ratio for p2) |
| Fresh-prompt (cold) encoder latency | 1.69–3.36 s first encode (warm process) | measured, this audit |
| Cold time-to-first-image, GPU-native session | ≈26–30 s recurring first-request penalty | measured twice (campaign + audit) |
| Warm E2E latency | 12.06 s (C) vs 13.03 s (A) | PASS (n=8/8) |
| 4-request throughput (warm) | 4.97 vs 4.60 img/min | PASS (n=8/8) |
| 4-request throughput (cold-inclusive session) | 4.80 vs 3.60 img/min (campaign) | PASS, session-start dominated |
| GPU TE residency elimination | C session never loads CLIP (verified by session design + 17.98 GB peak) | PASS |
| Full-pipeline VRAM peak saving | 16.18 GB (balanced audit) / 17.26 GB (campaign warm-session definition; 16.8–17.4 across request sets) | PASS, scope: turbo BF16 DiT, 8 steps, 1024², cfg=1 — **not** extrapolable |
| CPU RAM cost | +16.8 GB worker RSS (deployment IR) | recorded |
| Total CPU+GPU memory | heterogeneous trade-off, documented per arm | recorded |
| Export/compile/storage cost | FP32 export 26 GB + ~40 GiB RAM peak; compile 84.6 s (FP32) / 1.2 s (FP16-stored); 45.5 GB artifacts | recorded |

## Statistical honesty

- Warm steady conclusion (8.05%) is the only figure here with non-overlapping observed ranges
  (A: 13.029–13.033, C: 12.027–12.198) — classified **measured-PASS** on this host.
- Cold-inclusive ratios mix a session-start effect and are labeled as such everywhere.
- All numbers are single-host (one EPYC 9334 + one gfx1100); multi-host generalization is
  NOT_RUN by design.
