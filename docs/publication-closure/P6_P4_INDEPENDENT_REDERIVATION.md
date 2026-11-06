# P6 — Independent Rederivation of the P4 Factorial Result

**Session:** final reproducibility & publication polish, 2026-11-06.
**Input (sole source):** the committed raw record
`runs/p4-fairab-225326/pipeline_profile/pipeline_profile.json` (blob
`7d100108…`). Fresh reimplementation of the statistics in this audit — the
runner's own `analyze()` output was **not** reused.

## 1. Cell statistics recomputed from the 24 raw batch walls

| Cell | n | mean | median | min–max | actual img/min = 4·60/mean |
|---|---:|---:|---:|---|---:|
| A-original | 6 | **53.996** | 53.765 | 53.620–54.994 | **4.445** |
| A-fast | 6 | **50.894** | 50.899 | 50.668–51.200 | **4.716** |
| C-original | 6 | **52.643** | 52.600 | 52.427–52.959 | **4.559** |
| C-fast | 6 | **49.967** | 49.957 | 49.705–50.226 | **4.803** |

Every mean/median/min/max/throughput equals the published P4 report table to
the last digit. (Attribution label: **MEASURED** — direct batch walls from the
raw record.)

## 2. Factorial effects recomputed

| Effect | Recomputed | Published claim | Label |
|---|---|---|---|
| Architecture gain, equivalent ORIGINAL clients (A/C) | 53.996/52.643 = **1.0257** (+2.6%) | "1.026×" | MEASURED (ratio of measured means) |
| Architecture gain, equivalent FAST clients (A/C) | 50.894/49.967 = **1.0186** (+1.9%) | "1.019×" | MEASURED (ratio of measured means) |
| Monitor gain within A | 53.996/50.894 = **1.0609** (+6.1%) | "+6.1%" | MEASURED |
| Monitor gain within C | 52.643/49.967 = **1.0536** (+5.4%) | "+5.4%" | MEASURED |
| Interaction (monitor effect A − monitor effect C) | (50.894−53.996)−(49.967−52.643) = **−0.426 s** | "−0.43 s" | DERIVED |

Published range "architecture ≈ +1.9–2.6% under equivalent clients": **confirmed**.

## 3. Attention points checked (mission list)

- **Block/session dependence**: per-block means — A-poll 53.836/54.156,
  A-fast 50.882/50.907, C-poll 52.566/52.720, C-fast 49.937/49.997 — every
  block reproduces every contrast's direction; block 2 reversed session order
  (C→A). No block flips any sign. (MEASURED per block.)
- **Interleaved execution order**: within every session batches alternated
  poll,fast — mode comparisons are within-session by construction; order is
  not a confounder for the monitor contrast. (MEASURED design field +
  batch sequence in the record.)
- **Model reloads**: 0/24 measured batches contain a loader re-execution
  (per-batch `model_reload_in_batch` false; the primary and warm-resident
  views coincide). The A sessions' documented post-warmup CLIP-eviction
  reload landed in the second warmup, outside all measured batches.
  (MEASURED flags; mechanism HYPOTHESIS per the P2 study.)
- **Persistent-state confounders**: sessions are warm by protocol; server RSS
  stable across batches per arm (record fields). Cross-session A-vs-C remains
  an inherent deployment difference (A sessions instantiate the GPU CLIP; C
  sessions never load it) — stated as a limitation, not hidden.
- **CPU worker startup**: worker load (~30 s) happens once per C session
  before warmup, never inside a measured batch (worker_load_s recorded per
  session; batches start after warmup). (MEASURED.)
- **Batch-wall semantics**: `batch_wall_s` spans first post → last write of
  the four requests (gaps included); service walls are per-request. The two
  are never mixed: recomputed service medians A-poll 13.034 s [13.030,13.073]
  vs A-fast 12.484 s [12.307,12.712] vs C-poll 12.066 s vs C-fast 12.096 s
  confirm the published secondary observation that A's *service* wall itself
  carried ~0.5 s of poll-quantization artifact. (MEASURED.)
- **Small sample size**: n=6 per cell (12 per arm). All four headline
  contrasts have non-overlapping batch ranges; no p-values, no universality
  claims — descriptive only. (Interpretation discipline.)
- **Source-code version**: `4ec5905a` + dirty worktree — see
  `P6_SOURCE_PROVENANCE_AUDIT.md` (SOURCE_RECONSTRUCTED_WITH_EVIDENCE) and
  the clean-HEAD regression (`P6_CLEAN_HEAD_PHYSICAL_VALIDATION.md`).

## 4. Resource costs recomputed (MEASURED)

- Full-pipeline VRAM peak medians: A ≈ **34.0 GB** vs C ≈ **17.0 GB** — the
  ~17 GB saving direction reproduces from raw per-request `vram_peak_mb`.
- CPU residency: C sessions carry the OV worker at ≈ 23.3 GB RSS (recorded
  per batch) vs A's ≈ 2.5 GB server RSS — the CPU-RAM-for-VRAM trade as
  taught.

## 5. Attribution vocabulary applied

- MEASURED: all cell walls/means/ranges, per-block values, service walls,
  reload flags, VRAM/RSS, worker timings.
- DERIVED: the five ratio/interaction effects (ratios of measured means),
  throughput values (240/wall).
- HYPOTHESIS (mechanism, not re-proven here): per-request RunMonitor teardown
  as the cause of the monitor effect (established by the P2 profiling study).
- ESTIMATED: none newly introduced.
- NOT VALIDATED / NOT_RUN: statistical significance beyond non-overlap,
  multi-host / other-workload generalization, cfg>1, 40-step.

## 6. Verdict

The published P4 numbers are **exactly reproducible from the committed raw
record** by an independent implementation; every attribution and limitation
statement in the P4 report survives rederivation. No document changes were
required by this audit beyond those already recorded (the reproduction-command
repair in `P4_FAIR_CLIENT_AB.md` §6, which touches no measurement).
