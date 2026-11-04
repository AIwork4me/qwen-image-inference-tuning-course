# Draft GitHub Issues (for the owner to publish — do NOT open without authorization)

*Prepared as part of the Radeon Cloud course upgrade. Each issue maps to
an M7 challenge task. The owner decides whether/when to publish them;
nothing here has been posted to GitHub.*

---

## Issue 1 — Title: `help wanted: harden scripts/cloud/start.sh on non-audited cloud images`

**Labels:** `good first issue`, `help wanted`
**Challenge:** M7 Task 1 (★☆☆)

The one-command onboarding (`scripts/cloud/start.sh`) was validated on a
single audited instance (W7900, ROCm 7.2.1, egress proxy, 98 GiB
/workspace). Other Radeon Cloud images — different ROCm releases, no
proxy, other SKUs (W7800/RX 7900/MI-class), smaller persistent storage —
may hit unhandled paths.

**Task:** run the script on your instance; when a step fails or degrades,
improve detection/reporting (never silent fallbacks). PRs follow
[docs/CONTRIBUTING_TUTORIAL.md](../CONTRIBUTING_TUTORIAL.md).

**Done means:** the failure mode prints an actionable message; re-run
succeeds or reports the exact blocker; before/after output + instance
facts (arch, VRAM, ROCm, free disk) in the PR.

## Issue 2 — Title: `enhancement: SUMMARY evidence links + reference-vs-student diff view`

**Labels:** `good first issue`, `enhancement`
**Challenge:** M7 Task 2 (★☆☆–★★☆)

`python -m lab report` writes correct but link-less SUMMARY tables.
Reading evidence means opening JSONs by hand.

**Task:** (a) emit relative links per row (run dirs, quality grids);
(b) add a read-only `report --diff` comparing student vs reference
medians per experiment (Δ, n, spread).

**Constraints:** reference-scope numbers must stay numerically identical
(formatting only); every link target must exist (testable on CPU CI).

## Issue 3 — Title: `test coverage: quality_gate.py pure functions`

**Labels:** `good first issue`, `tests`
**Challenge:** M7 Task 3 (★☆☆)

`collect_measured` / `make_paired_grid` / `make_native_crops` /
`lpips_table` and the manifest `quality_semantics` mapping have no direct
unit tests. Regression risk grows as students extend the tooling.

**Task:** CPU-only pytest coverage with tiny synthetic PNGs; mock
`_lpips_pair_impl`. Must cover: odd/even file sets, empty intersection
(SystemExit), crop geometry, per-prompt medians never pooled, and the
fidelity/distance mapping for every shipped manifest.

## Issue 4 — Title: `research: why does ck attention add exactly nothing on the int8-turbo path?`

**Labels:** `help wanted`, `research`, `gpu`
**Challenge:** M7 Task 4 (★★☆)

Measured facts: ck attention gives bf16-40-step a 1.25× DiT speedup, but
LF2 (turbo+int8+default attn) = LF3 (turbo+int8+ck) = 9.02 s to the
millisecond. No kernel-level explanation exists yet.

**Task:** bounded rocprofv3 windows on bf16 vs int8 runs; table of
attention kernel invocations (name/grid/duration) explaining the tie;
1-paragraph recommendation on remaining headroom. Commit analysis JSON +
writeup under `docs/research/` (NOT the raw CSVs).

## Issue 5 — Title: `research: INT8 relative gain 1.08× at 40 steps vs 1.35× on turbo — where does the delta live?`

**Labels:** `research`, `gpu`
**Challenge:** M7 Task 5 (★★☆)

Same quantization format, very different relative step-time gains (F2
1.220 s vs B0 1.345 s; LF2 0.916 s vs L1 1.337 s). Hypotheses:
shape-dependent dequant overhead, L2 residency of half-size weights,
per-shape kernel selection.

**Task:** per-shape GEMM time bucketing from bounded traces of F2/LF2;
quantified split of int8 path time between dequant kernels and GEMMs.

## Issue 6 — Title: `experiment: single-GPU batching (batch_size > 1) vs sequential BEST throughput`

**Labels:** `enhancement`, `gpu`, `advanced`
**Challenge:** M7 Task 6 (★★★)

Every course measurement is batch_size=1; the untested throughput
direction on one GPU is batching multiple images into one denoise.
Reference frontier: sequential BEST ≈ 6.6 img/min (fresh-prompt).

**Task:** new `BATCH1` manifest + workflow variant (never edit frozen
manifests); paired per-image latency + throughput at N ∈ {2, 4} with VRAM
peaks; verify batched numerics are per-row independent before any
quality-parity claim.

**Note:** requires owner agreement on a new experiment id in the
methodology matrix before merging.
