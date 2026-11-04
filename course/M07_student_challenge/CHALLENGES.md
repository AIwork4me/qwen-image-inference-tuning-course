# M7 — Student Challenge: Radeon AI Inference Speed Challenge

**Problem.** You have finished the course — now contribute something real
to an open-source project. The tasks below are genuine gaps in THIS
repository: each one is independently testable, ships acceptance criteria
and evidence requirements, and none is already implemented (features that
exist — report charts, the quality CLI — are listed under "already done"
so you don't re-propose them).

**Ground rules for all tasks**

1. Reproduce the frozen B0 baseline first (`python -m lab baseline`);
   your change is only meaningful against it.
2. One variable per experiment; never multiply independent speedups.
3. A run record (schema-valid, with attention evidence) or it didn't
   happen.
4. Negative results are welcome — with measurements.
5. Hardware, tiered honestly by measured VRAM peaks: the bf16 40-step
   experiments (B0, F1) peak at ~37.7 GB and need a gfx1100-class 48 GB
   card (W7900/W7900D); the int8/turbo experiments (F2, LF1–LF3, BEST)
   peak at ~31 GB and need ≥ ~33 GB. Smaller Radeons (24 GB class) can
   run the course's evidence-reading exercises and may run reduced
   resolutions/smoke variants, but the committed experiment matrix is NOT
   claimed to fit — cloud `doctor --cloud` accepts non-W7900 cards with a
   WARN, and your numbers are an independent measurement, never pooled
   with the reference.
6. Workflow: Fork → Edit → Test → PR — follow
   [docs/CONTRIBUTING_TUTORIAL.md](../../docs/CONTRIBUTING_TUTORIAL.md).
   CPU tests must pass (`python -m pytest tests/ -v`) before you push.

**Already implemented (do NOT re-propose as new features):**
SUMMARY charts from `report` (bar + waterfall, M2 links them), the
`python -m lab quality` CLI with manifest-driven semantics, scoped
reporting (`--scope reference|student|all`), runs/ evidence isolation,
cloud staged doctor. Improvements to these are fine as tasks 1–3 below.

---

## Task 1 — Quickstart & diagnostics hardening (★☆☆ good-first-issue)

**Gap:** `scripts/cloud/start.sh` was validated on ONE audited instance
(W7900, ROCm 7.2.1, proxy set). Other cloud images (different ROCm
releases, no proxy, other Radeon SKUs, smaller `/workspace`) may hit
unhandled paths.

**Do:** run the script on your instance; when a step fails or degrades,
fix the script (or `lab/doctor.py`) to detect and report your case
clearly — never to hide it.

**Expected evidence:** before/after script output on your hardware + the
instance facts (GPU arch, VRAM, ROCm, free disk) in the PR description.

**Acceptance:** the failure mode is detected with an actionable message
(no silent fallbacks); re-running the script after the fix succeeds or
reports the exact blocker; `bash -n` clean; no secrets in logs.

## Task 2 — Evidence navigation: SUMMARY linking & diffing (★☆☆–★★☆)

**Gap:** reading evidence today means opening JSON files by hand. The
SUMMARY tables (reference + student) could link each row to its run dirs,
records, and quality grids — and a `report --diff` could compare student
vs reference medians per experiment with spread and n.

**Do:** extend `lab/report.py` (markdown only — charts already exist) to
emit relative links per row, and add a read-only diff view.

**Expected evidence:** regenerated SUMMARY.md (reference scope must stay
numerically identical — this is a formatting change only) + a student
SUMMARY diffed against the reference.

**Acceptance:** CPU-only tests assert (a) reference numbers unchanged
byte-wise in the table cells, (b) every link target exists, (c) the diff
lists only experiments present in both scopes with n and Δ.

## Task 3 — Quality tool regression tests (★☆☆ careful-semantics)

**Gap:** `lab/quality_gate.py` (grids, crops, LPIPS table, verdict
checkboxes) has unit tests for the CLI wrapper but NOT for its pure
functions: `collect_measured` filename parsing, `make_paired_grid` /
`make_native_crops` geometry, `lpips_table` stats, and the
manifest-semantics mapping.

**Do:** add CPU-only pytest coverage with tiny synthetic PNGs (Pillow) —
no GPU, no lpips model needed (mock `_lpips_pair_impl`).

**Expected evidence:** the new test file passing on the CPU CI runner.

**Acceptance:** tests cover at least: an odd/even file set, a mismatched
pair set (empty intersection → SystemExit), crop coordinates vs canvas
size, per-prompt medians never pooled across prompts, and
`quality_semantics` fidelity/distance mapping from every manifest.

## Task 4 — Kernel profiling investigation: why does ck add nothing on
int8-turbo? (★★☆ intermediate, needs GPU)

**Background:** P2 measured attention at 32.4% of GPU busy (`attn_fwd`,
mean 13.8 ms, n=1248/gen) and ck attention making the bf16 pipeline
1.25× faster on DiT — but showing no measurable benefit on the int8 turbo
path (LF2 vs LF3: 9.023 vs 9.022 s medians, n=9 each, fully overlapping
distributions). Nobody has yet explained why at the kernel level.

**Reproduce:** `python -m lab profile --date attn-deep`, then compare
`rocprof-dit/analysis.json` across a bf16 run and an int8 run (manifests
F2/LF2, bounded `-P` windows per REPRODUCTION_GUIDE §3).

**Expected output:** a table of attention kernel invocations (name, grid,
duration) for bf16-DiT vs int8-DiT, explaining the LF2=LF3 result.

**Acceptance:** the explanation cites kernel rows from committed traces,
not conjecture; a 1-paragraph recommendation (is there headroom left?).

**Evidence:** trace CSVs are large — do not commit them; commit the
analysis JSON + a markdown writeup under `docs/research/`.

## Task 5 — INT8 kernel analysis: why only 1.08× at 40 steps but 1.35× on
turbo? (★★☆, needs GPU)

**Background:** F2 (int8, 40 steps) step time 1.220 s vs B0 1.345 s; LF2
(turbo int8) 0.916 s vs L1 1.337 s. Same quantization format, very
different relative gains. Hypothesis space: GEMM shape-dependent dequant
overhead, L2 residency of the half-size weights, kernel selection per
shape.

**Reproduce:** profile F2 and LF2 with bounded rocprofv3 windows; bucket
GEMM kernels by shape (M,N,K derivable from grid dims); compare bf16 vs
int8 GEMM total time per step.

**Acceptance:** per-shape GEMM time table + a quantified statement of
where the int8 path spends its time (dequant kernels vs GEMM).

## Task 6 — Single-GPU batching & throughput experiments (★★★ advanced,
needs GPU)

**Background:** every measurement in the course is batch_size=1. The one
untested throughput direction on one GPU: batch multiple images into a
single denoise (better GPU utilization per step) vs sequential BEST
requests (measured frontier: ~6.6 img/min fresh-prompt).

**Do:** add a `BATCH1` manifest + workflow variant (EmptyLatentImage
batch_size=N; prompts duplicated) — a NEW experiment id, never editing
existing manifests; measure warm latency per image and images/min vs
sequential BEST at N ∈ {2, 4}.

**Acceptance:** paired per-image latency + throughput table (N=1 vs N=2
vs N=4), VRAM peaks, and an honest quality note (batching changes the
batch-norm-free DiT math? verify numerics are per-row independent before
claiming quality parity).

**Evidence:** run records under `runs/` (never `evidence/`).

## Task 7 — 2048² exploration (BLOCKED until the owner records the 1024²
quality verdict)

The mission gates 2048² behind stable 1024² + quality approval. On a
48 GB card the bf16 VAE decode that OOM'd 24 GB cards may fit — but do
not run this until the human verdict lands.

---

**Submission:** PRs to `main`, conventional commits, CPU tests green
(`pytest tests/ -v`), GPU evidence attached as run-record directories
under `runs/`. Owners of the repo review all quality verdicts personally.
