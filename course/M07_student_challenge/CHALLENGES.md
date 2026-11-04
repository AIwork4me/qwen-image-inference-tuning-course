# M7 — Student Challenge: Radeon AI Inference Speed Challenge

Student-friendly contribution tasks for the Linux W7900 lab. Difficulty is
honest: none of these is a `good first issue` unless marked so — GPU
performance work needs the discipline this repo enforces (baseline →
profile → one variable → evidence). Every task below ships with reproduction
commands, expected outputs, acceptance criteria and evidence requirements.

**Ground rules for all tasks**

1. Reproduce the frozen B0 baseline first (`python -m lab
   baseline`); your change is only meaningful against it.
2. One variable per experiment; never multiply independent speedups.
3. A run record (schema-valid, with attention evidence) or it didn't happen.
4. Negative results are welcome — with measurements.
5. Hardware: any single Radeon gfx1100 GPU works for most tasks; absolute
   numbers will differ from the W7900, methodology must not.

---

## 1. Attention kernel profiling deep-dive (difficulty: ★★☆ intermediate)

**Background:** P2 measured attention at 32.4% of GPU busy (`attn_fwd`, mean
13.8 ms, n=1248/gen) and ck attention making the bf16 pipeline 1.25× faster
on DiT — but adding nothing on the int8 turbo path (LF2=LF3). Nobody has yet
explained *why* at the kernel level (kernel names? shapes? occupancy?).

**Reproduce:** `python -m lab profile --date attn-deep`
then compare `rocprof-dit/analysis.json` across a bf16 run and an int8 run
(manifests F2/LF2, add `-P` windows per REPRODUCTION_GUIDE §3).

**Expected output:** a table of attention kernel invocations (name, grid,
VGPRs, duration) for bf16-DiT vs int8-DiT, explaining the LF2=LF3 result.

**Acceptance:** the explanation cites kernel rows from committed traces, not
conjecture; a 1-paragraph recommendation (is there headroom left?).

**Evidence:** trace CSVs (>10 MB — do not commit; commit the analysis JSON +
markdown).

## 2. INT8 kernel analysis: why only 1.08× at 40 steps but 1.35× on turbo? (★★☆)

**Background:** F2 (int8, 40 steps) step time 1.220 s vs B0 1.345 s; LF2
(turbo int8) 0.916 s vs L1 1.337 s. Same quantization format, very different
relative gains. Hypothesis space: GEMM shape-dependent dequant overhead,
L2 residency of the half-size weights, kernel selection per shape.

**Reproduce:** profile F2 and LF2 with bounded rocprofv3 windows; bucket GEMM
kernels by shape (M,N,K from grid dims is derivable); compare bf16 vs int8
GEMM total time per step.

**Acceptance:** per-shape GEMM time table + a quantified statement of where
the int8 path spends its time (dequant kernels vs GEMM).

## 3. Better memory scheduling (★★★ advanced)

**Background:** B0 peak device VRAM 37.7 GB on a 48 GB card; LF2 31.0 GB.
The W7900 has headroom the server doesn't use (models fully resident).
ComfyUI flags: `--highvram`, `--disable-smart-memory`, `--reserve-vram`.
Windows measurements showed `--highvram` REGRESSING 0.2–0.3×; Linux untested.

**Reproduce:** `python -m lab run --experiment B0 --date <tag>`
with `server_flags` overridden in a **new manifest** (never edit B0's) —
e.g. copy `B0.json` to your own lab branch with `"server_flags":
["--disable-smart-memory"]`.

**Acceptance:** paired warm medians (9 runs) + VRAM peaks + LPIPS fidelity
(bit-identical expected for pure VRAM levers — noise floor 0.0000).

## 4. CPU–GPU pipeline overlap on a beefier CPU (★★☆)

**Background:** H2 proved 50.4 s of genuine overlap but CPU TE cost 160–221
s/prompt (EPYC 9334, bf16). A modern desktop Zen 5 / Xeon or an int8 CPU TE
might change the calculus — or not.

**Reproduce:** `python -m lab run --experiment H2` on your host;
compare your CPU TE encode times with `p5-H1-r2-H1-cpu-te/h1_probe.json`.

**Acceptance:** H1 probe + H2 A/B on your hardware; conclusion states the
break-even CPU-encode speed at which pipelining wins (there is a closed-form
from the H2 numbers — derive it).

## 5. Automatic benchmark reports (★☆☆ good-first-issue)

**Background:** `python -m lab report` writes SUMMARY.md/json.
A chart per experiment (matplotlib, already a dep) committed next to the
summary would make PRs self-evident.

**Reproduce:** run any experiment + `report`; extend
`lab/report.py` with a `charts()` step (bar: warm median per
experiment with n= labels; waterfall: B0→L1→LF2 stages).

**Acceptance:** charts generated from SUMMARY.json only (never hand-edited
numbers); CPU-only unit test asserting chart files regenerate byte-stably.

## 6. Image quality assessment tooling (★☆☆ good-first-issue, careful semantics)

**Background:** quality gates are per-experiment manual invocations today; a
`python -m lab quality --experiment LF2` wrapper that runs
`lab/quality_gate.py` with the right semantics (fidelity for F2, distance
for turbo family) would remove a foot-gun.

**Acceptance:** semantics table lives in ONE place (manifest field), wrapper
refuses to guess, CPU tests pin the mapping. No quality verdicts are ever
auto-generated.

## 7. 2048² exploration (blocked until the owner records the 1024² quality verdict)

The mission gates 2048² behind stable 1024² + quality approval. On the W7900's
48 GB the bf16 VAE decode that OOM'd 24 GB cards may fit — but do not run
this until gate P6's human verdict lands.

---

**Submission:** PRs to `main` (or a course
branch the owner names), conventional commits, CPU tests green
(`pytest tests/ -v`), GPU evidence attached as run-record directories.
Owners of the repo review all quality verdicts personally.
