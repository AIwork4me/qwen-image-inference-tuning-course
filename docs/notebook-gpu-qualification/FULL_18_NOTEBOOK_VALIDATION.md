# Full 18-Notebook Validation

**Execution engine:** real Jupyter kernels (nbclient 0.10.4 + ipykernel 7.2.0, Python 3.12.3) — no `exec()` shortcuts.
**Campaign:** `runs/notebook-gpu-qualification/w7900-20261106-073429/`

## Pass A — REFERENCE (CPU-only, CI profile)

All 18 notebooks executed in fresh kernels with `QIR_COURSE_MODE=REFERENCE`:
no GPU touched, no network, no writes outside `runs/`, `[MEASURED_TODAY]` never printed.
Result: **18/18 PASSED** (final-head matrix: `matrix_REFERENCE_final.json`; this same
execution runs in CPU CI via `tests/test_notebooks.py::test_notebook_reference_run_all`).

## Pass B — GPU_CLASSROOM (physical W7900)

All 18 notebooks executed in fresh kernels with `QIR_COURSE_MODE=GPU_CLASSROOM`
on the qualification W7900. Every module runs its own bounded physical exercise
or consumes a current-campaign physical record with the run id displayed
(EN and ZH variants each execute independently; where a variant analyzes the
same shared campaign record, that record is displayed and is **not** double-counted
as two independent measurements).

Per-notebook rows (status, admission, gpu run ids, errors) live in
`NOTEBOOK_EXECUTION_MATRIX.json`; the summary table is below.

## Pass B matrix (GPU_CLASSROOM, physical W7900, 2026-11-06)

| Notebook | Language | Execution | Physical experiment | Physical run ids |
|---|---|---|---|---|
| 00_experimental_discipline.ipynb | en | PASSED | MEASURED (device admission + synced tensor probes) | — |
| 00_experimental_discipline.zh-CN.ipynb | zh-CN | PASSED | MEASURED | — |
| 01_baseline_and_uncertainty.ipynb | en | PASSED | MEASURED | 104534-nbq-B0 |
| 01_baseline_and_uncertainty.zh-CN.ipynb | zh-CN | PASSED | MEASURED | 105157-nbq-B0 |
| 02_profile_and_bottlenecks.ipynb | en | PASSED | MEASURED (pipeline from own records + campaign rocprofv3 trace) | — |
| 02_profile_and_bottlenecks.zh-CN.ipynb | zh-CN | PASSED | MEASURED | — |
| 03_less_computation.ipynb | en | PASSED | MEASURED | 105736-nbq-B0, 110245-nbq-L1 |
| 03_less_computation.zh-CN.ipynb | zh-CN | PASSED | MEASURED | 110440-nbq-B0, 111020-nbq-L1 |
| 04_faster_computation.ipynb | en | PASSED | MEASURED | 111233-nbq-B0, 111757-nbq-F1, 112131-nbq-F2 |
| 04_faster_computation.zh-CN.ipynb | zh-CN | PASSED | MEASURED | 112529-nbq-B0, 113104-nbq-F1, 113440-nbq-F2 |
| 05_cpu_offload_negative.ipynb | en | PASSED | MEASURED (pilots + campaign H1/H2) | 120754/120816-nbq-m51-te, q1-075553-H1-cpu-te, q1-075553-H2-pipeline |
| 05_cpu_offload_negative.zh-CN.ipynb | zh-CN | PASSED | MEASURED | 121131/121148-nbq-m51-te, q1-075553-H1/H2 |
| 06_openvino_and_fair_ab.ipynb | en | PASSED | MEASURED (OV artifact verify + campaign records) | q1-075553-H3_OV_BF16 |
| 06_openvino_and_fair_ab.zh-CN.ipynb | zh-CN | PASSED | MEASURED | q1-075553-H3_OV_BF16 |
| 07_combined_and_quality.ipynb | en | PASSED | MEASURED | 114607-nbq-B0, 115134-nbq-BEST |
| 07_combined_and_quality.zh-CN.ipynb | zh-CN | PASSED | MEASURED | 115309-nbq-B0, 115854-nbq-BEST |
| 08_capstone_and_pr.ipynb | en | PASSED | NOT_APPLICABLE (aggregates earlier records) | — |
| 08_capstone_and_pr.zh-CN.ipynb | zh-CN | PASSED | NOT_APPLICABLE | — |

Every performance-bearing notebook (M1, M3, M4, M6) executed **its own**
bounded physical arms per language variant (independent kernels, fresh
servers); M5 variants ran their own bounded encoder pilots and, in addition,
analyzed the shared campaign H1/H2/H3 records with the run ids displayed
(shared records are never counted as two independent measurements).

## Pass C — research experiments

Full frozen-protocol physical A/B executed by the campaign driver
(`gpu_campaign.py`): B0/L1/F1/F2/BEST × (3 prompts × 3 seeds × warmup+3 measured
+ fresh arm), `lab profile` (pipeline + two rocprofv3 windows), quality gates,
H1/H2, M5.1 encoder arms and pipeline arms. See `GPU_PERFORMANCE_MATRIX.md`.

## Two-dimensional verdicts (never collapsed)

Each module reports:
1. **Notebook execution:** PASSED / FAILED / BLOCKED
2. **Physical GPU experiment:** MEASURED / SMOKE_ONLY / REFERENCE_ONLY / BLOCKED / NOT_APPLICABLE

A CPU-green notebook is never reported as a physical performance PASS.
