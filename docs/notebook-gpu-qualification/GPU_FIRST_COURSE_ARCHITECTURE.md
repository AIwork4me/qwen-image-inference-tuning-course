# GPU-First Course Architecture

**Qualification campaign:** `runs/notebook-gpu-qualification/w7900-20261106-073429/` ·
**Source HEAD at start:** `cef55be` (PR #4) · **Hardware:** AMD Radeon PRO W7900 (gfx1100, 48 GB), Radeon Cloud.

## What changed relative to the previous (evidence-first) PR #4 state

| Aspect | Before (cef55be) | After |
|---|---|---|
| Primary evidence in notebooks | committed historical JSON | **the student's own physical W7900 measurements** (`GPU_CLASSROOM`), with historical data as reference/hypothesis |
| Execution modes | CPU default + opt-in smoke flags | one first-cell variable: `COURSE_MODE = REFERENCE \| GPU_CLASSROOM \| GPU_RESEARCH` |
| Admission | none in notebooks | `lab/notebook_runtime.py` hardware admission banner (device, gfx arch, VRAM, busy-GPU guard) with student-readable blockers |
| GPU invocation | `ns.gpu_smoke(...)` + env ack | bounded profiles through the pinned `python -m lab` CLI; one-GPU-workload filesystem lock; per-experiment timeouts; research profile gated to GPU_RESEARCH |
| Measurement semantics | smoke ≠ formal (prose) | machine-labelled populations: `classroom-bounded` / `smoke` / `formal protocol`, and comparisons classified `BOUNDED_PHYSICAL_COMPARISON` vs `FORMAL_PROTOCOL_MEASURED` |
| CI | exec() of code cells | all 18 notebooks executed in **real Jupyter kernels** (nbclient) in REFERENCE mode, plus negative GPU-protocol tests with a fake torch |
| M5.0/M5.1 | reference-only + simulation | bounded physical pilots (GPU/CPU encoder), OV artifact byte-verification, campaign A/C pipeline + fair factorial records |

## The learning loop each module implements

Understand → **Predict** → run GPU baseline → select/implement optimization → run GPU candidate → measure (median/spread/VRAM) → compare (labelled) → quality gate (human boundary) → explain deviations → contribute (M7 PR).

## Safety envelope (Run All must stay bounded)

- `REFERENCE` never imports torch; no network; no writes outside `runs/`.
- `GPU_CLASSROOM` budgets: 1 prompt × (cold + warmup + 2 measured) per arm ≈ 5 min/arm; per-experiment timeouts (900–1500 s); research protocol requires `GPU_RESEARCH`.
- Records are append-only (`runs/<run-id>/`), `evidence/` is never written (validated).
- GPU lock serializes workloads across kernels; busy-GPU admission refuses to start when the device already holds >8 GB.

## Files

- `lab/notebook_runtime.py` — the runtime (modes, admission, budgets, A/B, plotting, sessions).
- `scripts/notebook-qualification/build_notebooks.py` — single source generating all 18 notebooks (identical EN/ZH code cells).
- `scripts/notebook-qualification/run_all.py` — real-kernel execution harness (modes A/B/C).
- `scripts/notebook-qualification/summarize.py` / `validate_evidence.py` — evidence aggregation + fail-closed validation.
- `tests/test_notebooks.py` / `tests/test_notebook_gpu_protocol.py` — Layer A (CPU CI) + negative GPU-safety tests.
