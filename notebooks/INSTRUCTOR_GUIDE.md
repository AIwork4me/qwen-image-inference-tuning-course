# Instructor guide — GPU-first Jupyter lab delivery

[中文](INSTRUCTOR_GUIDE.zh-CN.md)

**Audience:** engineering graduate students with basic Python who have completed Radeon Cloud onboarding and hold a real Radeon PRO W7900 instance. **Mode:** 9 sequential labs. `GPU_CLASSROOM` is the default student experience; `REFERENCE` remains the offline/CI fallback; `GPU_RESEARCH` is for capstone work.

### Prerequisites per student instance

1. `bash scripts/cloud/start.sh` completed (staged weights, pinned ComfyUI, `.venv-lab`, cloud doctor not FAIL). For M4/M6 arms and the full matrix: `--all-models` (~61 GB).
2. Jupyter launched from the repository root with the `.venv-lab` kernel available.
3. For M5.1: OpenVINO artifacts provisioned (`bash scripts/m51/provision_ov.sh`) or `M51_OV_ARTIFACTS` pointing at verified IRs. `PASS_WITH_WARNINGS` (unpinned XML digest) is an honest state — teach it, don't round it up.

### Suggested class cadence

M0 + M1 as warm-up (admission banner, sync discipline, your-own baseline vs the 55.206 s reference), M2 + M3 for profiling/Amdahl and the less-computation A/B **with the confound stated up front**, M4 for attention/quantization with backend-active verification, M5.0 + M5.1 for the measured negative result and its OpenVINO recovery (encoder speedup ≠ pipeline gain; fair 2×2 factorial), M6 for combined optimization + human quality review, M7 capstone PR. Ask students to **predict before running**, then defend denominator, provenance, population and limitations.

### Assessment rubric

- 20% admission + provenance discipline (mode honesty, measured/historical/estimated tags)
- 25% own-GPU measurement quality (warm populations, spread, VRAM accounting)
- 25% matched A/B design and non-confounded speedup claims (no F1×F2 multiplication)
- 15% fidelity/distance semantics + human-verdict boundary (PENDING_HUMAN_REVIEW)
- 15% capstone: reproducible report, honest evidence class (PILOT vs FORMAL)

### Classroom operations

- One GPU workload at a time: the runtime enforces a filesystem lock across kernels.
- A classroom arm is ~5 min (boot + cold + warmup + 2 measured). Budget ~20 min per GPU module; the full 9-module classroom pass fits a 2–3 h lab session.
- Run All is bounded in `GPU_CLASSROOM` by design (per-experiment timeouts, single prompt). Students must switch `COURSE_MODE` explicitly; nothing silently escalates to the research protocol.
- Kernel-level execution is now part of CI: `tests/test_notebooks.py` executes all 18 notebooks in REFERENCE mode through real Jupyter kernels (nbclient) when available.

**Teacher checks (no GPU):** `python -m pytest tests/test_notebooks.py tests/test_notebook_gpu_protocol.py -q`, `python scripts/course-quality/check_notebooks.py`.
**Teacher checks (W7900):** `python scripts/notebook-qualification/run_all.py --mode GPU_CLASSROOM --campaign runs/notebook-gpu-qualification/<id>` then `validate_evidence.py --campaign ...`.

**Failure policy:** a blocked admission is course material — the notebook prints the exact reason and the student switches to `REFERENCE` for the same methodology. A defensible measured regression earns full credit; a copied historical number does not.
