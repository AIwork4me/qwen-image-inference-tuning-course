# Instructor guide — Jupyter lab delivery

[中文](INSTRUCTOR_GUIDE.zh-CN.md)

**Audience:** engineering graduate students with basic Python. **Mode:** 9 sequential labs (~20–40 minutes each), or an evidence-first 100-minute subset. Do not require GPU just to teach scientific reasoning.

### Suggested class cadence

Use M0 + M1 as warm-up (source identity, reference vs smoke), M2 + M3 for profiling and Amdahl, M4 for precision/memory tradeoffs, M5.0 + M5.1 for a negative-result recovery and matched 2×2, M6 for quality, M7 for the evidence-backed PR capstone. Ask students to **predict the result before running each calculation**, then defend the denominator, provenance, workload and limitations.

### Assessment rubric

- 25% source provenance & baseline sampling (cold/warm/smoke separation)
- 25% algebra and independent rederivation from raw JSON
- 25% matched A/B design and non-confounded speedup claims
- 15% fidelity/distance/owner-quality verdict boundary
- 10% documented hypothesis, reproducible script and PR explanation

### Classroom operations

Start Jupyter in the repo root; use Python 3.12. The default Run All mode needs **no GPU**. Optionally enable chart cells with matplotlib. Demonstrate GPU smoke only on a provisioned ROCm teaching instance after separate explicit acknowledgement; remind students that --smoke cannot be a published median. Do not mark M6 BEST PASS; that requires independent owner assessment.

CI runs the notebook cells as Python code via pytest (not a live Jupyter kernel) to verify safe default behavior. A passing notebook test says nothing about physical ROCm execution or UI widget rendering. Student results must never overwrite evidence/.

**Teacher checks:** `python -m pytest tests/test_notebooks.py -q` and `python scripts/course-quality/check_notebooks.py`.

**Failure policy:** if hardware is blocked, teach the exact same statistical and quality methodology from committed evidence. A defensible negative result earns full credit.
