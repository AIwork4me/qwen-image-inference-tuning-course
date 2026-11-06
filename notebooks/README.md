# Hands-on Jupyter Labs — Qwen-Image ROCm Optimization

[中文说明](README.zh-CN.md) · [Course syllabus](../course/SYLLABUS.md) · [Main README](../README.md)

**Nine exercises, two full language tracks, one scientific rule: no speedup without evidence.** The notebooks implement measurement, profiling, algorithmic reasoning, controlled A/B, quality gates, and PR evidence generation against committed physical Radeon PRO W7900 results.

## Start in 3 steps (GPU NOT required)

1. Clone the repository and open **the repository root** in a Jupyter server; on Radeon Cloud you can use its existing Jupyter UI. If Jupyter is not present, install the **separate optional** `requirements-notebooks.txt` into a notebook environment, not the pinned ROCm inference environment.
2. Open `notebooks/00_experimental_discipline.ipynb` (or the `.zh-CN.ipynb` partner) in a Python 3 kernel. Select **Run All**. Every notebook runs CPU-only from committed local JSON, with no network, model download, or GPU execution.
3. Proceed in order. Compare your calculations with the cited Markdown lessons and source JSON. Only run optional GPU cells after reading the resource notes.

**Important:** launch Jupyter from the cloned repository root; the first code cell locates `lab/reference_runs.json`. If your Jupyter server starts elsewhere, navigate to the repo before opening a notebook.

## Learning sequence

| Module | English notebook | 中文 Notebook | Key reproducible exercise |
|---|---|---|---|
| M0 · Environment / 实验规范 | [Open](00_experimental_discipline.ipynb) | [打开](00_experimental_discipline.zh-CN.ipynb) | Offline source-derived analysis |
| M1 · Baseline / 基准统计 | [Open](01_baseline_and_uncertainty.ipynb) | [打开](01_baseline_and_uncertainty.zh-CN.ipynb) | Offline source-derived analysis |
| M2 · Profiling / 性能剖析 | [Open](02_profile_and_bottlenecks.ipynb) | [打开](02_profile_and_bottlenecks.zh-CN.ipynb) | Offline source-derived analysis |
| M3 · Less compute / 少计算 | [Open](03_less_computation.ipynb) | [打开](03_less_computation.zh-CN.ipynb) | Offline source-derived analysis |
| M4 · Faster compute / 快计算 | [Open](04_faster_computation.ipynb) | [打开](04_faster_computation.zh-CN.ipynb) | Offline source-derived analysis |
| M5.0 · Negative offload / CPU 卸载负结果 | [Open](05_cpu_offload_negative.ipynb) | [打开](05_cpu_offload_negative.zh-CN.ipynb) | Offline source-derived analysis |
| M5.1 · OpenVINO fair A/B / 公平对照 | [Open](06_openvino_and_fair_ab.ipynb) | [打开](06_openvino_and_fair_ab.zh-CN.ipynb) | Offline source-derived analysis |
| M6 · Combined & quality / 组合与画质 | [Open](07_combined_and_quality.ipynb) | [打开](07_combined_and_quality.zh-CN.ipynb) | Offline source-derived analysis |
| M7 · Evidence-backed PR / 开源挑战 | [Open](08_capstone_and_pr.ipynb) | [打开](08_capstone_and_pr.zh-CN.ipynb) | Offline source-derived analysis |

## Three safe learning modes

**A · Evidence-only (default):** uses Python standard library and immutable `evidence/` + pinned `runs/` JSON. Any figures require optional matplotlib; all plotting cells default to disabled. This is also the mode tested on CPU CI.

**B · Bounded GPU smoke:** where offered, students intentionally set `RUN_GPU = True` and independently set `QIR_NOTEBOOK_GPU_ACK=YES` in their trusted kernel environment. The helper enforces allowlisted experiments with `--smoke`, fixed timeouts, and student-only `runs/` outputs. The user must first provision the ROCm/model stack with [Radeon Cloud quickstart](../docs/RADEON_CLOUD_QUICKSTART.md). **Do not run whole sessions in a shared teaching kernel or treat smoke data as a formal benchmark.** Notebook GPU is not tested by GitHub Actions.

**C · Formal research:** follow [reproduction guide](../docs/REPRODUCTION_GUIDE.md), including full-protocol runs, matching baseline, source SHA, CPU/GPU budgets, separate quality review, and PR evidence. Notebooks do not silently invoke formal runs.

## Quality and integrity

- English and Chinese notebooks have **identical executable cells** (CI-enforced) and localized learning goals, questions, cautions, and references.
- Source inputs are always read-only. No notebook writes to `evidence/`.
- M5.0 negative results are preserved. M5.1 C1b human acceptance **does not** make INT8 PASS. M6 BEST human-quality verdict remains **PENDING_HUMAN_REVIEW**.
- The instructor can verify every notebook without GPU using `python -m pytest tests/test_notebooks.py -q` and `python scripts/course-quality/check_notebooks.py`.
- Optional student output is **explicit opt-in**, written only to `runs/notebook-student/`, and labeled reference analysis rather than a new W7900 measurement.

**Requirements:** Python 3.10+ (recommended 3.12). Evidence-only needs stdlib. Optional Jupyter install: `python -m pip install -r requirements-notebooks.txt` in a dedicated notebook environment. Do **not** reinstall or replace the cloud image's ROCm torch.

Instructor notes: [English](INSTRUCTOR_GUIDE.md) · [中文](INSTRUCTOR_GUIDE.zh-CN.md).
