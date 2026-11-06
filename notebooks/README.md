# Hands-on Jupyter Labs — GPU-First Qwen-Image ROCm Optimization

[中文说明](README.zh-CN.md) · [Course syllabus](../course/SYLLABUS.md) · [Main README](../README.md)

**Nine modules, two full language tracks, one scientific rule: no speedup without evidence — and on Radeon Cloud, the primary evidence is your own W7900.**

The labs implement the full experimental loop on a physical AMD Radeon PRO W7900:
**Understand → Predict → Run GPU Baseline → Implement/Select Optimization → Run GPU Candidate → Measure → Compare → Validate Quality → Explain → Contribute.**

## Start in 3 steps (GPU_CLASSROOM, the default lab mode on Radeon Cloud)

1. Complete the [Radeon Cloud quickstart](../docs/RADEON_CLOUD_QUICKSTART.md) once: it stages the pinned ROCm environment, ComfyUI and model weights and runs `python -m lab doctor` (must not FAIL).
2. Launch Jupyter **from the repository root** and open `notebooks/00_experimental_discipline.ipynb` (or the `.zh-CN.ipynb` partner) with the course environment's Python kernel.
3. In the **first code cell** set your mode, then *Kernel → Restart & Run All*:

```python
COURSE_MODE = "GPU_CLASSROOM"
```

An unmistakable banner shows the admitted GPU (`AMD ... gfx1100 ... 48GB`) or a student-readable blocker with the exact fix. Every physical run prints its `run_id` and lands under `runs/<run-id>/` — append-only, never overwriting, never touching frozen `evidence/`.

No GPU yet? Leave `COURSE_MODE = "REFERENCE"`: the notebook analyzes the committed frozen campaign (CPU-only, offline, no network) — the same cells, clearly labelled `[HISTORICAL]`.

## The three modes

| Mode | What runs | Where outputs land | Use it for |
|---|---|---|---|
| `REFERENCE` | frozen committed evidence only | nothing (analysis) | CPU CI, offline study, debugging |
| `GPU_CLASSROOM` | **your own bounded W7900 measurements** (1 prompt × warmup+2 measured per arm, ~5 min/arm) | `runs/` | the default student experience |
| `GPU_RESEARCH` | full frozen protocol (3 prompts × 3 seeds × warmup+3 measured + fresh-prompt arm) | `runs/` | capstone, research-grade reproduction |

Guards (enforced by the runtime and by tests): one GPU workload at a time (filesystem lock), hardware admission before any execution, bounded classroom budgets with per-experiment timeouts, research profile restricted to `GPU_RESEARCH`, append-only records, `QIR_NOTEBOOK_GPU_ACK`-style opt-in retained for the legacy smoke helper.

**A classroom exercise is a bounded physical experiment, not a formal performance claim.** Formal claims need the research protocol (repeated prompts/seeds, matched populations, declared ordering) — the notebooks label every comparison `BOUNDED_PHYSICAL_COMPARISON` vs `FORMAL_PROTOCOL_MEASURED` automatically.

## Where does each number come from? (student provenance)

The repository ships real physical W7900 records (the instructor campaigns under
`runs/`), so a fresh clone already contains physical evidence that is **not
yours**. Every number a notebook shows is therefore pinned to one of eight
machine-validated origins ([contract](../docs/notebook-gpu-qualification/STUDENT_PROVENANCE_CONTRACT.md)):

`CURRENT_SESSION_PHYSICAL` (your GPU, this session) · `SAME_STUDENT_PRIOR_PHYSICAL`
(your earlier session, selected by exact run id) · `SHARED_CAMPAIGN_PHYSICAL` ·
`COMMITTED_PHYSICAL_REFERENCE` (tracked in Git — instructor evidence, valuable
but **never** "measured today by you") · `FROZEN_REFERENCE` (`evidence/`) ·
`SIMULATED` · `NOT_RUN` · `BLOCKED`.

You choose the evidence source in the notebook's first code cell:

```python
RUN_OWN_GPU_ARMS = True    # run my own bounded arms on THIS GPU (default)
# or, to reuse earlier runs without spending GPU time (metered instances):
# RUN_OWN_GPU_ARMS = False
# REUSE_TE_RUNS = {"cuda": "<exact run id>", "cpu": "<run id>", "h3": "<run id>"}
# REUSE_PIPELINE_RUN = "<exact run id>"
```

Selecting a run always requires its **exact run id** (printed next to every
measurement and stored in the session records under `runs/` → `notebook-sessions/`).
Nothing is ever picked "because it is the newest file", and a reused run is
never counted as a second independent measurement — e.g. the Chinese and
English M5.1 notebooks may share one physical A/C run, labelled as reuse.

**When things go wrong (all recoverable, nothing is overwritten):**

- *Interrupted run / timeout* — re-run the cell: each attempt gets a fresh
  `run_id`; completed records stay untouched.
- *Missing OpenVINO artifacts (M5.1 admission `BLOCKED`)* — follow the printed
  remediation: `bash scripts/m51/provision_ov.sh`, or point
  `$M51_OV_ARTIFACTS` at the shared provisioned root (e.g. `/opt/m51-artifacts`).
- *Artifact `FAIL` (digest/identity)* — the artifact is invalid; dependent
  phases stop. Re-provision; never continue with an unverified artifact.
- *GPU busy* — another workload holds the device; finish it first (one GPU
  workload at a time, enforced by lock).
- *RAM admission `BLOCKED`* — free memory or use a larger instance; the
  safety headroom is never silently lowered.

## Learning sequence

| Module | English | 中文 | Physical exercise (GPU modes) |
|---|---|---|---|
| M0 · Environment / 实验规范 | [Open](00_experimental_discipline.ipynb) | [打开](00_experimental_discipline.zh-CN.ipynb) | device admission, sync-vs-async timing, model-identity digests |
| M1 · Baseline / 基准统计 | [Open](01_baseline_and_uncertainty.ipynb) | [打开](01_baseline_and_uncertainty.zh-CN.ipynb) | your own B0 (BF16, 40 steps) baseline + stats vs the 55.206 s reference |
| M2 · Profiling / 性能剖析 | [Open](02_profile_and_bottlenecks.ipynb) | [打开](02_profile_and_bottlenecks.zh-CN.ipynb) | pipeline decomposition + real rocprofv3 kernel trace, Amdahl ceiling |
| M3 · Less compute / 少计算 | [Open](03_less_computation.ipynb) | [打开](03_less_computation.zh-CN.ipynb) | B0 vs L1 (Turbo 8 steps) physical A/B, confound analysis |
| M4 · Faster compute / 快计算 | [Open](04_faster_computation.ipynb) | [打开](04_faster_computation.zh-CN.ipynb) | F1 ck-attention & F2 INT8 arms, backend-active verification |
| M5.0 · Negative offload / CPU 卸载负结果 | [Open](05_cpu_offload_negative.ipynb) | [打开](05_cpu_offload_negative.zh-CN.ipynb) | GPU vs CPU text-encoder pilots (the negative result, measured) |
| M5.1 · OpenVINO fair A/B / 公平对照 | [Open](06_openvino_and_fair_ab.ipynb) | [打开](06_openvino_and_fair_ab.zh-CN.ipynb) | OV artifact admission, encoder A/B, pipeline A/C, fair factorial |
| M6 · Combined & quality / 组合与画质 | [Open](07_combined_and_quality.ipynb) | [打开](07_combined_and_quality.zh-CN.ipynb) | B0 vs BEST physical + your own paired quality grid |
| M7 · Evidence PR / 开源挑战 | [Open](08_capstone_and_pr.ipynb) | [打开](08_capstone_and_pr.zh-CN.ipynb) | capstone report from your records + PR draft |

## Quality and integrity

- English and Chinese notebooks have **identical executable cells** (CI-enforced) and localized learning prose.
- Source inputs are read-only. No notebook writes to `evidence/`.
- Measured / historical / estimated / simulated values are tagged in every summary line.
- M5.0 negative results stay visible. M5.1 INT8 numerical **FAIL stays FAIL**; the C1b owner acceptance is scoped to its historically evaluated deployment. M6 BEST human-quality verdict remains **PENDING_HUMAN_REVIEW** — no cell auto-approves quality.
- Instructors can re-execute all 18 notebooks headlessly:
  `python scripts/notebook-qualification/run_all.py --mode REFERENCE` (CPU CI) or
  `--mode GPU_CLASSROOM` on the W7900 host.

**Requirements:** Python 3.10+ (3.12 recommended). REFERENCE mode needs the standard library only. GPU modes need the pinned `.venv-lab` ROCm environment from the quickstart (never reinstall the cloud image's torch). Optional notebook extras: `requirements-notebooks.txt`.

Instructor notes: [English](INSTRUCTOR_GUIDE.md) · [中文](INSTRUCTOR_GUIDE.zh-CN.md).
