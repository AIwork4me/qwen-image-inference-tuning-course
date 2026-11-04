# Case-study reports (verbatim lab journals)

These P0–P6 reports are the engineering journals of the original measurement
campaign (2026-11-03, physical Radeon PRO W7900), lightly path-adapted from
the parent repository [AIwork4me/qwen-image-rocm](https://github.com/AIwork4me/qwen-image-rocm)
branch `feat/linux-w7900-optimization-lab`. They are the *evidence layer*
the course modules teach from.

Path map (parent repo → this course repo):

| Parent repo | This repo |
|---|---|
| `bench/linux_w7900/` (python package) | `lab/` |
| `bench/results/linux-w7900/<run>/` | `evidence/results/<run>/` |
| `bench/results/linux-w7900/quality-<exp>/` | `evidence/quality/<exp>/` |
| `bench/prompts.json`, `bench/schema/` | `lab/prompts.json`, `lab/schema/` |
| `bench/quickstart_config.py` (weight pins) | `lab/weights.py` |
| `phase2-comfyui/ComfyUI/` (pinned checkout) | `ComfyUI/` |
| `python -m bench.linux_w7900 ...` | `python -m lab ...` |

References to `bench/…` inside P0's repository-audit narrative describe the
parent repo's layout at audit time (historical record, kept verbatim).

Reading order: P0 (preflight) → P1 (baseline) → P2 (profiling) →
P3 (less computation) → P4 (faster computation) → P5 (parallel) →
P6 (combined + quality gate).
