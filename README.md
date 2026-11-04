# Making Generative AI Faster — A Measured Optimization Course

**以 Qwen-Image 2.1 推理加速调优为例的课堂教学项目**
*A hands-on inference-acceleration engineering course, built on a real,
fully-evidenced measurement campaign on a single AMD Radeon PRO W7900.*

> **One GPU. One model. One baseline. Three optimization strategies.
> Measurable results — every claim traces to a committed run record.**

This repository teaches **how to make an image-generation workload faster**
the way professionals do it:

**Baseline → Profiling → Less Computation → Faster Computation →
Parallel & Heterogeneous Computing → Combined Optimization → Quality Gate →
Public Reproduction.**

Everything is anchored in one real case study: Qwen-Image-2.1 on a single
Radeon PRO W7900 (gfx1100, 48 GB) under Linux ROCm, measured 2026-11-03 —
146 schema-valid run records, all committed under `evidence/`.

## The headline results (all measured, warm = median of 9, 1024²)

| Stage | Config | Warm | vs baseline | Peak VRAM |
|---|---|---|---|---|
| **B0** | bf16, 40 steps (frozen baseline) | 55.21 s | 1.00× | 37.7 GB |
| L1 | Turbo 8-step (*less computation*) | 12.13 s | 4.55× | 37.7 GB |
| F1 | ck attention (*faster computation*) | 45.09 s | 1.22× | 37.7 GB |
| F2 | INT8 weights (*memory first: −6.7 GB*) | 51.09 s | 1.08× | **31.0 GB** |
| **BEST** | Turbo INT8 + default attention | **9.02 s** | **6.12×** | **31.0 GB** |

Plus two **proven negative results** (the most instructive parts):
CPU text-encoding is 152–391× slower than the GPU despite numerical parity,
and CPU–GPU request pipelining — with 50.4 s of *timestamp-proven* overlap —
still loses 14× in throughput.

Quality: INT8 is near-lossless (LPIPS fidelity 0.005–0.031); Turbo changes
the schedule (distance 0.19–0.30) and its **human three-tier verdict
(无感/略降/不可接受) is pending** — speed never overrides the quality gate.

## Course structure (8 hours + labs + 90-min lecture)

| Module | Teaches | Key measured fact |
|---|---|---|
| [M0 Setup & discipline](course/M00_setup_and_discipline/LESSON.md) | fail-closed preflight, evidence culture | doctor gate, pinned stack |
| [M1 Baseline](course/M01_baseline/LESSON.md) | measuring before optimizing | 55.21 s, 1.345 s/step, reproducibility |
| [M2 Profiling](course/M02_profiling/LESSON.md) | pipeline + kernel tracing, Amdahl | GEMM 61.3% / attention 32.4% of busy |
| [M3 Less computation](course/M03_less_computation/LESSON.md) | step distillation, attribution | 4.55× — and why it isn't 5× |
| [M4 Faster computation](course/M04_faster_computation/LESSON.md) | attention backends, INT8, combinations | ck 1.22×; INT8 1.08× + memory win; LF2=LF3 |
| [M5 Parallel & heterogeneous](course/M05_parallel_heterogeneous/LESSON.md) | what one GPU + one CPU can't do | overlap proven, 14× throughput loss |
| [M6 Combined & quality gate](course/M06_combined_quality/LESSON.md) | BEST selection, LPIPS semantics | 6.12×; human verdict binding |
| [M7 Student challenge](course/M07_student_challenge/CHALLENGES.md) | open-source contribution practice | 6 graded tasks |

Full plan: [course/SYLLABUS.md](course/SYLLABUS.md) ·
90-minute lecture package: [course/LECTURE_DEMO.md](course/LECTURE_DEMO.md)
(8 slides + 4 interactive Q&As, every number real).

## Run the lab yourself

```bash
python -m lab doctor                # fail-closed preflight — must PASS
python -m lab baseline              # frozen B0 baseline (~16 min on a W7900)
python -m lab profile               # pipeline + rocprofv3 kernel profiling
python -m lab run --experiment L1   # or F1 F2 LF1 LF2 LF3 L1O BEST H1 H2
python -m lab quality --experiment F2
python -m lab report                # SUMMARY.json/md + charts
```

Requirements: Linux + ROCm with a working torch, any single Radeon
gfx1100-class GPU, ~62 GB disk for the SHA256-pinned weights, ComfyUI at
commit `b5a4321` (v0.43.0). Step-by-step: [docs/REPRODUCTION_GUIDE.md](docs/REPRODUCTION_GUIDE.md).
CPU-only students can still do M0–M2 exercises against the shipped evidence.

## Evidence & claims

- `evidence/results/` — every run record (timing, VRAM, attention-path log
  evidence, image hashes), env snapshots, profiling analyses, SUMMARY + charts
- `evidence/quality/` — paired grids + native-resolution typography crops
  for the human verdict; `evidence/samples/` — B0 & BEST measured images
- [docs/CLAIM_EVIDENCE_MATRIX.md](docs/CLAIM_EVIDENCE_MATRIX.md) — 25 claims,
  each with its evidence file and caveats
- [docs/case-study/](docs/case-study/) — the verbatim P0–P6 engineering
  journals (reading homework)
- [docs/FINAL_ENGINEERING_REPORT.md](docs/FINAL_ENGINEERING_REPORT.md) —
  executive summary, verdicts, recommendations

## The five laws (what this course really teaches)

1. Measure before optimizing.
2. One variable at a time.
3. Never multiply speedups — measure combinations end-to-end.
4. Speed ≠ quality; the human verdict is binding.
5. Claims ≤ evidence.

## Provenance & license

This course is derived from the community benchmark project
[AIwork4me/qwen-image-rocm](https://github.com/AIwork4me/qwen-image-rocm)
(MIT) — its Linux W7900 workstream provided the lab code and every
measurement. Code: **MIT**. Course texts, evidence data and reports:
**CC-BY-4.0**. Model weights are not included and remain under their
upstream licenses. Details: [LICENSE](LICENSE),
[DATA_LICENSE.md](DATA_LICENSE.md), [ATTRIBUTION.md](ATTRIBUTION.md).

Independent educational material — not an official AMD or Alibaba/Qwen
project. 中文说明见 [README.zh-CN.md](README.zh-CN.md)。
