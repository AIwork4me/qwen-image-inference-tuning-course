# Course Syllabus — Making Generative AI Faster

**以 Qwen-Image-2.1 在 AMD Radeon PRO W7900 上的推理加速调优为例**
*(a hands-on inference-acceleration engineering course built on a real,
fully-evidenced measurement campaign)*

中文版：[SYLLABUS.zh-CN.md](SYLLABUS.zh-CN.md)

- **Audience:** graduate students / engineers who can write Python and have
  used PyTorch; no prior AMD/ROCm experience required.
- **Format:** 7 modules ≈ 8 course hours + a 100-minute lecture (2026-11-19
  16:30–18:10) + lab work.
- **Hardware for labs:** a free Radeon Cloud GPU instance (single GPU,
  PyTorch-ROCm image) provisioned via `bash scripts/cloud/start.sh` — see
  [the quickstart](../docs/RADEON_CLOUD_QUICKSTART.md).
- **Language:** English-first canonical files with complete co-located
  Chinese versions (`*.zh-CN.md`).
- **Methodology spine:** [METHODOLOGY.md](METHODOLOGY.md) — Measure → Model
  → Hypothesize → Optimize → Validate → Explain → Reproduce → Contribute;
  read it before M1 and again after M6.

## The one-line story

> One GPU. One model. One baseline. Three optimization strategies.
> Measurable results — every claim traces to a committed run record.

## Modules

| # | Module | Methodology stage | Key measured fact (from the case study) |
|---|---|---|---|
| 0 | [Setup & the discipline](M00_setup_and_discipline/) | — | doctor fail-closed; evidence-before-claims |
| 1 | [Measure first: the frozen baseline](M01_baseline/) | Baseline | 55.21 s warm, 1.345 s/step (18 runs across two 9-run sessions, spread 0.3%) |
| 2 | [Profiling: where the time actually goes](M02_profiling/) | Profiling | GEMM 61.3% / attention 32.4% of GPU-busy 97.8% |
| 3 | [Less computation](M03_less_computation/) | Less compute | Turbo 8-step: 4.55× — and why it isn't 5× |
| 4 | [Faster computation](M04_faster_computation/) | Faster compute | ck attention 1.22×; INT8 1.08× + −6.7 GB VRAM |
| 5.0 | [PyTorch CPU text encoding baseline](M05_parallel_heterogeneous/LESSON.md) | Parallel baseline | measured 14× throughput LOSS (negative result) |
| 5.1 | [OpenVINO CPU text encoder](M05_parallel_heterogeneous/M5_1_OPENVINO_CPU_TE.md) | Research extension | same-stack W7900 A/B **COMPLETED 2026-11-05** (encoder 331–480× on positive prompts; warm per-request service latency ≈8% better, actual 4-request batch throughput ≈3% better; cold-inclusive 1.30–1.33×; 17.26 GB VRAM peak saved; INT8 negative-conditioning gate FAIL → exploratory; C1b-vs-G human quality verdict owner-accepted — scope in [PR2_OWNER_QUALITY_VERDICT](../docs/release-closure/PR2_OWNER_QUALITY_VERDICT.md); see [M51_RESULTS](../docs/M51_RESULTS.md)) |
| 6 | [Combined optimization & the quality gate](M06_combined_quality/) | Combined + quality | BEST 9.02 s = 6.12×; human verdict pending |
| 7 | [Student challenge](M07_student_challenge/) | Open source | 6 graded tasks with acceptance criteria |

Lab workflow (smoke vs formal, with GPU budgets):
[docs/CLASSROOM_WORKFLOWS.md](../docs/CLASSROOM_WORKFLOWS.md)

Plus: [LECTURE_DEMO.md](LECTURE_DEMO.md) — the 100-minute classroom package
(8 slides + 4 interactive Q&As + bounded teacher demo and student exercise,
all backed by the case-study data).

## Accompanying Jupyter practice

All nine teaching units (M0–M7 including M5.1) now have [English](../notebooks/README.md) and [中文](../notebooks/README.zh-CN.md) notebooks. Run All is **offline/CPU/read-only by default**; the code independently recomputes published W7900 statistics. GPU smoke is a separate explicit opt-in and is not valid as formal performance evidence. Use [instructor guidance](../notebooks/INSTRUCTOR_GUIDE.md) for pacing and grading.

## Grading rubric (suggested)

| Component | Weight | Pass bar |
|---|---|---|
| Lab notebooks (M1–M6 exercises) | 40% | commands + outputs committed, questions answered |
| Reproduced baseline on own GPU | 15% | schema-valid records, spread < 5% |
| One optimization experiment (A/B) | 25% | paired measurement + correct semantics label |
| Quality reasoning | 10% | fidelity vs distance not confused; verdict protocol |
| Challenge task PR (M7) | 10% | evidence attached, claims ≤ evidence |

**Academic-integrity rule (the course's core value):** a performance number
without a run record is not a result. Fabricating measurements is treated
like plagiarism. Negative results honestly reported earn full credit.

## The five laws of this course

1. **Measure before optimizing** — no baseline, no optimization claim.
2. **One variable at a time** — every A/B has a paired reference.
3. **Never multiply speedups** — combinations are measured end-to-end.
4. **Speed ≠ quality** — every speed change passes a declared-semantics
   quality gate; the human verdict is binding.
5. **Claims ≤ evidence** — see [docs/CLAIM_EVIDENCE_MATRIX.md](../docs/CLAIM_EVIDENCE_MATRIX.md)
   for what this looks like in practice (25 claims → files).

## Repository map

```
course/    teaching modules (this directory) + lecture demo
lab/       the runnable lab: python -m lab doctor|baseline|profile|run|report
evidence/  all measurements from the case study (records, grids, charts)
docs/      reproduction guide, claim matrix, troubleshooting, final report
docs/case-study/  the P0-P6 engineering journals (reading homework)
```
