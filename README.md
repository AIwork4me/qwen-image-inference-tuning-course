# Making Generative AI Faster — a measured optimization course on free Radeon Cloud GPUs

**以 Qwen-Image-2.1 推理加速调优为例的动手课程** ·
*A hands-on inference-acceleration engineering course. Every claim below
traces to a committed run record measured on a single AMD Radeon PRO W7900
(gfx1100, 48 GB, ROCm 7.2.1, PyTorch 2.9.1) — and you reproduce the whole
methodology yourself on a free Radeon Cloud GPU.*

---

## The measured result (this is real, not marketing)

Same prompt, same seed, same GPU — only the engineering changed:

| B0 frozen baseline · bf16, 40 steps · **55.21 s** | BEST · turbo-int8, 8 steps · **9.02 s** |
|:---:|:---:|
| <img src="evidence/samples/B0/p1-zh-scene-42-r0.png" width="380" alt="B0: water town at dusk, 55.21 s"> | <img src="evidence/samples/BEST/p1-zh-scene-42-r0.png" width="380" alt="BEST: same prompt, 9.02 s"> |
| <img src="evidence/samples/B0/p2-en-scene-42-r0.png" width="380" alt="B0: hiker at dawn"> | <img src="evidence/samples/BEST/p2-en-scene-42-r0.png" width="380" alt="BEST: same prompt"> |
| <img src="evidence/samples/B0/p3-poster-text-42-r0.png" width="380" alt="B0: typography poster"> | <img src="evidence/samples/BEST/p3-poster-text-42-r0.png" width="380" alt="BEST: same prompt — typography is the quality axis"> |

**6.12× faster, measured end-to-end** (warm, median; B0 pools two
independent 9-run sessions, every other row n=9, 1024×1024):

| Stage | Config | Warm | vs baseline | Peak VRAM |
|---|---|---|---|---|
| **B0** | bf16, 40 steps (frozen baseline) | 55.21 s | 1.00× | 37.7 GB |
| L1 | Turbo 8-step (*less computation*) | 12.13 s | 4.55× | 37.7 GB |
| F1 | ck attention (*faster computation*) | 45.09 s | 1.22× | 37.7 GB |
| F2 | INT8 weights (*memory first: −6.7 GB*) | 51.09 s | 1.08× | **31.0 GB** |
| **BEST** | Turbo INT8 + default attention | **9.02 s** | **6.12×** | **31.0 GB** |

> ⚠ **The honest caveat (this course's core value):** the 6.12× comes with
> a **quality review that is still `PENDING_HUMAN_REVIEW`**. Turbo uses a
> different schedule, so BEST's images legitimately differ from B0's
> (LPIPS distance 0.19–0.30; INT8 quantization itself is near-lossless,
> fidelity 0.005–0.031 vs a 0.0000 noise floor). Whether turbo's output is
> acceptable — especially poster typography at native resolution — is a
> **human three-tier verdict (无感/略降/不可接受) that the repo owner has not
> cast yet**. Speed never overrides the quality gate. The full grids:
> [`evidence/quality/BEST/`](evidence/quality/BEST/quality_grid.png),
> native-res text crops:
> [`quality_crops_poster.png`](evidence/quality/BEST/quality_crops_poster.png).

Two **proven negative results** (the most instructive parts of the course):
CPU text-encoding is 152–391× slower than the GPU despite numerical parity,
and CPU–GPU request pipelining — with 50.4 s of *timestamp-proven* overlap —
still loses 14× in throughput.

---

## Start learning (three steps, ~15 minutes to your first image)

**1 · Get a free Radeon Cloud GPU.** Apply for free AMD Radeon Cloud GPU
compute time (platform: <https://developer.amd.com.cn/radeon>, docs:
<https://amd-aim.github.io/radeon-cloud-docs/zh-cn/>), then start a
**single-GPU Linux Notebook with a PyTorch-ROCm image**. Quotas (hours, GPU
models, storage) are shown in the platform console — this course never
promises a specific number. [Readiness audit of a live instance →](docs/RADEON_CLOUD_READINESS.md)

**2 · Clone the course into `/workspace`:**

```bash
cd /workspace
git clone https://github.com/AIwork4me/qwen-image-inference-tuning-course.git
```

**3 · Run one command and get your first image:**

```bash
cd qwen-image-inference-tuning-course
bash scripts/cloud/start.sh
```

The script diagnoses your instance, borrows the preinstalled ROCm PyTorch
(never reinstalls it), fetches ComfyUI at the pinned commit, stages the
minimum SHA256-pinned model set (~25.5 GB, resumable), runs the fail-closed
doctor, and generates one bounded smoke image whose path it prints.
[Full quickstart + tested recovery guide →](docs/RADEON_CLOUD_QUICKSTART.md)

---

## What you will learn: the three ways to make inference faster

**① Less Computation** — remove work instead of speeding it up.
Step distillation: 40 → 8 denoising steps is the biggest single lever
(4.55×). But it is *not* 5× (Amdahl's law caps it at ~4.6×), per-step cost
is unchanged (1.345 → 1.337 s — proof the gain is purely removed compute),
and the output legitimately changes → a quality tradeoff a human must judge.

**② Faster Computation** — same work, faster kernels.
Attention backends (ck: 1.22×) and INT8 quantization (1.08× at 1024²,
but −6.7 GB VRAM — the difference between fitting and not fitting on
smaller cards). Hardware-usage percentages are not latencies: only paired
end-to-end measurement counts, and independent gains never multiply
(4.55 × 1.08 × 1.25 would "predict" 6.14×; measured 6.12× — and ck shows
no measurable benefit on the int8-turbo path: LF2 vs LF3 medians 9.023 vs
9.022 s, n=9 each, distributions fully overlapping).

**③ Parallel & Heterogeneous Computing** — when one GPU + one CPU can't.
We measured the seductive idea (CPU text-encoding overlapped with GPU
denoise) and it **lost 14×**: real overlap existed (50.4 s, proven by
timestamps), but the 8B CPU encoder is 152–391× slower and its matmuls
starve the GPU's launch threads. A negative result with proof is worth
more than an optimistic guess.

## Course structure (8 hours + labs + 100-min lecture)

| Module | Teaches | Key measured fact |
|---|---|---|
| [M0 Setup & discipline](course/M00_setup_and_discipline/LESSON.md) | fail-closed preflight, evidence culture | doctor gate, pinned stack |
| [M1 Baseline](course/M01_baseline/LESSON.md) | measuring before optimizing | 55.21 s, 1.345 s/step, reproducibility |
| [M2 Profiling](course/M02_profiling/LESSON.md) | pipeline + kernel tracing, Amdahl | GEMM 61.3% / attention 32.4% of busy |
| [M3 Less computation](course/M03_less_computation/LESSON.md) | step distillation, attribution | 4.55× — and why it isn't 5× |
| [M4 Faster computation](course/M04_faster_computation/LESSON.md) | attention backends, INT8, combinations | ck 1.22×; INT8 1.08× + memory win; LF2=LF3 |
| [M5 Parallel & heterogeneous](course/M05_parallel_heterogeneous/LESSON.md) | what one GPU + one CPU can't do | overlap proven, 14× throughput loss |
| [M6 Combined & quality gate](course/M06_combined_quality/LESSON.md) | BEST selection, LPIPS semantics | 6.12×; human verdict binding |
| [M7 Student challenge](course/M07_student_challenge/CHALLENGES.md) | open-source contribution practice | graded tasks + Fork→PR tutorial |

Full plan: [course/SYLLABUS.md](course/SYLLABUS.md) ·
Lecture (2026-11-19, 100 min): [course/LECTURE_DEMO.md](course/LECTURE_DEMO.md).

## Run the lab yourself

```bash
bash scripts/cloud/start.sh --all-models   # staged setup (or add models later)
.venv-lab/bin/python -m lab doctor         # strict preflight (formal runs)
.venv-lab/bin/python -m lab baseline       # frozen B0 baseline (~16 min on W7900-class)
.venv-lab/bin/python -m lab profile        # pipeline + rocprofv3 kernel profiling
.venv-lab/bin/python -m lab run --experiment L1 [--smoke]   # F1 F2 LF1-3 BEST H1 H2
.venv-lab/bin/python -m lab quality --experiment F2
.venv-lab/bin/python -m lab report [--scope reference|student|all] [--write]
#   (read-only by default; --kind smoke|all filters student smoke runs;
#    --write regenerates SUMMARY+charts — the reference scope then rewrites
#    the frozen evidence files, a deliberate maintenance action)
```

**GPU time is a metered student resource.** Classroom work uses bounded
`--smoke` variants (1–3 min); full protocol sessions cost 4–16 min each
(budget table in the [quickstart](docs/RADEON_CLOUD_QUICKSTART.md)). The
slow CPU experiments (H1/H2) are taught from the committed evidence and are
opt-in research tasks. Stop or destroy your instance when idle — `runs/`
persists with `/workspace`. Your measurements live under `runs/` and can
never modify the frozen reference evidence under `evidence/`.

## Evidence & claims

- `evidence/results/` — every run record (timing, VRAM, attention-path log
  evidence, image hashes), env snapshots, profiling analyses, SUMMARY +
  charts. The official campaign is pinned in
  [`lab/reference_runs.json`](lab/reference_runs.json); new runs land in
  `runs/` (isolated by code, aggregated with `report --scope student
  --write`; smoke runs are excluded from your formal medians by default).
- `evidence/quality/` — paired grids + native-resolution typography crops
  for the human verdict; `evidence/samples/` — B0 & BEST measured images.
- [docs/CLAIM_EVIDENCE_MATRIX.md](docs/CLAIM_EVIDENCE_MATRIX.md) — claims,
  each with its evidence file and caveats.
- [docs/case-study/](docs/case-study/) — the verbatim P0–P6 engineering
  journals (reading homework).
- [docs/FINAL_ENGINEERING_REPORT.md](docs/FINAL_ENGINEERING_REPORT.md) —
  executive summary, verdicts, recommendations.

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
