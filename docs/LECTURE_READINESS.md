# Lecture Readiness Checklist — 2026-11-19, 16:30–18:10 (100 minutes)

*Gap analysis of what exists, what is missing, and what must be prepared
before the lecture. Honest inventory — items marked MISSING are not yet
built and are listed with the effort to close them.*

---

## A. What exists (verified against the repo)

- [x] Timed 100-minute agenda with clock ranges —
      [course/LECTURE_DEMO.md](../course/LECTURE_DEMO.md)
- [x] The four interactive questions with data-backed answer notes
      (Q1 where the time goes · Q2 why not 5× · Q3 why gains don't
      multiply · Q4 why overlap can lose) — same file
- [x] Teacher demonstration script (18 min, bounded smokes, with a
      no-hardware fallback plan) — same file
- [x] Student exercise (10 min, bounded, with worksheet acceptance) —
      same file
- [x] Slide-by-slide content with every number traced to committed
      evidence — same file (8 "slides" as structured text)
- [x] Reference imagery committed:
      `evidence/samples/{B0,BEST}/*.png` (9 each),
      `evidence/quality/*/quality_grid.png` +
      `quality_crops_poster.png` for 8 experiments,
      `evidence/results/chart_warm_latency.png`,
      `evidence/results/chart_waterfall.png`
- [x] Prerequisite policy: students arrange cloud access BEFORE the
      lecture (announced one week ahead) — stated in the lecture header
- [x] Budget discipline: demo and exercise use `--smoke` variants only

## B. What is MISSING (gap analysis)

| # | Gap | Impact | Effort to close | Owner decision |
|---|---|---|---|---|
| 1 | **No rendered slide deck** — LECTURE_DEMO.md is structured text, not a .pptx/.pdf deck | the lecturer must build slides from the outline | ~4 h (template + 8 slides, all content exists) | build from outline, or present the markdown directly with a projector split (live demo left, outline right) |
| 2 | **No committed kernel-class chart image** — slide 2's table is data (`rocprof-dit/analysis.json`), no pie/bar PNG | slide 2 is table-only | ~30 min (matplotlib from the committed analysis.json; acceptable as a student PR per M7 Task 2) | optional; the table carries the point |
| 3 | **No student worksheet file** — the exercise questions live inside LECTURE_DEMO.md only | students can't fill answers without a handout | ~30 min (extract the 3 questions + acceptance into `course/worksheets/M1_lecture_worksheet.md`) | recommended before the lecture |
| 4 | **No Amdahl-ceiling plot** — Q2's live computation is described but has no visual | minor (the live computation IS the pedagogy) | ~20 min | optional |
| 5 | **BEST quality verdict pending** — slide 7 must present it as pending, and it will remain pending unless the owner completes [QUALITY_REVIEW_CHECKLIST.md](QUALITY_REVIEW_CHECKLIST.md) before the lecture | slide 7 wording is ready either way; but a pre-lecture verdict would make the lecture's tradeoff section conclusive | owner: 20–30 min | strongly recommended before 11-19 |
| 6 | **Demo instance dry-run not yet scheduled** — the script assumes a pre-provisioned instance with `start.sh --all-models` run the night before | live-demo risk | ~30 min the evening before | required if presenting the live demo |

## C. Pre-lecture timeline (recommended)

| When | Action |
|---|---|
| 11-12 (one week ahead) | announce the cloud-access prerequisite; point students at `docs/RADEON_CLOUD_QUICKSTART.md`; assign M0 |
| 11-17 | confirm students' `start.sh` first-image results (course forum); identify who needs the evidence-reading track |
| 11-18 evening | demo instance: re-run `bash scripts/cloud/start.sh --all-models`, then `.venv-lab/bin/python -m lab doctor` (strict PASS) and one smoke of each demo command; verify projector/zoom setup |
| 11-19, 16:00 | instance booted, repo dir open, two terminals ready (`doctor`, `run`) |
| 11-19, 16:30 | lecture per the timed agenda |

## D. In-lecture risk mitigations

- Live demo fails (network/proxy hiccup) → fallback script is already
  written (committed grids + analysis JSON); practice the fallback once.
- Student exercise instances slow → the exercise is 2–3 min of GPU time;
  students without an instance use the evidence-reading variant (already
  specified).
- Time overrun → the agenda's flexible segment is the 18-min demo; slides
  6–7 compress to 5 min if needed (the combined table + tradeoff slide).
- Q&A runs long → the exercise doubles as the buffer (it runs while
  questions continue).
