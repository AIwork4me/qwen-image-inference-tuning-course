# PR #2 — Phase P6: Student-Facing Correctness and Lecture Readiness

## 1. Factual drift corrected (this audit)

| File | Before | After |
|---|---|---|
| `course/SYLLABUS.md:31` | M5.1 "same-stack W7900 A/B **NOT_RUN**" | COMPLETED 2026-11-05 with the honest scope note (encoder 331–480×, E2E steady ≈4%, 17.26 GB VRAM peak saved, INT8 negative gate FAIL → exploratory, human review PENDING) |
| `course/LECTURE_DEMO.md:108` | "not a measured result in this course … same-host A/B is still required" | measured-research-extension text with the same scope + explicit "Zen5/Zen4 stay historical cross-machine references" guard |

`README.md`, `README.zh-CN.md`, `course/M05_parallel_heterogeneous/LESSON.md`,
`M5_1_OPENVINO_CPU_TE.md`, and `docs/M51_*` were already consistent with the 2026-11-05
campaign (verified line-by-line this audit); no drift remained there.

## 2. Required instructional clarity — where each concept lives

1. **Why moving the TE to CPU first made things slower** — M5.0/LESSON.md (PyTorch CPU 136–207 s
   vs GPU 0.55 s); M5_1 lesson "C0" arm table.
2. **What OpenVINO actually optimized** — oneDNN BF16 graph execution on AVX512-BF16 Zen4,
   precomputed export graph, no per-op Python casts (`M5_1_OPENVINO_CPU_TE.md`; measured
   0.285–0.555 s).
3. **Why 331–480× encoder ≠ 331–480× images** — Amdahl box in `M51_RESULTS.md` §"break_even"
   and the pipeline evidence: throughput is bounded by ~12–13 s GPU denoise; a hidden CPU
   encoder buys at most ~4–8% warm (and 1.30–1.33× only in cold-inclusive sessions that
   include the recurring ~26 s GPU-CLIP reload).
4. **VRAM ↓ while CPU RAM ↑** — heterogeneous trade-off tables in M5_1 lesson + pipeline
   records (−17.26 GB GPU peak, +30 GB worker RSS for the BF16 deployment arm).
5. **Amdahl's law** — M5_1 lesson + `M51_RESULTS.md` break-even section.
6. **Local acceleration vs end-to-end value** — the pipeline arms B/C/D vs A comparison and
   the "SYSTEM_LEVEL_GAIN_SMALL" verdict teach exactly this.
7. **Why INT8 can be faster but fail fidelity** — C2 encodes fastest (0.144–0.406 s) yet the
   8-token empty negative diverges (relL2 28.25%, cos 0.9608); numbers in
   `M51_CONDITIONING_ALIGNMENT.md`; gate now structurally unfakeable (P0-A).
8. **Why human review stays mandatory** — `PENDING_HUMAN_REVIEW` in every summary + the
   review package (`PR2_HUMAN_QUALITY_REVIEW_PACKAGE.md`).

Measured vs derived vs pending vs failed are labeled throughout (`M51_CLAIM_EVIDENCE_MATRIX.md`
maps every claim to its raw record; the audit's `PR2_EVIDENCE_REDERIVATION.md` adds an
independent [RR]/[DR] classification).

## 3. Beginner path & bounded M5.1 access

- M0–M4 default startup is untouched (OpenVINO remains optional; full CPU test suite green,
  including all M0–M4 contract tests).
- Students are NOT required to export the 26 GB FP32 IR: the M5_1 lesson's evidence-reading
  path uses the committed run JSONs; a bounded smoke
  (`.venv-lab/bin/python -m lab run --experiment H3_OV_BF16 --smoke`) works with the 13 GB
  FP16-stored deployment IR only, and the audit adds honest gating so a blocked student sees
  `BLOCKED` + the exact provisioning command, never a fake PASS.
- Hosts without 55 GiB RAM / 60 GB disk get an explicit evidence-reading alternative
  (`M51_REPRODUCTION.md` already documents `M51_OV_ARTIFACTS` relocation; now consistent
  everywhere per P1-B).

## 4. Verdict

A graduate student can currently: reproduce the frozen numbers from raw records, run a bounded
physical smoke, interpret the Amdahl-limited system gain, and state exactly which claims are
measured (RR), derived, or pending. Remaining prerequisite for full closure: the owner's human
quality verdict (P5 package).
