# M5.1 Owner Image-Quality Verdict — Record of Provenance

**Verdict: PASS (acceptable), scope M5.1 C1b deployment vs G_native only.**
**Date of owner decision: 2026-11-05.**

## 1. What happened

The repository owner reported, in the course-v2 mission brief of 2026-11-05:

> "The human image-quality review is complete and the M5.1 result is acceptable."

Per the verdict protocol in
[`PR2_HUMAN_QUALITY_REVIEW_PACKAGE.md`](PR2_HUMAN_QUALITY_REVIEW_PACKAGE.md) §4,
an owner verdict is recorded as an explicit JSON record —
[`PR2_OWNER_QUALITY_VERDICT.json`](PR2_OWNER_QUALITY_VERDICT.json) — which the
gate mechanism (`lab/m51_contract.py::settle_human_quality`) consumes. This
page documents the provenance and scope of that record so a future auditor can
trace exactly what was accepted, by whom, and on what evidence.

## 2. Provenance (what this record is and is not)

| Field | Value |
|---|---|
| Reviewer | AIwork4me (repository owner) |
| Decision source | Owner-provided acceptance statement relayed verbatim in the course-v2 mission brief, 2026-11-05 |
| Recorded by | Autonomous engineering session on branch `feat/world-class-bilingual-inference-course` |
| NOT done | No GitHub Approve review was submitted; no handwritten signature was created; no image samples beyond the declared review package were inspected by the recording session |

The recording session did not view or judge the images itself; it maps the
owner's explicit acceptance statement onto the existing human-verdict mechanism
with full provenance. If the owner wants a stronger attestation (e.g. a commit
or review authored by the owner account), they can replace or amend the JSON
record — the mechanism accepts any explicit record with a reviewer and verdict.

## 3. Scope of acceptance

- **Arm:** C1b — OpenVINO CPU Text Encoder, **FP16-stored OpenVINO IR with BF16
  runtime execution**, replacing the native GPU Text Encoder.
- **Compared against:** G_native (native GPU Text Encoder).
- **Execution scope:** the already-measured campaign scope — turbo 8-step,
  1024×1024, cfg=1, seed 42, Qwen-Image-2.1 turbo BF16 DiT + original VAE,
  three course prompts (p1 中文场景, p2 English scene, p3 中文海报排版).
- **Reviewed artifacts:** the committed review sheets
  (`docs/release-closure/review-sheets/campaign_*_G_vs_C1b.jpg`,
  `crop_poster_title_*.jpg`) and, through them, the full-resolution
  SHA256-pinned G/C1b cfg1 PNG pairs from run `m51-20261105-cleanroom-01`
  (now committed in git; see `review-sheets/INDEX.json`).

## 4. Explicit non-propagation

This acceptance **does not** extend to:

1. **C2 (OpenVINO INT8)** — its empty-negative conditioning gate stays **FAIL**
   (relL2 28.25%); a subjective acceptance can never override a declared
   numerical failure.
2. **C1b-uncompressed (FP32-stored IR + BF16 execution)** — negative gate stays
   **FAIL** (6.63%).
3. **M6 BEST (6.12×, 55.21 s → 9.02 s)** — course-wide quality verdict remains
   independently pending under `runs/quality/BEST/quality.md`.
4. **40-step BF16/CFG sweep** — remains **NOT_RUN**.
5. Any workload outside the measured scope (other prompts, resolutions, step
   counts, seeds, cfg values, or hardware).

## 5. Gate effect

Re-running `python -m lab run --experiment H3_OV_BF16` after this record now
propagates `human_quality: PASS` (all other gates unchanged). `H3_OV_INT8`
runs keep `human_quality: PENDING` because the record's scope does not cover
them, and their `release_status` remains `RELEASE_FAIL` regardless, per the
`release_status` precedence (FAIL beats BLOCKED beats PENDING).
