# Course-v2 Final Release Audit (P8)

**Auditor stance:** this section reviews the branch
`feat/world-class-bilingual-inference-course` (stacked on PR #2 head
`a7fa69a`) as if written by another contributor. Audit date: 2026-11-05/06,
after all work landed.

## 1. Scientific correctness

| Item | Verdict | Evidence |
|---|---|---|
| All headline numbers traced to evidence | PASS | `SCIENTIFIC_CLAIMS_AUDIT.md` (23 claims re-verified; registry `course/data/verified_claims.json` + CI gate) |
| Metric drift fixed (1.04× → 1.33×/1.081×/1.029× taught separately) | PASS | M5.1 lesson §3.4–3.5; `PR2_LATENCY_VS_THROUGHPUT.json` |
| 331–480× scoped to positive prompts | PASS | M5.0/M5.1 lessons, SYLLABUS, lecture |
| 16.18 GB vs 17.26 GB definitions kept distinct | PASS | M5.1 lesson pair (both values, both definitions, non-interchange note) |
| INT8 negative FAIL preserved, structurally unfakeable | PASS | alignment verdict untouched; registry immutable-verdict test; owner-verdict scope test |
| BEST quality verdict independently scoped | PASS | M6 pair + READMEs state PENDING separately from the M5.1 owner acceptance |
| 40-step sweep NOT_RUN preserved | PASS | registry + limitations untouched |

## 2. Evidence integrity

| Check | Result |
|---|---|
| `evidence/` working tree vs P0 freeze (SHA256, 254 files) | **0 violations** |
| Frozen campaign `runs/m51-20261105-cleanroom-01/` + `evidence/` git blobs vs `a7fa69a` | **0 changed, 6 added** — the additions are exactly the six digest-verified full-resolution review PNGs (blob IDs recorded in `review-sheets/INDEX.json`) |
| Historical M5.1 measurements replaced? | **No** — `m51_pipeline_warm.py` / `m51_e2e.py` untouched; new study used a separate experiment module and run IDs |
| Owner-verdict provenance | explicit record (`PR2_OWNER_QUALITY_VERDICT.json` + provenance page); no fabricated signature/review; scope-matched consumption in `lab/m51_run.py` with negative tests |

## 3. Bilingual completeness

All nine teaching units (M0, M1, M2, M3, M4, M5.0, M5.1, M6, M7) have
complete EN + ZH lessons; SYLLABUS, METHODOLOGY, LECTURE_DEMO, both READMEs,
and all five student-onboarding docs are bilingual with substantive parity
(commands, experiment IDs, artifact references, structure) enforced by
`check_bilingual.py` in CI. Numerical parity of critical values enforced by
`check_claims.py`. READMEs navigate to the correct language versions.

## 4. Tests and CI

- CPU tests at course-v2 head `4ec5905`: **101 passed**
  ([CI run 42049486025](https://github.com/AIwork4me/qwen-image-inference-tuning-course/actions/runs/42049486025)).
  (An earlier revision of this audit said "100 tests" — stale metadata,
  corrected by the publication-closure P6 pass; the exact count always
  belongs to the exact HEAD.)
- CI: pytest + the course-quality gates (links / bilingual / claims /
  examples / metric pairs); CPU-only, no GPU claims.
- Standard cloud bootstrap (M0–M4): **no new mandatory dependencies** —
  OpenVINO remains optional behind `scripts/m51/provision_ov.sh`.

## 5. Physical W7900 validation

Recorded in `runs/overnight-course-v2-a7fa69a-p2*` and
`docs/quality-v2/PIPELINE_OVERHEAD_STUDY.md`: instrumented profiling + a
controlled interleaved A/B (correctness contract: conditioning SHA256
stable). Result: +4.85% batch throughput (below the pre-registered 5%
exploratory target), root cause attributed to per-request monitor teardown;
production implementation deliberately unchanged. GPU budget used: ~23 min
(of 3 h allowed).

## 6. Repository hygiene

- No model weights or large artifacts added beyond the 6 review PNGs
  (~9.2 MB, justified and hash-pinned); total branch delta +9,827/−269
  lines across 71 files.
- Link integrity gate: all local targets resolve (1 real broken link caught
  and fixed during the audit).
- Licenses/attribution unchanged (MIT code, CC-BY-4.0 texts).

## 7. Remaining limitations (honest)

1. C0 encoder A/B p1/p3 remain single-session (pre-existing, documented).
2. The P2 fast-client result is n=2 interleaved batches per mode (small
   sample, stated in the study).
3. The weixin guide link could not be verified from the audit network
   (egress proxy 403 on qq.com domains); owner-provided URL retained.
4. M6 BEST human verdict still pending (owner action, independent gate).
5. Lecture slide deck (pptx) not rendered — LECTURE_DEMO is the canonical
   package (documented in `LECTURE_READINESS.md`).

## 8. Release recommendation

**READY_FOR_OWNER_REVIEW.**

The content is correct, substantive, navigable and testable; all automated
gates pass; evidence is immutable; provenance of the one human verdict is
explicit and scope-limited. Remaining items are owner decisions (merge PR #2
first, then retarget this stacked PR to main; optionally the M6 BEST
verdict), not engineering defects.
