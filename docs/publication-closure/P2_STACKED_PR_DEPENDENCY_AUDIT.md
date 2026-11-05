# P2 — Stacked-PR Dependency Audit & Repair

**Session:** publication-closure Phase P2, 2026-11-06.
**Question:** can PR #2 be reviewed and merged as a **self-contained** M5.1
release candidate, without any dependency on PR #3?

## 1. Finding (P0-B) — the dependency inversion

At freeze (see [`P0_BASELINE.md`](P0_BASELINE.md)) PR #2 contained the M5.1
physical implementation and its release audit, but five pieces of its own
release closure lived only in the stacked PR #3:

| Piece needed by PR #2 | Lived only in PR #3 |
|---|---|
| Owner human-quality verdict record | `docs/release-closure/PR2_OWNER_QUALITY_VERDICT.json` |
| Verdict provenance page | `docs/release-closure/PR2_OWNER_QUALITY_VERDICT.md` |
| Review-package remediation (committed-vs-host-only truth) | `docs/release-closure/PR2_HUMAN_QUALITY_REVIEW_PACKAGE.md` (P0-D rewrite) |
| Review-sheet INDEX remediation (git blobs, provenance notes) | `docs/release-closure/review-sheets/INDEX.json` |
| Six SHA256-verified full-resolution G_native/C1b PNGs | `runs/m51-20261105-cleanroom-01/e2e/raw/*.png` (force-added) |
| Scoped verdict consumption in the runner | `lab/m51_run.py` (`load_owner_verdict`, H3_OV_BF16-only) |
| Verdict-scope & numerical-precedence tests | `tests/test_m51_contract.py` (+2 tests) |

Consequence: merging PR #2 alone would have produced a `main` whose M5.1
release gate (`human_quality`) stays PENDING with the verdict record missing,
and whose review package claims committed images that a fresh checkout cannot
resolve.

## 2. Repair — surgical port to `feat/m5-1-openvino-cpu-te-course`

Applied on top of PR #2 head `a7fa69ae` as **two focused commits** (no
wholesale cherry-pick; no curriculum content):

1. `fix(m51): close owner verdict and image evidence in PR2` — verdict JSON +
   provenance page + review-package rewrite + INDEX remediation + the image
   evidence audit page (`docs/quality-v2/IMAGE_EVIDENCE_AUDIT.md`, referenced
   by the review package, hence required for PR #2 link integrity) + the six
   PNGs (identical git blobs to PR #3's: `65676395`, `dacd2704`, `48a71df7`,
   `e6316b7a`, `c538f20e`, `6ef0a4cb`).
2. `test(m51): enforce scoped human and numerical release gates` —
   `lab/m51_run.py` scoped consumption + `tests/test_m51_contract.py` verdict
   tests (scope-matched load; INT8 never rescued; malformed/wrong-scope
   records keep PENDING).

Two provenance lines were updated **honestly** during the port (content
otherwise byte-identical): `recorded_by` in the verdict JSON and the
"Recorded by" row in the provenance page now state that the record was made
2026-11-05 on the course-v2 branch and ported unchanged to the M5.1 branch by
this dependency repair — so neither branch's history makes a false claim about
where the bytes originated.

Deliberately NOT ported (stay in PR #3, out of PR #2's scope):
`lab/m51_pipeline_profile.py`, `tests/test_course_quality.py`,
`scripts/course-quality/*`, all curriculum/README/quality-v2 course-audit
content beyond the one image-evidence page required by PR #2's own links.

## 3. Gate semantics preserved exactly (no widening)

- `settle_human_quality`: a human PASS can settle ONLY `human_quality`; the
  `release_status` precedence stays `FAIL > BLOCKED > PENDING` — a numerical
  FAIL (C2 INT8 relL2 28.25%, C1b-uncompressed 6.63%) is unreachable by any
  human record.
- `load_owner_verdict`: applies only to `H3_OV_BF16` **iff** the committed
  record's `scope` equals `M5.1 C1b deployment vs G_native` byte-for-byte;
  missing file / malformed JSON / wrong scope / missing reviewer → `None` →
  gate stays PENDING (fail-closed).
- Experiment execution status, historical validation, current-run evidence
  and overall release status remain separate fields in the run records.
- The verdict record itself remains an owner-relayed statement with explicit
  non-coverage (INT8, uncompressed, M6 BEST, 40-step sweep) — no GitHub
  Approve, no signature, no claim that the recording session viewed images.

## 4. Stack re-wiring (safe merge, no rebase, no force)

1. PR #2 branch received the two commits above → new PR #2 head (CI observed
   before push-out; see §5).
2. PR #3 branch merged the updated PR #2 branch with a plain `git merge`
   (common ancestor `a7fa69ae`). Conflicts resolved in favor of the identical
   ported files (they are byte-identical to PR #3's, so the merge is a
   no-change for PR #3's content) — bilingual work untouched.
3. PR #3 stays based on `feat/m5-1-openvino-cpu-te-course`; it must be
   retargeted to `main` only after the owner merges PR #2.
4. `main` untouched; both PRs remain Draft and unmerged; no history rewritten
   anywhere.

## 5. Verification

- PR #2 CPU suite at the ported head: `python -m pytest tests/ -v` → all pass
  (84 tests: 82 pre-existing + 2 ported verdict tests; exact count recorded
  for the exact HEAD in the PR #2 closure comment).
- Scope regression on PR #2: the ported tests exercise (a) scope-matched load,
  (b) INT8 RELEASE_FAIL not rescuable by an owner PASS, (c) malformed and
  wrong-scope records → PENDING.
- PR #3 after the merge: full suite + all five course-quality gates green
  (see [`P0_BASELINE.md`](P0_BASELINE.md) §2 for the run-link convention).
- Evidence integrity: `evidence/` tree hash `33e735b7…` identical on all refs;
  cleanroom campaign tree unchanged except the six PNG additions (already
  true pre-port on PR #3; now also true on PR #2).

## 6. Result

PR #2 is now independently understandable, independently verifiable and
independently mergeable: its human-quality gate resolves from files it itself
carries, its review evidence resolves from git blobs it itself contains, and
nothing in it references PR #3 content. PR #3's incremental diff over the new
PR #2 base is pure curriculum.
