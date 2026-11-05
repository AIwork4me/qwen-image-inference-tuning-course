# Course-v2 Image-Evidence Accessibility Audit (P0-D)

2026-11-05/06. Method: `git ls-files` + `git cat-file -e HEAD:<path>` for every
`committed_in_git` claim in `docs/release-closure/review-sheets/INDEX.json`,
plus `sha256sum` of working-tree copies against recorded digests.

## 1. Finding (before remediation)

| Claim | Reality | Root cause |
|---|---|---|
| `INDEX.json` recorded **6 PNGs** under `runs/m51-20261105-cleanroom-01/e2e/raw/` as `committed_in_git: true` | **6/6 NOT in git** — `git cat-file` MISSING for all six | `.gitignore` rule `runs/**/raw/` excludes them; they were never force-added |
| `PR2_HUMAN_QUALITY_REVIEW_PACKAGE.md` §1 table headers said "Reference (committed in git)" / "C1b deployment (committed in git)" for those PNGs | Inaccurate | inherited from INDEX |
| Same doc: "Also committed: C1a FP32 and C2 INT8 full-res pairs" | Inaccurate (host-only) | inherited |
| `INDEX.json` recorded **8 PNGs** under `runs/pr2-p4-pipewarm-120822/.../raw/` as `committed_in_git: false` with "audit host only; regenerate via lab.m51_pipeline_warm" | Honest and accurate; local digests match 8/8 | — |

Working-tree SHA256s matched the recorded digests **6/6** — the provenance data
was correct; only the availability claim was wrong. All 27 campaign PNGs exist
on the measurement host; 21 (C1a/C2/G_precomputed/C0/cfg2/warmup) had no INDEX
digest entries (they were secondary artifacts).

## 2. Remediation (this PR)

Per the mission rule *"Never call a host-only image committed"*, and because
these exact six files are the owner-review scope, the honest minimal fix was to
**make the claim true rather than weaken the record**:

1. The six digest-verified full-resolution PNGs (G_native + C1b, prompts
   p1/p2/p3, cfg1) were **force-added to git** past `runs/**/raw/`
   (≈9.2 MB total). Byte identity is guaranteed by the pre-existing SHA256
   records and now also by git blob IDs added to `INDEX.json`
   (`git_blob` field per file).
2. `INDEX.json` now carries a `provenance_note` on each remediated record
   documenting the original false claim and the repair.
3. `PR2_HUMAN_QUALITY_REVIEW_PACKAGE.md` §1 was rewritten: the six core PNGs
   are genuinely committed; **C1a/C2 cfg1 pairs, all cfg2 variants, and
   G_precomputed/C0 pairs are explicitly labeled host-only** with their exact
   regeneration path (`python -m lab.m51_e2e`, pinned weights, seed 42).
4. `evidence/` images referenced by the READMEs were re-verified: all tracked
   (`evidence/samples/{B0,BEST}/*`, `evidence/quality/BEST/*`).

## 3. Convention note

`runs/**/raw/` remains gitignored for **student and future** runs (hashes +
analyses are the committable record). The six force-added files are a one-time
remediation of the owner-review artifacts, justified because the review record
(`PR2_OWNER_QUALITY_VERDICT.json`) must point at accessible evidence. New
campaigns must not use this as precedent without recording digests and an
explicit justification in their INDEX.

## 4. Verdict

After remediation, every availability claim in `INDEX.json` and the review
package is **VERIFIED** against the git tree; no host-only image is called
committed; image provenance (SHA256 + git blob) is preserved and strengthened.
