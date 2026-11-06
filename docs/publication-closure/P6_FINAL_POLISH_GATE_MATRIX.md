# P6 — Final Polish Gate Matrix

**Session:** final reproducibility & publication polish, 2026-11-06.
Classification vocabulary: `PASS | PASS_WITH_LIMITATIONS | FAIL | BLOCKED | NOT_RUN`.
Every row names its evidence path or test — no gate is claimed from prose alone.

| # | Gate | Before | After | Evidence | Test | Blocking | Result |
|---|---|---|---|---|---|---|---|
| 1 | Artifact verification command validity | P4 guide taught the invalid form of feeding a JSON manifest to the sha256sum check mode (always fails; JSON ≠ GNU checksum list) | official `M51_OV_ARTIFACTS=<root> bash scripts/m51/provision_ov.sh --check-only` documented EN+ZH; executed → `ARTIFACTS_VERIFIED` | `P6_ARTIFACT_VERIFICATION_CLOSURE.md`; `P4_FAIR_CLIENT_AB.md` §6; both M5.1 lessons | `test_examples_reject_sha256sum_check_on_json_manifest` (+3 more) | YES | **PASS** |
| 2 | Artifact digest integrity | admitted per-run by `verify_artifact` (untested doc-level story) | all three arms byte-verified this session: fp32 PASS, int8 PASS, bf16 PASS_WITH_WARNINGS (unpinned XML digest computed & disclosed — provisioning-tool gap, artifacts untouched) | `P6_ARTIFACT_VERIFICATION_CLOSURE.md` §2–3 | existing validator matrix in `tests/test_m51_contract.py` (reused: digest/truncate/missing/corrupt-JSON/identity/symlink) | YES | **PASS_WITH_LIMITATIONS** (bf16 XML digest not pinned by the export tool; disclosed, not hidden) |
| 3 | Historical source provenance honesty | P4 campaign ran dirty; HEAD alone ≠ executed bytes; nothing documented | forensic audit: `SOURCE_RECONSTRUCTED_WITH_EVIDENCE` (pre-commit pyc structural identity + size identity; unexplained 23:23:18 write event documented); never upgraded to EXACT | `P6_SOURCE_PROVENANCE_AUDIT.md` | bytecode-structure comparison (in-audit); `test_dirty_tracked_file_detected_with_changed_digest` etc. | YES | **PASS** |
| 4 | Future run source-byte identity | no source manifest in run records | `lab/source_provenance.py`: per-module SHA256/status, tracked-diff digest, exact-byte snapshots, fail-closed `provenance_class`; proven on hardware → `EXACT_SOURCE_VERIFIED` | `runs/p4-fairab-cleanhead-003750/…json` `environment.source_provenance` | `tests/test_source_provenance.py` (8 cases: clean/dirty/untracked/missing/determinism/no-secrets) | YES | **PASS** |
| 5 | Clean-HEAD physical regression | original factorial only from dirty worktree | bounded rerun from committed `edc89508`, 2 batches/cell, 11.2 min GPU: every effect reproduces in direction & scale (3 cells ≤0.2%, one 1.3%); `CLEAN_HEAD_PHYSICAL_REGRESSION_PASS` | `P6_CLEAN_HEAD_PHYSICAL_VALIDATION.md`; raw JSON committed | raw record + rederivation table | optional | **PASS_WITH_LIMITATIONS** (smoke-scale n=2; original keeps its own label) |
| 6 | P4 raw-statistics rederivation | published values unverified independently | fresh reimplementation from committed JSON reproduces every mean/median/range/throughput/effect exactly | `P6_P4_INDEPENDENT_REDERIVATION.md` | in-audit recomputation vs blob `7d100108…` | YES | **PASS** |
| 7 | Fair-client A/B attribution | effects unlabeled by evidence class | every attribution tagged MEASURED/DERIVED/HYPOTHESIS/NOT VALIDATED; no significance inflation | `P6_P4_INDEPENDENT_REDERIVATION.md` §5 | same | YES | **PASS** |
| 8 | Cold/warm population separation | (fixed in prior closure; verified still green) | README pairs same-population; provenance registry authoritative | `P1_METRIC_PROVENANCE.json` | `check_metric_pairs.py` gate + negative tests | YES | **PASS** |
| 9 | M5.1 owner verdict scope | (established; unchanged this pass) | C1b-vs-G only; runner scope-match; claims-gate containment | `PR2_OWNER_QUALITY_VERDICT.json` | `test_claims_rejects_owner_scope_mismatch`, `…_applied_to_int8`, `…_applied_to_m6_best` | YES | **PASS** |
| 10 | INT8 numerical FAIL preservation | (established; unchanged) | relL2 28.25% FAIL immutable, enum-enforced, hardware-reverified (prior session smoke) | `course/data/verified_claims.json` | `test_claims_rejects_int8_fail_promoted_to_pass`; `runs/pr2-port-verify-int8-…/H3_RESULTS.json` | YES | **PASS** |
| 11 | M6 BEST pending preservation | (established; unchanged) | `PENDING_HUMAN_REVIEW`; NOT settled by M5.1 acceptance | registry `M6-BEST-QUALITY` | NOT_RUN-can-hold-no-verdict cross-field test | YES | **PASS** |
| 12 | README EN/ZH consistency | no changes this pass; parity verified | identical scientific conclusions, gates green | READMEs | `check_bilingual`, `check_claims` PARITY_VALUES, `check_metric_pairs` | YES | **PASS** |
| 13 | M5.1 lesson EN/ZH consistency | reproduction guide absent from lessons | verified bilingual "reproduce the fairness experiment" panels (identical commands/numbers/limits; CLI identifiers untranslated) | `M5_1_OPENVINO_CPU_TE.md` / `.zh-CN.md` | `check_bilingual` structure parity; `check_examples` command validity both files | YES | **PASS** |
| 14 | Documented command correctness | one invalid command shipped; path drift unguarded | admission command executed on hardware; every scanned doc's command paths resolve; bash blocks parse; checksum forms validated | `P6_ARTIFACT_VERIFICATION_CLOSURE.md`; lessons; P4 guide | `check_examples.py` (+ new checks) incl. `test_examples_accept_legitimate_sha256_usage`, `…_reject_broken_command_path` | YES | **PASS** |
| 15 | CPU tests | 122 at `6503e8d` | **134 passed** at final head (122 + 8 provenance + 4 example-gate tests) | local run observed | `python -m pytest tests/ -v` | YES | **PASS** |
| 16 | Course-quality gates | 5 gates | all 5 green at final head (links / bilingual / claims / examples / metric-pairs) | local run observed | each gate script exit 0 | YES | **PASS** |
| 17 | GitHub Actions | run 42130732184 (122 passed) at `6503e8d` | observed post-push at the final pre-matrix head; URL recorded in §Observation | GitHub Actions log inspected | run log grep of test count | YES | **PASS** (see §Observation) |
| 18 | Historical evidence immutability | frozen | `evidence/` = `33e735b7…` (unchanged since main); cleanroom `93cbd6a7…`; original P4 tree `eecb46de…` + JSON blob `7d100108…` byte-identical | `git rev-parse` comparisons this session | same | YES | **PASS** |
| 19 | PR #2 dependency isolation | PR #2 @ `532b501` self-contained | untouched this pass (no PR-#2-branch commits); PR #3 still stacked, merge-base = `532b501` | git refs | `git merge-base` | YES | **PASS** |
| 20 | PR #3 merge-readiness | Draft @ `6503e8d`, CI green | Draft, MERGEABLE/CLEAN, base = PR #2 branch, CI green at final head; description updated with this pass | GitHub PR API | `gh pr view` state | YES | **PASS** |

## Observation — GitHub Actions (gate 17)

Observed as part of this task at final head `b4ed3a6`:
[run 42140170241](https://github.com/AIwork4me/qwen-image-inference-tuning-course/actions/runs/42140170241)
→ **134 passed** + all five course-quality gates OK (link / bilingual / claims /
examples / metric-pair), confirmed by reading the run log, not the badge.
Commits after `b4ed3a6`, if any, are documentation deltas on top of that
verified head and their CI runs are observed by the same standard.

## Release verdict

All 20 gates PASS or PASS_WITH_LIMITATIONS (limitations explicitly disclosed:
bf16 XML digest unpinned by the export tool; smoke-scale regression sample;
original campaign's reconstructed-source label retained). Zero FAIL, zero
BLOCKED, zero NOT_RUN among blocking gates. **READY_FOR_OWNER_REVIEW.**
