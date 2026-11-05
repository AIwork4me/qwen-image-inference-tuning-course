# Publication-Closure Release Gate Matrix (P7)

Verified state after all closure work. Every gate names its actual source
path or test — no gate is claimed from a Markdown summary alone.

| Gate | PR | Before | After | Source evidence | Test result | Blocking |
|---|---|---|---|---|---|---|
| PR #2 self-contained (owner-verdict evidence resolves from its own tree) | #2 | verdict record, provenance, INDEX remediation, 6 PNGs, scoped consumption, verdict tests lived only in PR #3 | all ported (commits `f49523a`, `166bfea`); PNG blobs identical to PR #3 | `docs/release-closure/PR2_OWNER_QUALITY_VERDICT.{json,md}`, `review-sheets/INDEX.json` git_blob fields | pytest 84/84 at ported head incl. 2 ported verdict tests; link-free of PR #3 paths | YES — resolved |
| Owner verdict correctly scoped, provenance honest | #2+#3 | record exists (PR #3) with branch-specific `recorded_by` | scope `M5.1 C1b deployment vs G_native` byte-matched by runner AND claims gate; port recorded in provenance | `PR2_OWNER_QUALITY_VERDICT.json` + `lab/m51_run.py::load_owner_verdict` | `test_owner_verdict_load_is_scope_matched`, `test_claims_rejects_owner_scope_mismatch`, record-drift test | YES — resolved |
| Numerical FAIL / NOT_RUN verdicts intact | #2+#3 | INT8 FAIL, uncompressed FAIL, sweep NOT_RUN, M6 BEST PENDING | unchanged, now enum-enforced + immutable | `course/data/verified_claims.json` v2.0.0 | `test_claims_rejects_int8_fail_promoted_to_pass`, NOT_RUN→PASS test, scope-leak tests | YES — pass |
| Six full-resolution review PNGs committed as Git blobs | #2 | only in PR #3 | identical blobs on both branches (`65676395 dacd2704 48a71df7 e6316b7a c538f20e 6ef0a4cb`) | `runs/m51-20261105-cleanroom-01/e2e/raw/*__cfg1.png` | `git rev-parse` equality check (closure session); INDEX digests | YES — pass |
| README time/throughput pairs same-population | #3 | 4 mixed pairs (2 EN + 2 ZH) | same-campaign pairs, cold/warm labeled | `P1_METRIC_PROVENANCE.json` populations | `check_metric_pairs.py` + 4 negative tests | YES — pass |
| No stale 1.04× presented as measured warm | #3 | M51_RESULTS §3.4 unlabeled | labeled as campaign estimate + measured successors | `docs/M51_RESULTS.md` §3.4 | manual + claims registry | YES — resolved |
| Strict status enums reject invalid/mixed strings | #3 | substring matching | exact 4-field enums, fail-closed cross-field | `scripts/course-quality/check_claims.py` | 15 negative tests (all fail as required) | YES — pass |
| Fair A/B physically measured or honestly BLOCKED | #3 | asymmetric C-fast vs A-original quote | 2×2 factorial measured: arch +1.9–2.6% under equal clients; monitor +5.4/+6.1%; n=6/cell, non-overlapping | `runs/p4-fairab-225326/pipeline_profile/pipeline_profile.json` | raw record + analysis recomputation in provenance registry | YES — measured |
| Bilingual tracks identical scientific conclusions | #3 | diagram mixing in both; exercise missing | repaired in both; fairness exercise + answer in both lessons; parity gates | READMEs, M5.1 lessons EN/ZH | `check_bilingual` (18 pairs), `check_claims` PARITY_VALUES, P5 audit | YES — pass |
| Local links resolve | #2+#3 | — | all tracked-markdown targets resolve | `scripts/course-quality/check_links.py` | gate green on both heads | YES — pass |
| Documented CLI examples valid | #3 | — | flags exist, bash parses | `check_examples.py` | gate green | YES — pass |
| CPU tests + course-quality gates on final heads | #2 | 82 passed at `a7fa69a` (CI 42032143208) | **84 passed** at new PR #2 head | `python -m pytest tests/ -v` | local run observed; GitHub CI observed post-push | YES — CI observed |
| CPU tests + 5 course-quality gates on final head | #3 | 101 passed at `4ec5905` (CI 42049486025) | **122 passed** + 5 gates at new PR #3 head | same | local run observed; GitHub CI observed post-push | YES — CI observed |
| Historical evidence immutable | both | trees frozen | `evidence/` = `33e735b7…` on main/PR2/PR3; cleanroom campaign additions-only (6 PNGs); `pr2-p4-*`, pipewarm, p2ab/p2profile trees identical | `P0_BASELINE.md` §3 | `git rev-parse` comparisons (this session) | YES — pass |
| PR descriptions match final commits | both | PR #2 said verdict pending (it was, at freeze); PR #3 said 100 tests | updated to final heads, exact test counts and CI links | PR bodies via API | observed post-edit | YES — updated |
| Dependency ordering correct | both | PR #3 stacked on PR #2 | preserved: base unchanged, merge brought PR #2 forward; PR #3 retarget to main only after PR #2 merges | git graph | `git merge-base` checks | YES — pass |
| No automated merge / false approval | both | — | both PRs Draft, OPEN, unmerged; no reviews submitted | GitHub API state | observed | YES — pass |
| Physical W7900 verification of the ported verdict path | #2+#3 | prior smokes at `a7fa69a` (PR #2 comments) | H3_OV_BF16 smoke at the merged closure head: owner verdict consumed, gates PASS, release no longer human-pending | `runs/pr2-port-verify-232500/` (new unique run ID) | run record + log | YES — pass |

## Known non-blocking items (owner decisions, not defects)

1. **M6 BEST quality verdict** — `PENDING_HUMAN_REVIEW` by design (owner gate;
   speed never overrides it).
2. **40-step BF16/CFG sweep** — `NOT_RUN` by design; VRAM claims explicitly
   not extrapolated to it.
3. **C2 INT8** — numerical FAIL, exploratory only, non-default forever unless
   new evidence.
4. **Persistent-monitor production adoption** — deliberately NOT applied in
   this closure (pre-registration discipline); P4 gives the owner the evidence
   base for a separate decision.
5. **weixin guide link** — unverifiable from the audit network (egress proxy);
   owner-provided URL kept, flagged in PR #3 limitations.
