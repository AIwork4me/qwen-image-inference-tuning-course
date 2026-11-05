# P3 — Scientific-Claim Gate Audit (Strict Status Semantics)

**Session:** publication-closure Phase P3, 2026-11-06.
**Gate:** [`scripts/course-quality/check_claims.py`](../../scripts/course-quality/check_claims.py)
**Registry:** [`course/data/verified_claims.json`](../../course/data/verified_claims.json) (v2.0.0)
**Tests:** `tests/test_course_quality.py` (18 claims-gate tests, 15 of them negative)

## 1. Finding (P0-C)

The pre-P3 validator enforced immutable verdicts by **substring search inside
free-form prose**:

```python
qs = str(c.get("quality_status", ""))
if "PASS" in IMMUTABLE.get(cid, ()) and "PASS" not in qs: ...
ok = any(a in qs for a in allowed) or qs == allowed[0]
if not ok and any(a in qs for a in ("PASS",)) and "FAIL" not in qs ...
```

Concrete evasions: `quality_status = "PASS (was FAIL)"` passes (contains FAIL,
so the promotion branch never fires); `"FAIL/PASS"` is neither exact nor
rejected; a sentence like "the sweep FAILED but positives PASS inspection"
reads as FAIL-family; a missing canonical field defaults to empty string and
skips guards. Statuses conflated measurement, numerical fidelity, human
quality and release decision in ONE field. Evidence paths were never checked
to exist.

## 2. New contract (exact enums, fail-closed, four separate fields)

Every registry claim now carries:

| Field | Enum |
|---|---|
| `measurement_status` | `MEASURED \| DERIVED \| ESTIMATED \| HISTORICAL \| NOT_RUN` |
| `numerical_verdict` | `PASS \| FAIL \| NOT_APPLICABLE \| NOT_RUN` |
| `human_quality_verdict` | `PASS \| FAIL \| PENDING_HUMAN_REVIEW \| OWNER_ACCEPTED_SCOPE_ONLY \| NOT_APPLICABLE \| NOT_RUN` |
| `release_decision` | `RELEASE_PASS \| RELEASE_FAIL \| RELEASE_PENDING \| BLOCKED \| NOT_APPLICABLE \| NOT_RUN` |
| `explanation` | free human prose — **never parsed** |
| `evidence_path` | first token must resolve inside the repo |
| `derivation` | required for `DERIVED`: `{numerator_ref, denominator_ref}` referencing real claim ids |

Legacy `quality_status` entries remain accepted only under a **strict
first-token rule**: exactly an enum value, or `ENUM (prose)` whose parenthetical
contains no other cross-family status token. Everything else — mixed, empty,
unknown, sentence — is rejected. The registry itself was migrated to canonical
fields (prose preserved in `explanation`).

## 3. Enforced semantics (each with a real failing test)

| Attack / mistake | Gate response | Test |
|---|---|---|
| INT8 numerical FAIL → PASS | immutable `numerical_verdict=FAIL` (canonical or legacy-parsed) | `test_claims_rejects_int8_fail_promoted_to_pass` |
| NOT_RUN experiment → verdict PASS | cross-field rule: `measurement_status=NOT_RUN` ⇒ every verdict and the release decision must be `NOT_RUN` | `test_claims_rejects_not_run_promoted_to_pass` |
| missing verdict field | `missing canonical field` | `test_claims_rejects_missing_quality_verdict_field` |
| empty / unknown enum value | `not an exact enum value` | `test_claims_rejects_empty_and_unknown_status` |
| mixed string `FAIL/PASS` | legacy parse rejected | `test_claims_rejects_mixed_status_strings` |
| sentence containing FAIL and PASS (`"PASS (was FAIL, promoted by owner)"`) | conflicting-token rejection | `test_claims_rejects_sentence_containing_fail_and_pass` |
| duplicate claim ids | `duplicate claim id` | `test_claims_rejects_duplicate_claim_ids` |
| owner-verdict scope widened in the claim | `verdict_scope must equal` the committed record scope | `test_claims_rejects_owner_scope_mismatch` |
| owner-verdict record itself scope-drifted | `owner record … scope mismatch` | `test_claims_rejects_owner_verdict_record_scope_drift` |
| M5.1 human verdict applied to INT8 | `scope leak` + `owner acceptance explicitly does NOT cover` | `test_claims_rejects_m51_human_verdict_applied_to_int8` |
| M5.1 human verdict applied to M6 BEST | same dual rejection | `test_claims_rejects_m51_human_verdict_applied_to_m6_best` |
| missing/unresolvable evidence path | `evidence_path does not resolve` | `test_claims_rejects_missing_source_evidence` |
| DERIVED claim without derivation | `requires derivation with numerator_ref and denominator_ref` | `test_claims_rejects_derived_claim_without_derivation` |
| DERIVED claim citing ghost references | `references unknown claims` | `test_claims_rejects_derived_claim_with_unknown_refs` |
| honest legacy prose `"PASS (numerics unchanged…)"` | accepted (backward compatibility) | `test_claims_accepts_honest_legacy_prose` |

The scope containment reads the **committed owner record**
(`docs/release-closure/PR2_OWNER_QUALITY_VERDICT.json`) itself: the claim's
`verdict_scope` and the record's `scope` must both equal `M5.1 C1b deployment
vs G_native`, and only the `M51-OWNER-QUALITY-VERDICT` claim may carry
`OWNER_ACCEPTED_SCOPE_ONLY`.

## 4. Latent registry defects the strict gate immediately caught

1. `NOISE-FLOOR.evidence_path` was `evidence/results/quality noise_floor.json`
   — the space-separated first token silently validated against the
   `evidence/results/quality` directory. Real path:
   `evidence/results/noise_floor.json` (fixed).
2. `TORCH-COMPILE.evidence_path` cited a nonexistent `P4_PREFLIGHT` file in
   the case-study directory (real rationale journal:
   `docs/case-study/P4_FASTER_COMPUTATION.md`; fixed).
3. `BEST-SPEEDUP` was `status: derived` with **no** machine-checkable
   derivation; now `derivation: {B0-WARM-MEDIAN, BEST-WARM-MEDIAN}`, and the
   previously implicit denominators were made explicit claims
   (`L1-WARM-MEDIAN` 12.132 s, `F1-WARM-MEDIAN` 45.086 s) so every derived
   ratio is recomputable from registered measurements.

No numerical value, scope statement or limitation was altered by the
migration — only status strings became enums, prose moved to `explanation`,
and two evidence paths were corrected to their true targets.

## 5. Bilingual precision

`check_claims.py` keeps the PARITY_VALUES presence check (critical values in
both language carriers) — deliberately not a global numeric diff, per the
mission. Claim IDs, experiment identities, units, warm/cold semantics and
quality statuses are additionally covered by `check_bilingual.py` structural
parity and by the P5 audit. All four (now five, with
`check_metric_pairs.py`) gates pass on the repaired tree:
`38 course-quality tests passed`, including every negative test above.

## 6. Verified verdict states after P3 (unchanged in substance)

| Experiment | measurement | numerical | human quality | release |
|---|---|---|---|---|
| M5.1 C1b (H3_OV_BF16) | MEASURED | PASS | OWNER_ACCEPTED_SCOPE_ONLY (scope-locked) | RELEASE_PENDING → owner-review release |
| M5.1 C2 INT8 (H3_OV_INT8) | MEASURED | **FAIL** | NOT_RUN | **RELEASE_FAIL** |
| C1b-uncompressed | MEASURED | **FAIL** | NOT_RUN | **RELEASE_FAIL** |
| M6 BEST | MEASURED | NOT_APPLICABLE | PENDING_HUMAN_REVIEW | RELEASE_PENDING |
| 40-step/CFG sweep | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
