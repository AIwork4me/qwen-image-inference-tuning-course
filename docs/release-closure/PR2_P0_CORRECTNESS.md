# PR #2 — Phase P2: P0 Correctness and Release-Gate Repairs

Audit HEAD at repair time: `6aacd6f` (pre-fix) → repairs committed on top (see P8 commit list).
All changes are surgical; frozen evidence untouched.

## P0-A. Execution success vs scientific success

**Before** (`lab/m51_run.py:71-73`): `"status": "PASS" if finite else "FAIL"` — a successfully
executed encoder with finite tensors reported `PASS`; no conditioning/quality/e2e/human gates
existed on this path; `honest_status()` was never wired in.

**After**: every H3 result now carries

```json
{"experiment": "H3_OV_BF16", "status": "PASS",
 "gates": {"environment": "PASS", "artifact_integrity": "PASS", "encoder_runtime": "PASS",
           "token_contract": "PASS", "conditioning_alignment": "PASS", "e2e_gpu": "NOT_RUN",
           "performance": "PASS", "human_quality": "PENDING"},
 "release_status": "PENDING_HUMAN_REVIEW"}
```

- `release_status()` (pure, `lab/m51_contract.py`): FAIL beats BLOCKED beats PENDING; the
  human gate is cleared only by an explicit human record (`settle_human_quality` with a
  `reviewer` field); missing evidence stays `NOT_RUN` and can never become PASS.
- The committed cleanroom numerical verdict is the truth-source per arm
  (`REFERENCE_VERDICT_KEYS`): a later reproducible INT8 run **cannot** promote the declared
  empty-negative failure — hash-repro AND committed verdict must both PASS.
- `e2e_gpu` is honestly `NOT_RUN` in H3 scope (E2E is a separate experiment), and `human_quality`
  is `PENDING`, so an encoder experiment can never emit `RELEASE_PASS` on its own.

**Proofs (tests, `tests/test_m51_contract.py`)**: `test_successful_encoder_is_not_numerical_pass`,
`test_int8_negative_failure_blocks_fidelity_claim`, `test_missing_evidence_cannot_become_pass`,
`test_blocked_experiment_never_upgraded_by_summary`, `test_human_review_not_auto_passed_by_numeric_checks`,
`test_h3_reference_verdicts_pin_the_declared_gates`, `test_release_status_rejects_unknown_gate_states`.

**Physical evidence**: `runs/pr2audit-114153-H3_OV_INT8-smoke/H3_RESULTS.json` — execution PASS
(encoder ran, conditioning hashes reproduce byte-identically) yet
`conditioning_alignment: FAIL` (inherited committed INT8 negative verdict) →
`release_status: RELEASE_FAIL`. Compare `runs/pr2audit-114034-H3_OV_BF16-smoke/H3_RESULTS.json`:
all measurable gates PASS → `PENDING_HUMAN_REVIEW` (not RELEASE_PASS).

## P0-B. Real artifact digest verification

**Before** (`lab/m51_contract.py::artifact_status`): presence + manifest JSON shape + "has a
digest field". `m51_run` copied `artifact_digest` from the manifest without reading a single
artifact byte.

**After**: `verify_artifact()` (pure stdlib) performs, at admission/provisioning time only:

- streaming SHA256 of XML and BIN (chunked; a 26 GB bin never enters RAM),
- size check against pinned sizes (truncation caught pre-hash),
- manifest must pin the SOURCE digest (chain anchor: fp32 manifest's `source_sha256` ==
  git-pinned TE checkpoint SHA; bf16/int8 manifests anchor to the fp32 XML digest),
- per-arm identity check (`fp32: dtype=fp32`; `bf16: storage=FP16 + runtime_precision=bf16`;
  `int8: mode=INT8_ASYM`) so a swapped artifact directory is BLOCKED,
- path safety: resolved files must live under the resolved root (dir symlinks OK; file-symlink
  escapes BLOCKED),
- legacy manifests without an XML pin produce `PASS_WITH_WARNINGS` with the computed digest
  recorded (never silent trust); `ov_compress.py` now pins `output_xml_sha256` for new exports,
- persistent verification cache is **opt-in** and keyed by (realpath, ino, size, mtime_ns,
  ctime_ns) — default is full re-verify per run because size+mtime alone is forgeable on
  coarse-granularity filesystems (demonstrated by test). Runtime policy: admission-time only,
  never per inference request.

**Proofs (negative tests)**: modified BIN, same-size modified XML, wrong digest, truncated file,
whole-dir swap (identity mismatch), missing/corrupt manifest, no-source-digest manifest,
symlink escape, unpinned-xml warning, cache invalidation on change —
`tests/test_m51_contract.py::test_verify_artifact_*` (9 cases).

**Physical verification (this audit, `/opt/m51-artifacts`)**:
`M51_OV_ARTIFACTS=/opt/m51-artifacts bash scripts/m51/provision_ov.sh --check-only` →

- fp32: **PASS** (xml `9fbbc226…` + bin `5681d9a6…` byte-verified, 26 GB streamed),
- bf16: **PASS_WITH_WARNINGS** — bin `62eee985…` verified; own XML digest unpinned in the
  legacy manifest, independently computed as **`28bd053ab3b1e3a83733da8023840e421d4e97b56c99558da7d9e509f537458d`**
  (recorded here and in every new H3 result; re-provisioning will pin it),
- int8: **PASS** (xml `511cb158…` + bin `330f3a53…` verified).

Runtime cost: 38 s for 45.5 GB on the audit host (once per run/provisioning, amortized; the
encoder itself never re-hashes).

## P0-C. Empty-negative token discrepancy — root cause and contract

**Observation**: frozen `conditioning_alignment.json` shows `token_ids_self_consistent=false`
for G and C0 while all cross-arm `token_ids_equal=true`.

**Root cause (verified live on the pinned ComfyUI runtime this audit)**:

| Text | production `clip.tokenize` (= `prevent_empty_text=False`, the pinned default at
`comfy/text_encoders/qwen_image21.py:18`) | diagnostic `prevent_empty_text=True` |
|---|---|---|
| `""` | 22 ids, **no** `220` | 23 ids, **inserts `220`** (literal word "empty") at index 17 |
| `" "`, `"  \n\t "` | 23 ids, identical under both policies | same |
| nonempty | identical | identical |

The OV encoder (`ov_encoder.py:95`, `prevent_empty_text=False`) **matches production**; the
`m51_baseline.py` manual *diagnostic* deviated (`prevent_empty_text=True`), so
`ids_equal=false` appeared only for the negative prompt, and `token_ids_self_consistent` in
`m51_alignment.py` compared non-equivalent semantics for G/C0 while being vacuously true for
OV arms. This was a **diagnostic artifact**, not a semantic mismatch reaching the encoder.

**Fixes**:

1. `lab/m51_baseline.py` — manual diagnostic now uses the production policy; the fallback
   policy is still recorded (`ids_manual_fallback_preventTrue`) for explanation, never used
   for conditioning. Historical records were NOT modified.
2. `lab/m51_token_contract.json` (new, committed) — canonical production token IDs for the
   4 frozen prompts + empty/whitespace/nonempty edge cases, generated from the pinned runtime.
   Every H3 run compares live probe IDs against it (`token_contract` gate) — tokenizer drift
   anywhere becomes a FAIL, not a hardcoded `true`.
3. `lab/m51_contract.py::empty_negative_explain()` — pure explainer of the fallback insertion.

**Proofs**: `test_token_contract_file_matches_frozen_production_ids` (contract file equals the
frozen campaign's production IDs; fallback differs by exactly one `220`),
`test_empty_negative_explain_fallback_insertion`, plus the live reproduction above
(`.venv-lab`, transformers 4.57.6, pinned ComfyUI tree).

**OpenVINO input semantics were NOT changed** — they were already correct; only the diagnostic
and its contract were repaired. The independent negative-conditioning accuracy gate (INT8
28.25% relL2 = FAIL) is untouched and now structurally unfakeable (P0-A).

## Result

All three P0 repairs implemented, tested (CPU-only, 82+ tests green), and validated on the
physical host. `docs/release-closure/PR2_P0_BASELINE.md` freeze manifest verified the frozen
evidence is untouched by these changes.
