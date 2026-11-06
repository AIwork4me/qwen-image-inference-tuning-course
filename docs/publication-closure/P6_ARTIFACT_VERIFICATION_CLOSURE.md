# P6 — Artifact Verification Closure (SHA256 Command Defect)

**Session:** final reproducibility & publication polish, 2026-11-06.

## 1. The defect

`docs/publication-closure/P4_FAIR_CLIENT_AB.md` §6 (as committed at
`6503e8d`, blob `55c78d96…`) taught students:

```bash
sha256sum -c /opt/m51-artifacts/bf16/conversion_manifest.json # quoted-defect
```

This is **invalid**: `conversion_manifest.json` is a structured JSON document
(`source_xml_sha256`, `output_bin_sha256`, `bin_bytes`, precision identity …),
not a GNU coreutils checksum list (`<digest>  <filename>` lines). Executing
the command fails with `no properly formatted SHA256 checksum lines found`.
A student following the guide on a correctly provisioned machine would hit an
error from the *verification step itself* — release-blocking for teaching
material. (Related prose in §2 of the same doc — "(`sha256sum -c` → OK)" —
described the session's ad-hoc herestring check and was likewise replaced.)

## 2. The supported verification path (inspected, then executed)

The repository's established entrypoint is
[`scripts/m51/provision_ov.sh --check-only`](../../scripts/m51/provision_ov.sh),
which performs admission via
[`lab.m51_contract.verify_artifact`](../../lab/m51_contract.py):

| Check (mission list) | Implementation | Verified how |
|---|---|---|
| Expected artifact root | `$M51_OV_ARTIFACTS` else `<repo>/runs/m51-ov-artifacts`; missing root → `BLOCKED/missing_artifact_root` | code read + existing tests |
| XML existence & content | `required` file list per arm; missing → `missing_artifacts`; digest pinned/compared where the manifest pins it | code read + tests (`missing`, `modified_xml_blocked…`) |
| BIN existence & content | same mechanism | code read + tests |
| SHA256 digest comparison | streaming SHA256 vs manifest pin; mismatch → `digest_mismatch` | code read + tests + real run below |
| Expected byte size | manifest `*_bytes` fields; truncation → `size_mismatch` (fires before hashing) | code read + truncation test |
| Conversion identity | `expected_identity` (`storage/runtime_precision/dtype/mode`) vs manifest fields → `identity_mismatch` | code read + identity tests |
| Artifact path safety | resolved files must stay under resolved root; symlink escapes → `unsafe_path` | code read + symlink-escape test |
| Manifest format validity | JSON parse; corrupt → `corrupt_manifest`; no source digest → `manifest_has_no_source_digest` | code read + tests |

Also covered by the script before the artifact loop: the **pinned source text
encoder** is verified (`lab.weights.verify_weight`) — the chain anchor.

**Executed this session** on the physical host (real 46 GB artifact set):

```bash
export M51_OV_ARTIFACTS=/opt/m51-artifacts
bash scripts/m51/provision_ov.sh --check-only
```

Observed output: source TE `ok: True` (pinned digest
`68bdc82b…`); `fp32 PASS`; `bf16 PASS_WITH_WARNINGS` (see §3);
`int8 PASS`; final line `ARTIFACTS_VERIFIED`, exit 0. Duration ≈ 2 min
(≈ 46 GB streamed). No downloads, no writes to artifacts (check-only exits
before pip/export; its only side effect is `mkdir -p` of an artifact root
that already exists).

## 3. Honest disclosure — the bf16 `PASS_WITH_WARNINGS`

The bf16 arm's manifest pins the SOURCE xml digest and the OUTPUT bin digest
+ size, but not its own output-XML digest, so the verifier computes and
discloses that digest with a warning (`manifest pins no digest … re-provision
to pin it`) rather than silently passing. This is the designed fail-visible
behavior (test: `test_verify_artifact_unpinned_xml_warns_not_silent`). The
manifest was NOT hand-edited to add the pin: editing a provenance anchor after
the fact would weaken, not strengthen, the chain (and the artifacts are
byte-frozen). A future provisioning-tool change can pin output-XML digests at
creation time; out of scope here.

## 4. Negative-test coverage (existing suite — reused, not duplicated)

`tests/test_m51_contract.py` already contains the exact matrix the mission
requires (verified by name and assertion this session): valid digest pass
(`test_verify_artifact_pass_and_cache`), invalid digest
(`…_incorrect_digest_blocked`, `…_modified_bin_blocked`), modified XML with
valid size (`…_modified_xml_blocked_even_with_valid_size`), truncated file
(`…_truncated_file_blocked`), missing artifacts
(`test_artifact_absence_and_corruption_are_blocked`), invalid JSON
(`corrupt_manifest` assertions), unexpected identity
(`…_swapped_artifact_identity_blocked`, `test_h3_specs_pin_identity_and_budget`),
plus symlink escape, same-size tamper, opt-in cache invalidation, and
`M51_OV_ARTIFACTS` portability (`test_artifact_root_env_portability`).
All run in CPU CI with small synthetic XML/BIN fixtures — full-model hashing
never runs in CI.

## 5. Repaired documentation (this pass)

- `P4_FAIR_CLIENT_AB.md` §6: explicit warning against the invalid form +
  the official three-step workflow (admission via `--check-only`, clean-worktree
  check, run command), expected outputs, budget, bounded-regression variant,
  recovery, and limits; `/opt/m51-artifacts` labeled as an **example** root.
- `P4_FAIR_CLIENT_AB.md` §2: admission line rewritten to describe the real
  streaming-manifest check (no `sha256sum -c` prose).
- `P0_BASELINE.md` line about the prior session's herestring check is a
  historical freeze record (its form — piping a digest line — is valid
  coreutils usage and is not presented as a copyable command); left untouched
  per the no-history-rewrite rule.
- Bilingual reproduction guides (M5.1 lessons EN/ZH) updated with the same
  verified commands — see `P6_FINAL_POLISH_GATE_MATRIX.md` gate 14 and the
  P5 section of the final report.
- `check_examples.py` hardened so the invalid pattern
  (`sha256sum -c` applied to a `.json` manifest) can never re-enter <!-- quoted-defect -->
  documented examples (negative-tested).

## 6. Acceptance check

- A student on a correctly provisioned machine copies §6 verbatim and the
  admission step prints `ARTIFACTS_VERIFIED` (executed this session).
- Corrupted artifacts fail clearly (block reasons above; test-enforced).
- The command does not modify or redownload verified artifacts (code path +
  observed run).
