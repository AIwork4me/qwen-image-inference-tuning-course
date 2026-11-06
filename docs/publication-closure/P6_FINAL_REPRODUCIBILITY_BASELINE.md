# P6 — Final Reproducibility Baseline (Pre-Polish Freeze)

**Session:** final reproducibility & publication polish, 2026-11-06 (UTC).
**Purpose:** freeze the exact identities that this polish pass must preserve,
before any edit.

## 1. Git state (verified against the live GitHub API; git fetch hit
transient proxy 503s, refs re-confirmed via `gh`)

| Ref | Commit | Note |
|---|---|---|
| `main` | `767071af4c399c65732189ab77178d247a78ee81` | untouched |
| PR #2 head `feat/m5-1-openvino-cpu-te-course` | `532b5017c3ed547766bd7078bf88dd1592ca438f` | OPEN, Draft, CI 84 passed (run 42130299278) |
| PR #3 head `feat/world-class-bilingual-inference-course` | `6503e8dcd6c0a78b6acf40b6757b73215a8436da` | OPEN, Draft, MERGEABLE, base = PR #2 branch, CI 122 passed + 5 gates (run 42130732184) |

`git merge-base(PR#3, PR#2) = 532b501` — stacking intact. Worktree at freeze:
tracked files clean (`git status --short` shows only untracked host-side
leftovers: the failed first-launch run `p4-fairab-224922` (host-side trail) and the
port-verify probe/stdout/stderr logs — retained untouched, none tracked).

## 2. Protected identities (frozen at `6503e8d`)

| Object | Git blob / tree |
|---|---|
| P4 raw record `runs/p4-fairab-225326/pipeline_profile/pipeline_profile.json` | blob `7d100108347279b8c478250bc9b6f7bb9188a8cb` |
| P4 report `docs/publication-closure/P4_FAIR_CLIENT_AB.md` | blob `55c78d96e5c3f0bca6de4b4e9d9ac70db8b85001` (this pass edits only its §2 admission line and §6 reproduction section — a documentation repair, not a result change; all numbers immutable) |
| `lab/m51_pipeline_fair_ab.py` | blob `5f6611271d03d8e2fc2ad1a342a266574736a8a2` |
| `lab/m51_pipeline_profile.py` | blob `e0e8f5a77fab0fceea6071fb2837c5c5f76c6025` |
| `evidence/` tree | `33e735b7837a5b51e5cbf55ff5718b5e81244a2b` (identical on main/PR2/PR3) |
| `runs/m51-20261105-cleanroom-01/` tree | `93cbd6a7a82c8a728b829a6ac21c201ae2dfd6d6` |
| `runs/p4-fairab-225326/` tree | `eecb46de5d7755bac093862ec3657bbf279432d4` |
| `docs/quality-v2/` tree | frozen; FINAL_RELEASE_AUDIT edited only in its stale-reference lines per §P7 scope |

## 3. Binary artifact identities (admission anchors)

`/opt/m51-artifacts` (example root on this host; `M51_OV_ARTIFACTS` points
here for every M5.1 run in this environment):

- `bf16/conversion_manifest.json`: `source_xml_sha256 9fbbc226…`,
  `output_bin_sha256 62eee98599e0e004f6b8a29bfea1f61e7c7bf2c9d0b9ed505cdf2a0e2eb380ce`,
  `bin_bytes 13,892,143,718` — matches the frozen campaign record byte-for-byte.
- Official admission (`bash scripts/m51/provision_ov.sh --check-only`) executed
  this session against these artifacts → fp32 `PASS`, bf16
  `PASS_WITH_WARNINGS` (unpinned XML digest — computed and disclosed, see
  `P6_ARTIFACT_VERIFICATION_CLOSURE.md`), int8 `PASS`, `ARTIFACTS_VERIFIED`.

## 4. The two defects this pass exists to close

1. **Invalid verification command in teaching material:** P4 report §6
   instructed students to run `sha256sum -c <manifest>.json` — a JSON manifest <!-- quoted-defect -->
   is not a GNU sha256sum checklist; the command fails outright. Repaired to
   the official `provision_ov.sh --check-only` workflow (P1).
2. **Dirty-worktree provenance of the original P4 campaign:** the run record
   shows HEAD `4ec5905a`, `dirty: true`, `lab/m51_pipeline_fair_ab.py`
   untracked, `lab/m51_pipeline_profile.py` modified — HEAD alone does not
   identify the executed bytes. Forensics + runner hardening + bounded
   clean-HEAD physical regression (P2/P3).

## 5. Safety rules in force

No force-push/reset/squash; `main` untouched; no PR merges; `evidence/` and
historical runs immutable (§2 trees re-verified at the end of this pass);
original P4 JSON and its source-HEAD/dirty flags never rewritten; no
FAIL→PASS; owner-verdict scope never widened; no new mandatory student
dependencies; ROCm/PyTorch/OpenVINO installations untouched.
