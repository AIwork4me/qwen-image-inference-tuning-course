# P0 — Publication-Closure Baseline Freeze

**Frozen at:** 2026-11-06, by the publication-closure engineering session.
**Purpose:** independently recorded state of `main`, PR #2 and PR #3 before any
publication-closure change. All fingerprints below are content-addressed Git
tree/blob hashes taken from the remote refs at freeze time; none of the frozen
trees may change in any publication-closure commit.

## 1. Git state at freeze

| Ref | Commit | Subject |
|---|---|---|
| `origin/main` | `767071af4c399c65732189ab77178d247a78ee81` | docs: update Radeon Cloud tutorial link |
| PR #2 head `origin/feat/m5-1-openvino-cpu-te-course` | `a7fa69ae18800fc1f7a83e24de1c0b8bad2e7eaa` | test(m51): final-HEAD physical smokes — BF16 gates PASS / INT8 RELEASE_FAIL preserved |
| PR #3 head `origin/feat/world-class-bilingual-inference-course` | `4ec5905a18ad99108dbb54c9b434fbbf43579626` | chore: record final HEAD and CI state in mission ledger |

Stacking (verified with `git merge-base`):

- `merge-base(main, PR#2) = 767071a` → PR #2 sits directly on `main`.
- `merge-base(PR#2, PR#3) = a7fa69a` → PR #3 is stacked **exactly** on the PR #2 head (base branch `feat/m5-1-openvino-cpu-te-course`, confirmed via the PR #3 API `baseRefName`).

Diff sizes at freeze:

| Range | Files | Insertions | Deletions |
|---|---|---|---|
| `main...PR#2` | 156 | 31,644 | 22 |
| `PR#2...PR#3` | 72 | 9,974 | 269 |

## 2. CI state at freeze (observed, not assumed)

| PR | Head | Run | Result | URL |
|---|---|---|---|---|
| #2 | `a7fa69ae` | 42032143208 | success (job 127260988984) | https://github.com/AIwork4me/qwen-image-inference-tuning-course/actions/runs/42032143208 |
| #3 | `4ec5905a` | 42049486025 | success (job 127321410009) | https://github.com/AIwork4me/qwen-image-inference-tuning-course/actions/runs/42049486025 |

- PR #3's green run at `4ec5905a` reports **101 passed** tests plus the four
  course-quality gates (links / bilingual / claims / examples). Any release
  document stating "100 tests" for this HEAD is stale metadata (repaired in
  Phase P6); historical statements about *earlier* CI runs stay untouched.
- PR #2's CI gate is pytest-only (no course-quality scripts — those arrive with
  PR #3's workflow extension).
- Intermediate PR #3 runs `9af8c55e`/`1d797ad7` failed and were fixed by
  `8244d4c`/`4ec5905` (public history; nothing rewritten).

Draft state: both PRs are **Draft** at freeze; `mergeStateStatus: CLEAN`,
`mergeable: MERGEABLE` for both. No reviews or unresolved review threads exist
on either PR (only three informational issue comments by the owner account on
PR #2).

## 3. Frozen evidence fingerprints (Git tree hashes)

A tree hash fingerprints an entire subtree content-addressedly; any byte change
anywhere below changes it.

| Path | `main` | PR #2 | PR #3 | Verdict |
|---|---|---|---|---|
| `evidence/` | `33e735b7…` | `33e735b7…` | `33e735b7…` | **identical across all three refs — frozen evidence untouched** |
| `runs/m51-20261105-cleanroom-01/` | — | `d297062a…` | `93cbd6a7…` | PR #3 adds ONLY the six full-resolution review PNGs under `e2e/raw/` (verified `git diff --name-status` = 6 × `A`, zero modifications/deletions) |
| `runs/pr2-p4-cpuab-114444/` | — | `70f31524…` | `70f31524…` | identical |
| `runs/pr2-p4-cpuab-v2-114711/` | — | `582f9a75…` | `582f9a75…` | identical |
| `runs/pr2-p4-pipewarm-120822/` | — | `a21d465f…` | `a21d465f…` | identical |
| `docs/release-closure/` | — | `a1eadf75…` | `373f12dd…` | PR #3 adds the owner-verdict record + provenance + INDEX remediation (subject of the P2 surgery) |
| `docs/quality-v2/` | — | — | `134691b5…` | course-v2 audit set (PR #3 only at freeze) |

Six owner-review PNGs (working-tree SHA256, spot-checked against
`review-sheets/INDEX.json` records — full six verified in
`docs/quality-v2/IMAGE_EVIDENCE_AUDIT.md`):

| File | SHA256 (prefix) | Git blob (PR #3) |
|---|---|---|
| `e2e/raw/G_native__p1-zh-scene__cfg1.png` | `c7ee7331…` | `65676395` |
| `e2e/raw/C1b_OV_BF16__p1-zh-scene__cfg1.png` | `2e922301…` | `dacd2704` |
| `e2e/raw/G_native__p2-en-scene__cfg1.png` | `f456d05c…` | `48a71df7` |
| `e2e/raw/C1b_OV_BF16__p2-en-scene__cfg1.png` | `50e0424a…` | `e6316b7a` |
| `e2e/raw/G_native__p3-poster-text__cfg1.png` | `cac599c5…` | `c538f20e` |
| `e2e/raw/C1b_OV_BF16__p3-poster-text__cfg1.png` | `003cb560…` | `6ef0a4cb` |

## 4. Hardware/software environment (closure work host)

- GPU: gfx1100 (Radeon W7900-class), 48 GiB VRAM (`rocm-smi`: 51,522,830,336 B)
- CPU: AMD EPYC 9334 32-core; cgroup budget `cpu.max = 1600000/100000` (16 CPUs),
  `memory.max ≈ 55 GiB` — the same budget class as every M5.1 campaign run.
- ROCm stack: system install (`/opt/rocm`), PyTorch 2.9.1+rocm (borrowed
  preinstalled stack; never reinstalled), ComfyUI pinned checkout in-repo.
- OpenVINO artifacts: `/opt/m51-artifacts/{fp32,bf16,int8}`; the BF16
  deployment IR bin byte-verified against its conversion manifest
  (`sha256sum -c` → `language_model_bf16.bin: OK`, 13,892,143,718 B) before
  the Phase P4 measurement.

## 5. Risks recorded at freeze

1. **Stacked-PR dependency (P0-B):** PR #2's `docs/release-closure/` at freeze
   does NOT contain the owner-quality verdict record, its provenance page, the
   remediated review-sheet INDEX, or the six committed PNGs — they exist only
   in PR #3. PR #2 is therefore not a self-contained M5.1 release candidate at
   freeze (repaired in Phase P2).
2. **Cross-run metric mixing (P0-A):** both READMEs' Mermaid diagrams pair
   `737 s` (historical H2 campaign, 0.33 img/min) with `0.35 img/min` (the
   M5.1 rebaseline's 680.0 s B arm), and pair `50 s` (C1b cold-inclusive
   campaign, 4.80 img/min) with `4.55 img/min` (balanced warm audit,
   52.74 s mean) — two populations each, repaired in Phase P1.
3. **Fragile verdict validation (P0-C):** `check_claims.py` validates
   immutable verdicts with substring matching inside free-form prose
   (`"PASS" in quality_status`), so e.g. `"PASS (was FAIL)"` would evade the
   immutable-FAIL guard (repaired in Phase P3).
4. **Asymmetric-client comparison (P1-A):** C-fast vs A-original quotes
   (1.079×) mix client generations across arms; the symmetric A-fast arm was
   never measured (Phase P4 measures it or honestly reports BLOCKED).
5. **Stale release metadata:** `docs/quality-v2/FINAL_RELEASE_AUDIT.md` and
   `runs/overnight-course-v2-a7fa69a/MISSION_LEDGER.json` state "100 tests"
   while the verified final CI run reports 101 (Phase P6).
6. **Dirty-worktree risk:** the closure host's worktree was clean at freeze
   (`git status` empty on `feat/world-class-bilingual-inference-course`);
   the P4 experiment module is added as a new untracked file and new run
   records land under a fresh run ID, so no frozen path is written.

## 6. Non-negotiables honored

No `main` commits; no force-push or history rewrite; frozen trees above are
the verification basis for every later phase; numerical FAIL/NOT_RUN verdicts
(C2 INT8, C1b-uncompressed, M6 BEST pending, 40-step sweep NOT_RUN) are carried
forward unchanged; owner-acceptance scope is never widened.
