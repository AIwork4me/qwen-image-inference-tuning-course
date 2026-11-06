# PR #2 Independent Release Closure — Phase P0 Baseline Freeze

- Auditor: independent release-closure session (opencode, GLM)
- Date: 2026-11-05 (UTC)
- Repository: https://github.com/AIwork4me/qwen-image-inference-tuning-course
- PR: https://github.com/AIwork4me/qwen-image-inference-tuning-course/pull/2
- Audit branch: `feat/m5-1-openvino-cpu-te-course`

## 1. Git / PR integrity

| Item | Observed value | Verdict |
|---|---|---|
| Working tree | clean (`git status --short` empty) at audit start | OK |
| Local HEAD | `6aacd6f502ed37bfac77ff42ae0c04d056c387de` | matches known PR HEAD |
| Remote PR head (`gh pr view 2`, `headRefOid`) | `6aacd6f502ed37bfac77ff42ae0c04d056c387de` | no drift; no rebase needed (rule 15 satisfied) |
| origin/main | `5b93c21ac08359b18db2a360d3110d7d3bd16cae` | matches known main; untouched |
| PR state | OPEN, **Draft**, `MERGEABLE`, mergeStateStatus `CLEAN`, base `main` | OK |
| PR commits | 5 commits: `20a7f7e` (docs bootstrap, 2026-11-04) → `840c484` (feat lab) → `bda9283` (evidence) → `9283778` (docs tighten) → `6aacd6f` (CLI verify) | OK |
| Diff size vs main | 96 files, +19,157 / −5; `git diff --check` clean | OK |
| CI | `CPU test gate (no GPU claims)` — SUCCESS (run 42014896184, 2026-11-05T10:57Z) | OK |
| Reviews | none submitted; PR is Draft | OK |
| Network note | one transient `CONNECT 503` from the HTTP proxy on `git ls-remote`; authoritative remote head confirmed via GitHub API (`gh pr view`) | OK |

Note: a local branch label `master` points at `20a7f7e` (bootstrap commit); the actual default branch is `main` (`5b93c21`). Cosmetic only; `main` was never modified.

## 2. Historical evidence freeze

- `docs/release-closure/PR2_BASELINE_FREEZE_MANIFEST.txt` records, at audit start:
  - git blob identities (path, blob SHA, size) for `evidence/`, `runs/`, `lab/`, `tests/`, `scripts/`, `docs/figures/` (493 tracked entries);
  - SHA256 of every raw JSON / safetensors / npz record inside `runs/m51-20261105-cleanroom-01/` (311 files).
- Frozen campaign `evidence/` (59 MB) and `runs/m51-20261105-cleanroom-01/` (186 MB) are fully git-tracked in the PR; no untracked or modified evidence files exist.
- Large OpenVINO IRs are NOT in git; they live at `/opt/m51-artifacts/{fp32,bf16,int8}` (26 GB / 13 GB / 6.5 GB) and are symlinked into the run dir. Their conversion manifests pin `output_bin_sha256` (and xml sha for fp32/int8; the bf16 manifest lacks its own xml digest — audited in P0-B).

## 3. Source / environment snapshot (audit host, at freeze time)

- Host: AMD EPYC 9334 ×2 sockets, 128 logical threads, 2 NUMA nodes; cgroup `cpu.max = 1600000 100000` (16 CPUs), `memory.max = 59055800320` (55 GiB).
- GPU: gfx1100 (DID `0x744b`, VBIOS `113-D7070910-100`), 48 GiB VRAM, ~2% used at freeze.
- ROCm 7.2.1 (`/opt/rocm/.info/version`); PyTorch 2.9.1+gitff65f5b (HIP 7.2.53211) in `/opt/venv` and `.venv-lab`.
- OpenVINO 2026.4.1 + NNCF 3.4.0 in `.venv-ov` (read-only `.pth` bridge to the ROCm PyTorch) — verified importable at freeze.
- ComfyUI v0.43.0 pinned tree at `ComfyUI/` (marker `b5a432157de6b8884f81104cd02926adf4d597e4`).
- Disk: `/workspace` 98 GB (47 GB free), `/` overlay 3.5 TB (1.6 TB free), `/tmp` 4 GB tmpfs.
- Competing workload present during audit: an unrelated python process (`/workspace/yolo26-rocm-day1-labs`) at ~99% of one core. Any new physical performance measurement in this audit must either account for it or be scheduled around it (recorded per measurement; see P4).

## 4. Provenance finding (campaign source attribution)

`runs/m51-20261105-cleanroom-01/environment.json` records `git.head = 20a7f7e5afcb...` (the Nov-4 docs bootstrap commit), while the implementation was committed later:

| Event | Time (UTC, 2026-11-05) | Commit |
|---|---|---|
| Baseline probes written (`baseline/probe_cpu.json`) | 08:09 | working tree @ `20a7f7e` (code uncommitted) |
| CLI smoke / int8-smoke / blocked runs (`source_head=20a7f7e`) | 10:33 | working tree @ `20a7f7e` (code uncommitted) |
| e2e / pipeline summaries written | 10:31–10:34 | working tree @ `20a7f7e` (code uncommitted) |
| `feat(m5.1): implement...` committed | 10:39:12 | `840c484` |
| evidence committed | 10:39:26 | `bda9283` |
| docs tightened | 10:50:16 | `9283778` |
| Formal CLI verification run (`source_head=9283778`, PASS) | 10:56:42 | committed as `6aacd6f` at 10:56:46 |

Classification: the campaign ran on **code that existed in the working tree before it was committed** (bootstrap `20a7f7e` + later commit `840c484`), and the formal CLI run at 10:56 is a **post-commit replay** that reproduced the encoder results at a fully committed HEAD. This is not inaccurate attribution of a *different* codebase, but the `source_head` values of 10:33–10:34 records understate the actual code state. The provenance addendum in `PR2_EVIDENCE_REDERIVATION.md` states this precisely; historical records were NOT rewritten.

## 5. P0 problem confirmation (code-level, pre-fix)

- **P0-A** `lab/m51_run.py:71-73` — `"status": "PASS" if finite else "FAIL"`: execution success (finite tensors) is reported as scientific PASS; no gates for token contract, alignment, e2e, or human review; `honest_status()` exists but is not wired into H3 result assembly.
- **P0-B** `lab/m51_contract.py::artifact_status()` — checks file presence + manifest JSON shape + presence of a digest field; never compares bytes. `m51_run.py` copies `artifact_digest` from the manifest without verification. BF16 manifest does not pin its own XML digest.
- **P0-C** empty-negative discrepancy — confirmed from raw evidence and pinned ComfyUI source:
  - production `clip.tokenize("")` → 22 ids, NO token `220`;
  - manual diagnostic `tokenize_with_weights(..., prevent_empty_text=True)` → 23 ids, inserts `220` (literal word "empty") at index 17;
  - pinned ComfyUI `comfy/text_encoders/qwen_image21.py:18` — `prevent_empty_text=False` is the **production default**; the OV encoder matches production;
  - therefore `token_ids_self_consistent=false` for G/C0 is an artifact of comparing non-equivalent semantics in the diagnostic (and is vacuously `true` for OV arms). Fix belongs in the diagnostic, not in OpenVINO input semantics.
- **P1-A** `lab/m51_run.py:48-52` — `free_ram` = cgroup `memory.max` (the ceiling), not actual availability; with 38.5 GiB already in use the gate still sees 55 GiB.
- **P1-B** `lab/m51_pipeline.py:192,197` — hardcoded `/opt/m51-artifacts/{bf16,int8}`; `m51_env.py` re-run would overwrite the frozen `environment.json` (evidence-safety hazard); `provision_ov.sh` defaults `TMPDIR=/workspace/tmp`.
- Reproduction-path drift: PR comment says `.venv-lab/bin/python -m lab run ...` — this is correct as the runner venv (`.venv-ov` is spawned internally by `m51_run`); confirmed both venvs exist and are functional.

## 6. Result

**BASELINE_PASS**

Baseline integrity is established: clean tree at the exact known PR head, remote in agreement, CI green, evidence complete and frozen (manifest above), physical environment present and matching the campaign claims. Destructive operations may proceed under the non-negotiable rules (no history rewrite, no main changes, no merge, frozen evidence untouched).
