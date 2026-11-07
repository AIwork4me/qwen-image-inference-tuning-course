# Student Independence Final Gate Matrix

Campaign: `runs/notebook-gpu-qualification/w7900-20261106-144348-sic` (physical W7900, gfx1100, 48 GB)
Source HEAD at qualification: `03726e8af0a6350ca6016c77fb759ceebf61e34e` (branch `feat/bilingual-jupyter-optimization-labs`)
CI at that HEAD: https://github.com/AIwork4me/qwen-image-inference-tuning-course/actions/runs/42242142758 (CPU gate SUCCESS, run #31, 216 tests)
Fresh-clone acceptance: isolated clone under `/workspace/student-acceptance` (zh own arms 485 s · EN explicit reuse 12 s · EN negative fail-closed 2.5 s)

Statuses: `PASS | PASS_WITH_LIMITATIONS | FAIL | BLOCKED | NOT_RUN`.
Provenance and artifact-admission gate failures are release-blocking.

| # | Gate | Status | Evidence |
|---|---|---|---|
| 1 | Fresh-clone source independence | PASS | clone contains tracked files only; shared pinned deps (weights 57 GB, pinned ComfyUI, `.venv-lab`, `.venv-ov`, `/opt/m51-artifacts`); isolated `runs/`; no instructor session state |
| 2 | No implicit instructor-run fallback | PASS | all `sorted(glob)[-1]` selectors removed (nb05 §3, nb06 Parts A/B, nb02 traces); CI-enforced builder spec; `tests/test_notebooks.py` |
| 3 | Explicit run-ID provenance | PASS | `REUSE_TE_RUNS`/`REUSE_PIPELINE_RUN` single edit point; EN acceptance consumed zh run ids verbatim (`SAME_STUDENT_PRIOR_PHYSICAL`) |
| 4 | Correct physical/current/reference classification | PASS | `lab/student_provenance.py`; 41 unit cases; live labels in both acceptance kernels |
| 5 | M5.1 BF16 artifact admission | PASS_WITH_LIMITATIONS | `admit_ov_bf16()` = PASS_WITH_WARNINGS (xml digest computed, not pinned — preserved, never rounded up); RAM 45.6 GiB usable vs 34 GiB; 16 CPU quota |
| 6 | Artifact verification failure propagation | PASS | live negative rehearsal (missing root → BLOCKED → session record BLOCKED → notebook FAILED with remediation); unit negatives: corrupted/truncated bin, identity & precision mismatch, corrupt/missing manifest, RAM gate, missing runtime — all FAIL/BLOCKED |
| 7 | Physical GPU-native A arm | PASS | zh acceptance run `143047-nbq-m51pipe` arm A: 4 requests, batch wall 54.14 s |
| 8 | Physical OpenVINO CPU-TE C arm | PASS | same run arm C: OV BF16 worker, overlap proven (1.58 s), worker RSS 30 014 MB |
| 9 | Matched A/C experimental semantics | PASS | workload identity: 4 requests, seed 42, prompt digest `8f96b33bf90f6edc`; `ac_pair` validation enforced |
| 10 | Correct throughput derivation | PASS | 4 images / batch wall; numerator/denominator/unit in the validated result record |
| 11 | CPU and GPU memory reporting | PASS | C worker RSS 30 014 MB, server RSS 1 906 MB, A VRAM peak 33 996 MB; encoder CPU RSS reported |
| 12 | Distinct encoder-only vs end-to-end speedups | PASS | 621× encoder-only vs 1.096× system batch, same session, both printed with run ids and the "NOT the system speedup" warning |
| 13 | INT8 numerical FAIL preserved | PASS | `test_int8_negative_failure_blocks_fidelity_claim`; INT8 specs and frozen verdicts untouched |
| 14 | Historical human-quality verdict scope preserved | PASS | `run_h3(apply_owner_verdict=False)` default; new H3 runs `PENDING_HUMAN_REVIEW` + reference-only note |
| 15 | Student result isolation | PASS | validated result records; isolated clone `runs/`; unique append-only tags; rerun created new run ids without touching old ones |
| 16 | No historical evidence modification | PASS | `git status evidence/` clean; `validate_evidence` gate 4 |
| 17 | English and Chinese M5.1 execution parity | PASS | byte-identical code cells (CI); zh own-arms PASSED + EN explicit-reuse PASSED in separate fresh kernels |
| 18 | Real Jupyter execution | PASS | nbclient + ipykernel, `allow_errors=False` (run_all.py + acceptance driver) |
| 19 | Full 18-notebook CPU regression | PASS | 18/18 PASSED at HEAD 03726e8 (campaign `matrix_REFERENCE_final.json`; re-verified by CI run #31) |
| 20 | Full GPU_CLASSROOM regression | PASS_WITH_LIMITATIONS | 18/18 REFERENCE + targeted physical revalidation of every changed GPU path at this HEAD (nb05 EN own arms 218.8 s PASSED; nb06 zh full own arms 485.2 s PASSED; nb06 EN reuse PASSED; nb06 negative fail-closed PASSED; GPU-busy admission block observed live and correctly refused). The full 18-notebook GPU_CLASSROOM re-run at this exact HEAD was attempted (`w7900-20261106-144348-sic`, waiter `wait_vram.log` 240 polls) but BLOCKED for the whole window by an external tenant holding ~31 GiB VRAM (host PIDs 3906994/2439485, outside this container's namespace); the frozen 18/18 GPU_CLASSROOM evidence at 9e550e8 (`w7900-20261106-073429`) remains valid for the unchanged execution paths |
| 21 | Negative test coverage | PASS | 41 new CPU negatives + live fail-closed rehearsals (missing artifacts; GPU busy; first-attempt m51pipe path bug caught and fixed by the physical rehearsal itself) |
| 22 | Bilingual documentation | PASS | README×2, notebooks/README×2, INSTRUCTOR_GUIDE×2, M5 lessons×4, STUDENT_PROVENANCE_CONTRACT.md; all six course-quality gates green |
| 23 | Updated PR description | PASS | PR #4 body rewritten via API (GPU-first deliverable + independence closure + disclosed limitations) |
| 24 | Final GitHub CI | PASS | run #31 (42242142758) SUCCESS at 03726e8: 216 CPU tests incl. 18-notebook REFERENCE execution in real kernels |
| 25 | Owner merge readiness | PASS_WITH_LIMITATIONS | owner review pending by design; gate 20 limitation disclosed above; no release-blocking provenance/admission issue |

## Fresh-clone acceptance measurements (Radeon Cloud W7900, gfx1100, this closure)

| Quantity | Value | Provenance |
|---|---|---|
| OV BF16 artifact admission | PASS_WITH_WARNINGS (xml digest computed, not pinned) | `/opt/m51-artifacts/bf16` (env `M51_OV_ARTIFACTS`) |
| Encoder PyTorch CPU → OV BF16 | 178.6 s → 0.288 s (≈621×, warm, this host) | CURRENT_SESSION_PHYSICAL, run `143017-nbq-H3_OV_BF16` (mode=formal, gates PASS, human PENDING) |
| A arm (GPU-native TE) | batch wall 54.14 s · 4.43 img/min · VRAM peak 33 996 MB · server RSS 1 961 MB | CURRENT_SESSION_PHYSICAL, run `143047-nbq-m51pipe` |
| C arm (OV BF16 CPU TE) | batch wall 49.39 s · 4.86 img/min · worker RSS 30 014 MB · overlap proven 1.58 s | same run |
| System batch speedup (C vs A) | 1.096× (BOUNDED_PHYSICAL_COMPARISON, one 4-request batch) | validated student result record |
| GPU TE pilot | 1.04 s warm (n=1) | run `142645-nbq-m51-te-cuda` |
| Resource envelope | 48 GB VRAM · 55 GiB cgroup (45.6 GiB usable) · 16 CPU quota | session records |

These are one student's bounded pilot numbers — **not** formal benchmark claims;
the committed campaign (`q1-075553-*`) remains the formal reference.

## Provenance demonstrations (exact examples)

- Current-session physical: `[PROVENANCE] CURRENT_SESSION_PHYSICAL | run_id=143047-nbq-m51pipe | measured on THIS GPU in THIS session (physical)`
- Previous student physical (EN kernel reusing zh kernel's runs): `[PROVENANCE] SAME_STUDENT_PRIOR_PHYSICAL | run_id=143017-nbq-H3_OV_BF16 | YOUR earlier physical run on this workspace (prior session, explicit run id)`
- Shared campaign physical: `SHARED_CAMPAIGN_PHYSICAL` via declared campaign roots (`$QIR_NOTEBOOK_CAMPAIGN`)
- Committed physical reference: `[PROVENANCE] COMMITTED_PHYSICAL_REFERENCE | run_id=q1-075553-H2-pipeline | committed physical reference tracked in Git (instructor campaign; NOT measured by you)`
- Frozen reference: `evidence/results/SUMMARY.json` → `FROZEN_REFERENCE`

Invalid promotions fail closed (41 tested cases): committed-as-current, reuse-as-MEASURED,
smoke-as-FORMAL, A/C workload mismatch, stale-session claims, numerical-FAIL-with-human-PASS,
M6 PENDING promoted to release PASS, REFERENCE_ONLY with speedup, invalid enums, duplicate
run-id metadata conflicts.
