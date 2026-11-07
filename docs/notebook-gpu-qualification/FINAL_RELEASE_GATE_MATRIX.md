# Final Release Gate Matrix

**Campaign:** `runs/notebook-gpu-qualification/w7900-20261106-073429/` · PR #4 branch `feat/bilingual-jupyter-optimization-labs`

| # | Gate | Status | Evidence |
|---|---|---|---|
| 1 | All 18 notebooks validated in real Jupyter kernels | PASS | `notebook_execution_matrix.json` (nbclient 0.10.4 + ipykernel 7.2.0), executed artifacts under `executed/` |
| 2 | All 18 pass CPU REFERENCE mode | PASS | 18/18 PASSED (pass A; re-run after final notebook build in CI test `test_notebook_reference_run_all`) |
| 3 | W7900 hardware admission passes | PASS | `P0_HARDWARE_ADMISSION.json` (strict doctor PASS, six pinned weights digest-verified) |
| 4 | GPU-first modules have genuine physical measurements | PASS | B0/L1/F1/F2/BEST full protocol, rocprofv3 traces, M5 pilots/pipeline/fairab — all `q1-*` run records on this host |
| 5 | M1 generates real baseline images | PASS | 16 PNGs of run `q1-075553-B0` (kept on the measurement host under runs/raw, gitignored by design; SHA256 per record) |
| 6 | M2 captures real profiler evidence | PASS | `runs/q1-075553-profile/` (pipeline + kernel-dit + kernel-te, verdict PASS; GEMM 61.2% reproduced) |
| 7 | M3 measures real less-computation candidate | PASS | L1 12.029 s vs B0 55.096 s → 4.58× FORMAL_PROTOCOL_MEASURED |
| 8 | M4 physically compares attention + quantization | PASS | F1 43.247 s (ck verified active) 1.274×; F2 50.104 s 1.100× + 6.66 GB VRAM saved |
| 9 | M5.0 demonstrates physical CPU-offload behavior | PASS | GPU TE 0.24–0.61 s vs PyTorch CPU 62–210 s (217–350×); H1 integrated 45.2 s vs 266.5 s; H2 pipelined 711 s (negative result measured, preserved) |
| 10 | M5.1 OpenVINO pipeline verified in host memory envelope | PASS (with warnings preserved) | OV BF16 gates PASS; OV/CPU-PT 340–486×; e2e A/C arms with PSNR/SSIM/LPIPS; fairab 2×2 reproduced (arch 1.012× matched clients); artifact admission bf16 = PASS_WITH_WARNINGS (unpinned xml digest) — distinction preserved |
| 11 | M6 compares physical B0/BEST, no auto-approval | PASS | 6.107× measured; grids + poster crops + LPIPS triage produced; verdict PENDING_HUMAN_REVIEW |
| 12 | M7 produces real student-style evidence package | PASS | capstone reports in the student `runs/` area + PR draft with evidence class |
| 13 | No GPU claim relies solely on historical JSON | PASS | every comparison cites `q1-*` run ids; historical values always tagged `[HISTORICAL]` |
| 14 | Bilingual notebooks preserve identical execution semantics | PASS | byte-identical code cells (CI-enforced); both tracks executed in both modes |
| 15 | Run All bounded and classroom-safe | PASS | bounded profiles + timeouts + GPU lock; research gated to GPU_RESEARCH; budgets enforced via --warmup/--measured |
| 16 | Current GPU measurement limitations stated | PASS | GPU_PERFORMANCE_MATRIX.md §Limitations; notebooks tag estimates/simulations |
| 17 | Frozen benchmark evidence unchanged | PASS | `git status evidence/` clean at every gate (validate_evidence check) |
| 18 | All CPU tests + course-quality checks pass | PASS | 216 tests (175 pre-closure + 41 student-provenance/admission); 6 course-quality gates green |
| 19 | Verified release-candidate CI; final merge-head check mandatory | PASS | At reviewed source baseline `33ada62`: [Actions #42316216029](https://github.com/AIwork4me/qwen-image-inference-tuning-course/actions/runs/42316216029) — 216 CPU tests plus six quality gates SUCCESS. Earlier physical-code HEAD `03726e8`: [Actions #42242142758](https://github.com/AIwork4me/qwen-image-inference-tuning-course/actions/runs/42242142758). Any later publication fix must pass its own PR-head CI before merge; historical CI is not a proxy for new code. |
| 20 | Exact GPU execution commands and artifacts recorded | PASS | `campaign_status.json` (per-step cmd + rc + wall), run records, session/matrix JSONs |
| 21 | Instructor can reproduce the student workflow | PASS | rehearsal script + STUDENT_USABILITY_AUDIT.md; instructor guides updated |

## Known limitations (not release-blocking, disclosed)

- M5.1 OpenVINO artifacts are reused from `/opt/m51-artifacts` (byte-verified; bf16 PASS_WITH_WARNINGS for the unpinned XML digest). A fresh provisioning needs ~55 GiB free disk.
- The fair-factorial campaign run executed with a dirty worktree (uncommitted docs-only changes at run time; experiment modules' SHA256 captured in `source_snapshot/`).
- `lab profile`'s pipeline layer pools the repo's committed `acceptance-b0-B0` warm records with the campaign's (same experiment; medians agree <0.2%).
- Classroom pass-B executed M1-EN with the pre-fix budget (6 generations for one prompt instead of 4); fixed for all subsequent runs; both variants' populations are labelled bounded either way.

**Verdict: READY_FOR_OWNER_REVIEW_WITH_LIMITATIONS** (216 CPU tests and six quality gates verified at source baseline `33ada62`; the exact final merge head requires an independent green check before merge). Full GPU_CLASSROOM 18/18 passed at earlier physical qualification HEAD `9e550e8`; the student-independence implementation was physically revalidated on affected paths at `03726e8`, while a complete new GPU_CLASSROOM 18/18 was blocked by an external GPU tenant. See [STUDENT_INDEPENDENCE_FINAL_GATE_MATRIX.md](STUDENT_INDEPENDENCE_FINAL_GATE_MATRIX.md), gate 20. Do not represent the new-head full GPU regression as completed.
