# Student Usability Audit — fresh-kernel rehearsal + friction log

**Question:** can a first-time graduate student, starting from the published
README, complete the GPU-first loop on a provisioned W7900 without touching
`lab/` source or shell pipelines?

## The rehearsed student path (exact steps)

1. **Onboarding (documented):** `bash scripts/cloud/start.sh` — staged weights, pinned ComfyUI, `.venv-lab`, cloud doctor (not FAIL), one smoke image.
2. **Open:** repository root in Jupyter → `notebooks/README.md` → module table → `00_experimental_discipline.ipynb`.
3. **Read:** learning objective, why-it-matters, the three-mode table.
4. **Configure:** first code cell — `COURSE_MODE = "GPU_CLASSROOM"` (and optionally `RUN_OWN_GPU_ARMS = False` on metered time).
5. **Run All:** admission banner shows `AMD Radeon Graphics gfx1100 … admission=PASS` or a student-readable blocker + exact fix.
6. **M0:** sync-vs-async demo measured on the student's own GPU; model digests verified.
7. **M1:** their own B0 bounded run (~5 min): `run_id` printed, median/spread/VRAM, deviation vs the 55.206 s reference computed and discussed.
8. **M2/M3/M4/M5/M6:** physical exercises per module (see FULL_18_NOTEBOOK_VALIDATION.md), each ending in a compact table + plot from measured records.
9. **M7:** capstone collects their records into an append-only capstone record under `runs/` + a PR draft with an honest evidence class.

A full fresh-clone rehearsal (file-transport clone, day-2 student reusing
staged weights, no re-downloads) executed M0 → M1 → M3 → M7 in real kernels;
results are in `runs/notebook-gpu-qualification/w7900-20261106-073429/`
(rehearsal section of the execution matrix).

## Friction found and fixed during qualification

| # | Friction | Where | Fix |
|---|---|---|---|
| 1 | classroom budgets silently used the manifest's full per-prompt measured count (CLI lacked `--warmup/--measured`) | `lab/__main__.py`, `lab/notebook_runtime.py` | added the two flags; runtime passes the declared bounded budget; verified on subsequent pass-B runs |
| 2 | `m51_pipeline_fair_ab.py` failed when launched as a script (`ModuleNotFoundError: lab`) | `lab/m51_pipeline_fair_ab.py` | path shim added; both `-m` and script forms now work |
| 3 | campaign drivers passing relative `--run` ids wrote output to the repo root | driver usage | documented + rerun with `runs/`-prefixed ids; outputs moved; summarize globs fixed |
| 4 | orphaned ComfyUI server after a killed parent kept 32 GB VRAM | runtime | busy-GPU admission (>8 GB) now blocks with guidance; the GPU lock also carries the owner pid |
| 5 | executed-notebook artifacts and sessions initially had no home | runtime/harness | campaign-scoped `sessions/` + `executed/<MODE>/` directories |

## Remaining student-facing caveats

- M5.1 needs `M51_OV_ARTIFACTS` (or the default `runs/m51-ov-artifacts`) with verified IRs; without them the module states the blocker instead of failing silently (PASS vs PASS_WITH_WARNINGS distinction preserved).
- The M5.0 CPU pilot occupies ~30 GB RSS for minutes; the notebook prints a RAM note before running it (cgroup-aware).
- Students sharing an instance must respect the one-GPU-workload rule; the runtime enforces it across kernels.
