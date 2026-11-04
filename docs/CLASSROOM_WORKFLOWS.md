# Classroom Workflows — smoke vs formal, concretely

*One page, no ambiguity: what to run in class (bounded, cheap, safe to
interrupt) and what to run for quotable results (full protocol, budgeted).
Everything here matches the CLI exactly (`python -m lab <cmd> --help`).*

---

## The classroom-safe smoke workflow (bounded by design)

**Cost: 1–3 GPU-minutes per command. Safe to interrupt — every run is
resumable and nothing is ever re-downloaded.**

```bash
# one-time setup (~15 min, downloads only what the smoke needs)
cd /workspace/qwen-image-inference-tuning-course
bash scripts/cloud/start.sh

# the course's smoke commands (add --smoke to any experiment run):
.venv-lab/bin/python -m lab run --experiment B0   --smoke --date m1-smoke
.venv-lab/bin/python -m lab run --experiment L1   --smoke --date m3-smoke
.venv-lab/bin/python -m lab run --experiment F1   --smoke --date m4-smoke
.venv-lab/bin/python -m lab run --experiment F2   --smoke --date m4-smoke
.venv-lab/bin/python -m lab run --experiment BEST --smoke --date m6-smoke
```

What `--smoke` does (enforced in the runner, not by convention):
- 1 prompt × 1 measured run (plus one cold) instead of 3×3 + fresh arm;
- writes ONLY under `runs/<tag>-<ID>-smoke/` (the `-smoke` suffix is what
  the report uses to keep them out of formal medians);
- never touches `evidence/` (frozen campaign) — verified by a CI test.

Inspecting smoke results without polluting anything:

```bash
.venv-lab/bin/python -m lab report --scope student --kind smoke   # read-only
.venv-lab/bin/python -m lab report --scope student --kind formal  # read-only
```

**Interruption recovery:** any killed run leaves a partial dir under
`runs/`; re-running with a NEW `--date` starts clean (evidence is
append-only). Models/venv/ComfyUI are verified, not re-downloaded, by
re-running `start.sh` (idempotent; safe after instance restarts).

## The formal benchmark workflow (quotable medians)

**Cost: 4–16 GPU-minutes per experiment (budget table below). Run when you
need a number you will quote, compare against the reference, or attach to
a PR.**

```bash
bash scripts/cloud/start.sh --all-models    # one-time: unlock the full matrix
.venv-lab/bin/python -m lab doctor          # STRICT preflight must PASS

# full protocol sessions (fresh server, cold+warmup+3x3 measured+fresh arm):
.venv-lab/bin/python -m lab run --experiment B0   --date <your-tag>
.venv-lab/bin/python -m lab run --experiment L1   --date <your-tag>
#   ... F1 F2 L1O LF1 LF2 LF3 BEST

# quality gates from YOUR raw images (manifest-declared protocol):
.venv-lab/bin/python -m lab quality --experiment F2
.venv-lab/bin/python -m lab quality --experiment BEST

# aggregate YOUR formal sessions (never your smokes, never the reference):
.venv-lab/bin/python -m lab report --scope student --write
```

**GPU-time budget (W7900-class, full sessions):**

| Experiment | Full | Smoke | Notes |
|---|---|---|---|
| B0 | ~16 min | ~3 min | needed as the pairing reference for everything |
| L1 / L1O | ~5 min | ~2 min | turbo-bf16 model required |
| F1 | ~13 min | ~3 min | ck attention; fidelity gate vs B0 |
| F2 | ~15 min | ~3 min | int8; fidelity gate vs B0 |
| LF1–LF3, BEST | ~4 min | ~2 min | turbo-int8 (already staged by default) |
| H1 / H2 | ~15 min each | — | **opt-in research only** (slow CPU-bound); taught from the committed evidence by default |

**Rules that keep formal numbers quotable:**
1. Strict doctor PASS before the session (wrong stack = not comparable).
2. One experiment per server lifecycle (the runner enforces fresh server).
3. Full protocol only — `--smoke` output is classroom evidence, never a
   median you quote.
4. `report --scope student --write` for YOUR numbers; the reference scope
   is read-only by default (`--write` on it rewrites the frozen SUMMARY —
   a maintenance action, not a student action).
5. H1/H2 reruns are research tasks: budget ~15 min each and expect the
   committed negative results to be the default teaching material.
