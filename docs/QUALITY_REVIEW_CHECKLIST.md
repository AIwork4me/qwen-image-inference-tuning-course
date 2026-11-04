# Owner Quality-Review Checklist — the binding three-tier verdict

**Who may complete this:** the repository owner, personally. Not the
instructor, not a contributor, not a majority of students. The course's
quality gate is binding by design: automation prepares evidence, a human
decides. Until this checklist is completed and committed, every
schedule-changing experiment (L1, L1O, LF1–LF3, BEST) stays
**PENDING_HUMAN_REVIEW** — including BEST, the 6.12× headline config.

**Time required:** 20–30 minutes with the images at native resolution.

---

## Step 0 — Set up an honest viewing environment

- [ ] View on a properly calibrated monitor, at 100% zoom (no fit-to-window
      scaling of the crops — that re-introduces exactly the downsampling the
      native-res crops exist to avoid).
- [ ] Good lighting; no time pressure. If you catch yourself skimming,
      stop and return later.

## Step 1 — Overall look: the paired grids (≈10 min)

For each experiment below, open the 256 px grid and scan prompt by prompt,
baseline column | experiment column. You are judging: *would this output
serve the same purpose as the baseline?*

| Experiment | Grid | What changed vs baseline |
|---|---|---|
| BEST | [`evidence/quality/BEST/quality_grid.png`](../evidence/quality/BEST/quality_grid.png) | turbo schedule + INT8 weights |
| L1 | [`evidence/quality/L1/quality_grid.png`](../evidence/quality/L1/quality_grid.png) | turbo schedule only |
| L1O | [`evidence/quality/L1O/quality_grid.png`](../evidence/quality/L1O/quality_grid.png) | turbo + official sigmas |
| F1 | [`evidence/quality/F1/quality_grid.png`](../evidence/quality/F1/quality_grid.png) | attention backend only (fidelity) |
| F2 | [`evidence/quality/F2/quality_grid.png`](../evidence/quality/F2/quality_grid.png) | INT8 weights only (fidelity) |
| LF1/LF2/LF3 | same pattern under [`evidence/quality/`](../evidence/quality/) | combinations |

Notes per experiment (write as you go):

- BEST: ______________________________________________
- L1: ________________________________________________
- F1: ________________________________________________
- F2: ________________________________________________

## Step 2 — Typography: native-resolution crops (≈10 min, the acceptance axis)

Chinese text rendering is where aggressive schedules fail silently at
thumbnail size. Open each crop poster and compare the baseline row to the
experiment row at 100% zoom, checking: stroke integrity, character
completeness, layout stability, sharpness of edges.

- [ ] BEST: [`evidence/quality/BEST/quality_crops_poster.png`](../evidence/quality/BEST/quality_crops_poster.png)
- [ ] L1: [`evidence/quality/L1/quality_crops_poster.png`](../evidence/quality/L1/quality_crops_poster.png)
- [ ] L1O: [`evidence/quality/L1O/quality_crops_poster.png`](../evidence/quality/L1O/quality_crops_poster.png)

For a side-by-side at full source resolution, the raw pairs are also
committed per prompt/seed under
[`evidence/samples/B0/`](../evidence/samples/B0/) and
[`evidence/samples/BEST/`](../evidence/samples/BEST/) (e.g. the poster:
`p3-poster-text-42-r0.png` in both).

Typography notes:

- BEST (「星际远航」/「2026年10月30日 全球上映」rendering): ______________
- L1: ____________________________________________________________

## Step 3 — Read the fidelity numbers (context, not verdict)

Fidelity gates measure numerical damage; they inform, never decide:

- F2 (INT8): LPIPS 0.005–0.031 (noise floor 0.0000) — near-lossless band.
- F1 (ck attention): 0.004–0.065 — the backend swap's numerical footprint.
- Turbo family vs B0: LPIPS 0.19–0.30 — **distance, not damage**: a
  different schedule legitimately produces different images. A high
  distance does NOT mean worse; only your eyes on the grids decide.

## Step 4 — Record the verdicts (the only binding step)

For each experiment, exactly one tier:

| Tier | Meaning |
|---|---|
| 无感 (imperceptible) | you would ship it without hesitation |
| 略降 (slightly degraded) | acceptable for most uses; note the specific weakness |
| 不可接受 (unacceptable) | name the failure (e.g. typography) and the fallback |

| Experiment | 无感 | 略降 | 不可接受 | Notes |
|---|---|---|---|---|
| BEST (turbo-int8) | [ ] | [ ] | [ ] | |
| L1 (turbo) | [ ] | [ ] | [ ] | |
| L1O (turbo, official σ) | [ ] | [ ] | [ ] | |
| F1 (ck) | [ ] | [ ] | [ ] | |
| F2 (INT8) | [ ] | [ ] | [ ] | |
| LF1 | [ ] | [ ] | [ ] | |
| LF2 | [ ] | [ ] | [ ] | |
| LF3 | [ ] | [ ] | [ ] | |

**Fallback rule (already documented in the course):** if turbo is rejected
for typography, the fastest *fidelity-approved* config is F2 (51.1 s,
near-lossless, −6.7 GB VRAM). Record that decision alongside the verdict.

## Step 5 — Commit the verdicts

- [ ] Copy the completed table into a new file
      `evidence/quality/VERDICTS.md` (dated, signed by the owner account).
- [ ] Update the affected `quality_status` wording in
      `lab/report.py::_quality_status` and the README/M6 tables in the same
      commit — the status labels and the verdict must move together.
- [ ] The 2048² exploration gate (M7 Task 7) unlocks only after this
      commit lands.

## Reminders (why this stays manual)

- LPIPS did not predict the parent project's typography rejection — the
  gate is human because the metric was demonstrably insufficient there.
- A verdict on 9 pairs per experiment is a *sampling* decision: state it
  as such in VERDICTS.md (it does not certify all prompts/seeds/users).
- New measurements (runs/quality/) never overwrite the frozen grids this
  checklist cites — re-review after new evidence only if you choose to.
