# M6 — Combined Optimization & the Quality Gate

**Problem.** You now have a fast configuration (BEST, 6.12×). Engineering
is not done: *is it acceptable?* — and "acceptable" is not a number an
automation can print. This module assembles the full evidence matrix,
teaches the semantics of the one automated metric we do use (LPIPS), and
hands the final verdict to a human — by design, not by laziness.

**Concepts (beginner glossary).**

| Term | Plain meaning |
|---|---|
| **quality gate** | the rule that no speed change ships without an accept/reject decision on its output |
| **LPIPS** | Learned Perceptual Image Patch Similarity — a neural net's estimate of perceptual difference between two images (0 = identical) |
| **fidelity vs distance** | LPIPS between *same-model numerics variants* measures damage (fidelity); between *different schedules/models* it measures mere difference (distance) — the semantics MUST be declared per experiment |
| **noise floor** | LPIPS between two identical-config runs (here 0.0000 on 9/9 pairs) — differences below it are meaningless |
| **paired grid** | side-by-side baseline|experiment images for human review |
| **native-resolution crops** | full-res zoom-ins — 256 px thumbnails hide exactly the glyph errors that matter for text rendering |
| **three-tier verdict** | 无感 (imperceptible) / 略降 (slightly degraded) / 不可接受 (unacceptable) — the human's binding call |

**Run.**

```bash
# classroom (~4 min GPU): the combined experiment itself
.venv-lab/bin/python -m lab run --experiment BEST --smoke --date m6-smoke

# formal (~4 min GPU): quotable BEST medians (you did the parts in M3/M4)
.venv-lab/bin/python -m lab run --experiment BEST --date m6-best

# quality gates from YOUR raw images (~1 min GPU each; needs your B0 + exp raws)
.venv-lab/bin/python -m lab quality --experiment F2     # fidelity semantics
.venv-lab/bin/python -m lab quality --experiment BEST   # distance semantics
```

The quality command resolves your latest raw image dirs under `runs/raw/`
and writes grids to `runs/quality/<exp>/` — the shipped grids under
`evidence/quality/` belong to the frozen campaign and are never
overwritten. Semantics come from the manifest field
(`quality_semantics`: F2 = fidelity; every schedule-changing experiment =
distance) — the tool refuses to guess.

**Observe (reference campaign — the full matrix).**

| ID | Config | Warm | vs B0 | VRAM | Objective quality | Verdict |
|---|---|---|---|---|---|---|
| B0 | bf16 40-step | 55.21 s | 1.00× | 37.7 GB | reference | — |
| L1 | turbo 8-step | 12.13 s | 4.55× | 37.7 GB | distance 0.19–0.30 | pending human |
| L1O | turbo, official σ | 12.07 s | 4.57× | 37.7 GB | distance 0.23–0.34 | pending human |
| F1 | ck attention | 45.09 s | 1.22× | 37.7 GB | distance 0.004–0.065 | pending human |
| F2 | INT8 40-step | 51.09 s | 1.08× | 31.0 GB | **fidelity 0.005–0.031** | near-lossless band |
| LF1/2/3 | combinations | 10.03/9.02/9.02 s | ≤6.12× | 31–37.7 GB | distance | pending human |
| **BEST** | turbo int8 + default attn | **9.02 s** | **6.12×** | 31.0 GB | distance 0.19–0.30 | **PENDING_HUMAN_REVIEW** |

BEST was re-run in a fresh process: 9.020 / 9.023 / 9.020 s. It does
**not** contain every technique (ck was dropped — it added nothing on
int8 turbo). The re-run landed on a live cloud instance at 9.023 s —
the shipped course's own first smoke image matched the frozen reference
median to the millisecond.

**Explain.** LPIPS semantics — the course's most-quoted foot-gun:

- **Fidelity** (same model, same seed, numerics changed): LPIPS measures
  *damage*. F2's 0.005–0.031 sits in the near-lossless band (noise floor
  0.0000).
- **Distance** (different schedule/model): LPIPS measures *difference*.
  Turbo's 0.19–0.30 does **not** mean "30% worse" — a visually excellent
  turbo image still distances from its 40-step sibling.

Forbidden inferences (each has embarrassed someone, somewhere):
"LPIPS low ⇒ high quality" · "distance high ⇒ bad" · "human saw no
difference on 9 pairs ⇒ all users get identical output".

The binding gate: automation prepares evidence; it never renders the
verdict:

- 256 px paired grids (`evidence/quality/<exp>/quality_grid.png`) —
  overall look;
- **native-resolution typography crops**
  (`quality_crops_poster.png`) — 256 px hides exactly the glyph errors
  that matter for Chinese text rendering;
- determinism noise floor 0.0000 (9/9 same-config cross-session pairs) so
  tiny LPIPS values are interpretable.

In the parent project, the human gate *rejected* a faster variant whose
LPIPS looked fine but whose typography (p3) was judged 不可接受 — LPIPS
did not predict the call. That's why the gate is binding.

The tradeoff slide (lecture favorite): if the owner rejects turbo for
typography, the fastest **fidelity-approved** config in the data is F2
(51.1 s, near-lossless, −6.7 GB). The fastest *unconditionally* is BEST
(9.02 s, verdict pending). "Which should we ship?" is a product decision
the engineer arms with evidence — both rows of it.

**Think (answer keys in `<details>`).**

1. Label each claim VALID or FORBIDDEN, with one sentence why:
   a. "INT8 is near-lossless: LPIPS 0.005–0.031 (fidelity)."
   b. "Turbo images are 30% worse: LPIPS 0.30."
   c. "BEST is 6.12× faster than B0 (same environment, paired protocol)."
   d. "Nobody can tell the difference, so turbo is safe for all users."
   <details><summary>answer</summary>a VALID — correct semantics
   (fidelity, quantization-only variant) and within the near-lossless
   band vs the 0.0000 noise floor. b FORBIDDEN — distance semantics: a
   different schedule legitimately produces different images; LPIPS is
   not a quality percentage. c VALID — paired protocol, committed
   records, warm-median definition stated. d FORBIDDEN — no human study
   supports it AND the verdict is explicitly pending; the gate exists
   precisely so this sentence cannot be written yet.</details>
2. Why does the quality tool write to `runs/quality/` instead of
   `evidence/quality/`?
   <details><summary>answer</summary>`evidence/quality/` holds the frozen
   campaign's grids — the reference every claim cites. Overwriting them
   with student grids would silently rewrite the evidence behind
   published statements. Your grids are *your* evidence, kept separate by
   code.</details>
3. Your BEST smoke produced an image in 9.0 s that you happen to prefer
   over B0's. Does that make BEST "higher quality"?
   <details><summary>answer</summary>No — preference on one sample is
   anecdote, not evidence, and "quality" here is defined *relative to the
   baseline contract* (does the 8-step output remain acceptable for the
   intended use, typography included?). It does make it a useful data
   point for the human reviewer.</details>

**Acceptance.**
- [ ] The VALID/FORBIDDEN answers with reasons
- [ ] Your verdict exercise on
      `evidence/quality/BEST/quality_crops_poster.png` (clearly labeled
      as an exercise — your instructor is not the project owner, and the
      official verdict stays PENDING_HUMAN_REVIEW until the owner casts
      it)
- [ ] The two-row recommendation table (fastest-pending vs
      fastest-fidelity-approved) as you would present it to a team
- [ ] Your own F2 quality grid generated and inspected
