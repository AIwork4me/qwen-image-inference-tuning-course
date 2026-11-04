# M6 — Combined Optimization & the Quality Gate

**Goal:** pick the best configuration under a *binding* quality gate, and
learn why the fastest config is not automatically the recommended one.

## Lesson

### The full matrix (all measured, warm = median of 9)

| ID | Config | Warm | vs B0 | VRAM | Objective quality | Verdict |
|---|---|---|---|---|---|---|
| B0 | bf16 40-step | 55.21 s | 1.00× | 37.7 GB | reference | — |
| L1 | turbo 8-step | 12.13 s | 4.55× | 37.7 GB | distance 0.19–0.30 | pending human |
| L1O | turbo, official σ | 12.07 s | 4.57× | 37.7 GB | distance 0.23–0.34 | pending human |
| F1 | ck attention | 45.09 s | 1.22× | 37.7 GB | distance 0.004–0.065 | pending human |
| F2 | INT8 40-step | 51.09 s | 1.08× | 31.0 GB | **fidelity 0.005–0.031** | near-lossless band |
| LF1/2/3 | combinations | 10.03/9.02/9.02 s | ≤6.12× | 31–37.7 GB | distance | pending human |
| **BEST** | turbo int8 + default attn | **9.02 s** | **6.12×** | 31.0 GB | distance 0.19–0.30 | **PENDING_HUMAN_REVIEW** |

BEST was re-run in a fresh process: 9.020 / 9.023 / 9.020 s. It does **not**
contain every technique (ck was dropped — it added nothing on int8 turbo).

### LPIPS semantics — the course's most-quoted foot-gun

- **Fidelity** (same model, same seed, numerics changed): LPIPS measures
  *damage*. F2's 0.005–0.031 sits in the near-lossless band.
- **Distance** (different schedule/model): LPIPS measures *difference*.
  Turbo's 0.19–0.30 does **not** mean "30% worse" — a visually excellent
  turbo image still distances from its 40-step sibling.

Forbidden inferences (each has embarrassed someone, somewhere):
"LPIPS low ⇒ high quality" · "distance high ⇒ bad" · "human saw no
difference on 9 pairs ⇒ all users get identical output".

### The binding gate: human three-tier verdict (无感 / 略降 / 不可接受)

Automation prepares evidence; it never renders the verdict:

- 256 px paired grids (`evidence/quality/<exp>/quality_grid.png`) — overall look;
- **native-resolution typography crops** — 256 px hides exactly the glyph
  errors that matter for Chinese text rendering;
- determinism noise floor 0.0000 (9/9 same-config cross-session pairs) so
  tiny LPIPS values are interpretable.

In the parent project, the human gate *rejected* a faster variant whose
LPIPS looked fine but whose typography (p3) was judged 不可接受 —
LPIPS did not predict the call. That's why the gate is binding.

### The tradeoff slide (lecture favorite)

If the owner rejects turbo for typography, the fastest **fidelity-approved**
config in the data is F2 (51.1 s, near-lossless, −6.7 GB). The fastest
*unconditionally* is BEST (9.02 s, verdict pending). "Which should we ship?"
is a product decision the engineer arms with evidence — both rows of it.

## Exercise

1. Run the quality gate: `python -m lab quality --experiment F2` (and for
   one turbo variant). Open the grid and the poster crops.
2. Label each of these claims VALID or FORBIDDEN, with one sentence why:
   a. "INT8 is near-lossless: LPIPS 0.005–0.031 (fidelity)."
   b. "Turbo images are 30% worse: LPIPS 0.30."
   c. "BEST is 6.12× faster than B0 (same environment, paired protocol)."
   d. "Nobody can tell the difference, so turbo is safe for all users."
3. You are the owner. Look at `evidence/quality/BEST/quality_crops_poster.png`
   vs the baseline row. Record your three-tier verdict *and* note that your
   instructor is not the project owner — the verdict is an exercise, not data.
4. Write the two-row recommendation table (fastest-pending vs
   fastest-fidelity-approved) as you would present it to a team.

**Acceptance:** the VALID/FORBIDDEN answers + your verdict with reasoning.
