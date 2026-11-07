# Quality gate — F2

## Review protocol (manifest-declared, never guessed)

- Reference experiment: **B0**
- Baseline: B0 frozen baseline
- Experiment: `/workspace/pr4/runs/raw/q1-075553-F2`
- LPIPS semantics: FIDELITY — same model, same seed, only the numerics path changed (quantization / attention backend). Lower LPIPS means closer perceptual similarity to the specified baseline image. LPIPS is not an absolute image-quality score; compare against the noise floor before reading anything into small values.
- Fidelity review threshold: per-prompt median LPIPS above **0.10** flags the grid for careful human review. The threshold TRIAGES the review; it never auto-passes or auto-fails anything.
- Quality-review status: **RECORDED_OR_PENDING_HUMAN_REVIEW** (the binding verdict is the owner's three-tier call below)


| Prompt | median LPIPS | max LPIPS | n pairs | review flag |
|---|---|---|---|---|
| p1-zh-scene | 0.0207 | 0.0302 | 3 | below threshold |
| p2-en-scene | 0.0093 | 0.0104 | 3 | below threshold |
| p3-poster-text | 0.0049 | 0.0311 | 3 | below threshold |

## Verdict (user final — 三档终审)

- [ ] 无感 (imperceptible)
- [ ] 略降 (slightly degraded)
- [ ] 不可接受 (unacceptable)

Notes: 

Grids: `quality_grid.png` (256px pairs), `quality_crops_poster.png` (native-res text crops, poster prompt).
