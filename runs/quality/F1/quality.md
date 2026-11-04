# Quality gate — F1

## Review protocol (manifest-declared, never guessed)

- Reference experiment: **B0**
- Baseline: B0 frozen baseline
- Experiment: `/workspace/qwen-image-inference-tuning-course/runs/raw/acceptance-F1-F1`
- LPIPS semantics: FIDELITY — same model, same seed, only the numerics path changed (quantization / attention backend). Lower LPIPS means closer perceptual similarity to the specified baseline image. LPIPS is not an absolute image-quality score; compare against the noise floor before reading anything into small values.
- Fidelity review threshold: per-prompt median LPIPS above **0.10** flags the grid for careful human review. The threshold TRIAGES the review; it never auto-passes or auto-fails anything.
- Quality-review status: **RECORDED_OR_PENDING_HUMAN_REVIEW** (the binding verdict is the owner's three-tier call below)


| Prompt | median LPIPS | max LPIPS | n pairs | review flag |
|---|---|---|---|---|
| p1-zh-scene | 0.0646 | 0.1304 | 3 | below threshold |
| p2-en-scene | 0.0066 | 0.0106 | 3 | below threshold |
| p3-poster-text | 0.0039 | 0.0051 | 3 | below threshold |

## Verdict (user final — 三档终审)

- [ ] 无感 (imperceptible)
- [ ] 略降 (slightly degraded)
- [ ] 不可接受 (unacceptable)

Notes: 

Grids: `quality_grid.png` (256px pairs), `quality_crops_poster.png` (native-res text crops, poster prompt).
