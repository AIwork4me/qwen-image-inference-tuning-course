# Quality gate — F2

- Baseline: B0 frozen baseline
- Experiment: `/workspace/qwen-image-inference-tuning-course/runs/raw/acceptance-F2-F2`
- LPIPS semantics: FIDELITY — same model, same seed, only the numerics path changed (quantization / attention backend). Lower LPIPS means closer perceptual similarity to the specified baseline image. LPIPS is not an absolute image-quality score; compare against the noise floor before reading anything into small values.

| Prompt | median LPIPS | max LPIPS | n pairs |
|---|---|---|---|
| p1-zh-scene | 0.0207 | 0.0302 | 3 |
| p2-en-scene | 0.0093 | 0.0104 | 3 |
| p3-poster-text | 0.0049 | 0.0311 | 3 |

## Verdict (user final — 三档终审)

- [ ] 无感 (imperceptible)
- [ ] 略降 (slightly degraded)
- [ ] 不可接受 (unacceptable)

Notes: 

Grids: `quality_grid.png` (256px pairs), `quality_crops_poster.png` (native-res text crops, poster prompt).
