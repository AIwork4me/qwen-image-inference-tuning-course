# Quality gate — BEST

- Baseline: B0 frozen baseline
- Experiment: `/workspace/qwen-image-inference-tuning-course/runs/raw/acceptance-BEST-BEST`
- LPIPS semantics: DISTANCE, NOT QUALITY — a different model/schedule (Turbo) legitimately produces different images; a high score does not mean worse. The three-tier verdict is the user's 目检 call on the grid.

| Prompt | median LPIPS | max LPIPS | n pairs |
|---|---|---|---|
| p1-zh-scene | 0.3016 | 0.3425 | 3 |
| p2-en-scene | 0.1900 | 0.2048 | 3 |
| p3-poster-text | 0.2094 | 0.2691 | 3 |

## Verdict (user final — 三档终审)

- [ ] 无感 (imperceptible)
- [ ] 略降 (slightly degraded)
- [ ] 不可接受 (unacceptable)

Notes: 

Grids: `quality_grid.png` (256px pairs), `quality_crops_poster.png` (native-res text crops, poster prompt).
