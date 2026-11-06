# Quality gate — BEST

## Review protocol (manifest-declared, never guessed)

- Reference experiment: **B0**
- Baseline: B0 frozen baseline
- Experiment: `/workspace/pr4/runs/raw/q1-075553-BEST`
- LPIPS semantics: DISTANCE, NOT QUALITY — a different model/schedule (Turbo) legitimately produces different images; a high score does not mean worse. The three-tier verdict is the user's 目检 call on the grid.
- Quality-review status: **RECORDED_OR_PENDING_HUMAN_REVIEW** (the binding verdict is the owner's three-tier call below)


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
