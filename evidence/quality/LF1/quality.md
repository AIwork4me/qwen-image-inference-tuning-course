# Quality gate — LF1

- Baseline: B0 Linux W7900 frozen baseline (bf16 DiT, 40 steps, euler/simple, cfg 1, pytorch SDPA attention)
- Experiment: `bench/results/linux-w7900/raw/p34-LF1-LF1`
- LPIPS semantics: DISTANCE, NOT QUALITY — a different model/schedule (Turbo) legitimately produces different images; a high score does not mean worse. The three-tier verdict is the user's 目检 call on the grid.
- LPIPS noise floor (same-config determinism): **0.0000** — differences below this are noise.

| Prompt | median LPIPS | max LPIPS | n pairs |
|---|---|---|---|
| p1-zh-scene | 0.2993 | 0.3455 | 3 |
| p2-en-scene | 0.1824 | 0.2029 | 3 |
| p3-poster-text | 0.2094 | 0.2692 | 3 |

## Verdict (user final — 三档终审)

- [ ] 无感 (imperceptible)
- [ ] 略降 (slightly degraded)
- [ ] 不可接受 (unacceptable)

Notes: 

Grids: `quality_grid.png` (256px pairs), `quality_crops_poster.png` (native-res text crops, poster prompt).
