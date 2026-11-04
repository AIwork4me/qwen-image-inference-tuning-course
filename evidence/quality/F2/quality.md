# Quality gate — F2

- Baseline: B0 Linux W7900 frozen baseline (bf16 DiT, 40 steps, euler/simple, cfg 1, pytorch SDPA attention)
- Experiment: `bench/results/linux-w7900/raw/p34-F2-F2`
- LPIPS semantics: FIDELITY — same model, same seed, only the numerics path changed (quantization / attention backend). Lower LPIPS means closer perceptual similarity to the specified baseline image. LPIPS is not an absolute image-quality score; compare against the noise floor before reading anything into small values.
- LPIPS noise floor (same-config determinism): **0.0000** — differences below this are noise.

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
