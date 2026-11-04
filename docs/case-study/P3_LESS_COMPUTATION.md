# P3 — Less Computation (Linux W7900)

**Date:** 2026-11-03 · **Runs:** `evidence/results/p34-L1-L1` (historical protocol), `p3-L1O-L1O` (official schedule) · **Quality:** `quality-L1/`, `quality-L1O/`

## Experiment L1 — Turbo 8-step diffusion (historical repo protocol)

B0 (bf16, 40 steps, warm median 55.21 s) vs L1 (turbo bf16 DiT, **8 steps**,
euler/simple/cfg 1 — identical to the repo's historical E1 protocol). Same TE,
VAE, prompts, seeds, resolution, attention backend, server lifecycle.

| Metric | B0 | L1 | Δ |
|---|---|---|---|
| Warm total (median, n=9) | 55.21 s | **12.13 s** | **4.55×** |
| DiT denoise (KSampler) | 53.88 s | 10.78 s | 5.00× |
| Steady step time | 1.345 s | 1.337 s | ≈1.0× (unchanged, as expected) |
| VAE decode | 0.90 s | 0.90 s | — |
| TE fresh | 0.56 s | 0.55 s | — |
| Fresh-prompt total | 56.11 s | 13.03 s | 4.31× |
| Full cold start | 74.2 s | 48.4 s | 1.53× (model-load bound) |
| Peak device VRAM | 37.7 GB | 37.7 GB | — |

**Attribution:** the entire gain comes from removing 32 of 40 denoising steps —
per-step compute is unchanged (1.337 vs 1.345 s), TE/VAE untouched. No execution-
behavior change is claimed.

**Why not 5.0× end-to-end?** DiT was 97.6% of the warm total; the remaining
~1.4 s (VAE + fixed costs) caps the speedup at 55.2/(1.35×8+1.4) ≈ 4.6× —
Amdahl in practice, measured 4.55×.

## Experiment L1O — official Turbo protocol (ManualSigmas schedule)

The Comfy-Org official Qwen-Image-2.1 Turbo template
(`Comfy-Org/workflow_templates/templates/image_qwen_image_2_1_turbo_t2i.json`)
samples with ManualSigmas `1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148,
0.704534, 0.414568, 0` + euler + CFG 1 (its full stack also uses int8 DiT/TE;
L1O isolates the **schedule variable** on the same turbo bf16 weights as L1).

| Metric | L1 (simple) | L1O (official sigmas) |
|---|---|---|
| Warm total | 12.13 s | 12.07 s |
| Steady step time | 1.337 s | 1.337 s |
| LPIPS vs B0 (p1/p2/p3 medians) | 0.298 / 0.187 / 0.212 | 0.336 / 0.226 / 0.317 |

Same step count → same compute (timing identical, as predicted). The schedule
changes **numerics**, not FLOPs: L1O images distance slightly further from the
40-step baseline than L1's. This is a *distance*, not a quality ranking —
both variants' grids (`quality-L1O/quality_grid.png`, native poster crops)
await the human three-tier verdict (无感 / 略降 / 不可接受).

## Quality (objective layer; human verdict pending)

- Determinism noise floor: **LPIPS 0.0000** (9/9 B0-r1 vs B0-r2 pairs).
- L1/L1O are **distance-semantics** experiments (different schedule ⇒ different
  images by design). Turbo is **not** an exact-output or lossless optimization.
- Grids + native-res typography crops committed under `quality-L1*/` for review.

## Reproduce

```bash
python -m lab run --experiment L1  --date <tag>
python -m lab run --experiment L1O --date <tag>
```
