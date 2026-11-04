# P6 — Combined Optimization & Quality Gate (Linux W7900)

**Date:** 2026-11-03 · **BEST rerun:** `p6-BEST-rerun-BEST` (fresh, independent process)

## Experiment matrix (all measured on the physical W7900, warm = median of 9)

| ID | Configuration | Category | Warm (s) | Step (s) | vs B0 | Peak VRAM dev (GB) | Cold (s) | Quality (objective) | Status |
|---|---|---|---|---|---|---|---|---|---|
| B0 | bf16 40 steps, pytorch SDPA | Baseline | 55.21 | 1.345 | 1.00× | 37.7 | 74.2 | reference | PASS |
| L1 | Turbo bf16 8 steps (simple) | Less Computation | 12.13 | 1.337 | 4.55× | 37.7 | 48.4 | distance 0.19–0.30 | PASS · review pending |
| L1O | Turbo bf16 8 steps (official σ) | Less Computation | 12.07 | 1.337 | 4.57× | 37.7 | 50.3 | distance 0.23–0.34 | PASS · review pending |
| F1 | bf16 40 + ck attention | Faster Computation | 45.09 | 1.077 | 1.22× | 37.7 | 84.6 | distance 0.004–0.065 | PASS · review pending |
| F2 | INT8 DiT 40 steps | Faster Computation | 51.09 | 1.220 | 1.08× | **31.0** | 80.3 | **fidelity 0.005–0.031 (near-lossless)** | PASS · review pending |
| LF1 | Turbo bf16 + ck | Combined | 10.03 | 1.067 | 5.50× | 37.7 | 53.7 | distance 0.18–0.30 | PASS · review pending |
| LF2 | Turbo INT8 + default attn | Combined | **9.02** | 0.916 | **6.12×** | **31.0** | **32.4** | distance 0.19–0.30 | PASS · review pending |
| LF3 | Turbo INT8 + ck | Combined | **9.02** | 0.906 | **6.12×** | **31.0** | 30.4 | distance 0.18–0.30 | PASS · review pending |
| H1 | CPU TE variant | Heterogeneous | 261.4 (integrated turbo gen) | — | 0.19× | −16.8 GB TE | — | cond parity cosine ≥ 0.999999 | PASS (negative result) |
| H2 | 4-request pipeline | Throughput | 51.1 s seq vs 737.1 s pipelined | — | 14× worse pipelined | — | — | overlap proven 50.4 s | PASS (negative result) |
| F4 | torch.compile | Optional | — | — | — | — | — | ceiling < 6% of DiT | **NOT_RUN** (rejected with P2 evidence) |
| **BEST** | **Turbo INT8 + default attn (= LF2)** | Final | **9.02** (rerun: 9.02) | 0.916 | **6.12×** | **31.0** | 32.4 | distance 0.19–0.30 | **PASS · PENDING_HUMAN_REVIEW** |

No infeasible combinations were silently dropped; every cell above is a
physical measurement except the two explicitly marked NOT_RUN-with-reason.

## BEST selection rationale

LF2 and LF3 measured identical warm latency (9.02 s); **BEST = LF2
configuration** (no `--use-ck-attention` flag): equal speed, simpler stack,
one less backend dependency, and identical objective-quality distances.
The best configuration does **not** contain all techniques — ck attention,
the single best F-level lever on bf16, contributes nothing on the int8 turbo
pipeline (measured, P4).

## Independent rerun evidence (fresh process, fresh server)

`p6-BEST-rerun-BEST`: warm 9.020/9.023/9.020 s per prompt (9 runs,
spread 0.03%), DiT 7.38–7.44 s, fresh-prompt totals 9.0–10.0 s,
VRAM 27.6–31.0 GB — matching the original LF2 session exactly.

## Final metrics (BEST, all latency definitions kept separate)

| Metric | Value |
|---|---|
| Warm generation (same prompt cached) | **9.02 s** (6.12× vs B0's 55.21 s) |
| Fresh-prompt generation | 9.0–10.0 s (TE adds 0.31–0.60 s) |
| Full cold start (launch → 1st PNG) | 32.4 s (models load ~23 s of it; server boot 7 s separate) |
| Throughput (sequential service) | ≈ 6.6 img/min with fresh prompts (H2 arm A measured 4.69 img/min on turbo bf16) |
| Peak VRAM (device, sampled) | 31.0 GB of 48 GB |
| Weights on disk | 7.26 + 17.53 + 0.68 = 25.5 GB |
| Quality verdict | **PENDING_HUMAN_REVIEW** (grids + crops committed; LPIPS recorded) |

## Quality gate state

- Objective layer computed on the frozen 3-prompt × 3-seed pairs for every
  experiment (`evidence/results/quality-*/quality.md`, grids, native
  poster crops). Noise floor 0.0000 (9/9).
- **F2 (INT8) is the only near-lossless-band variant** (fidelity 0.005–0.031) —
  consistent with the historical Windows int8 verdicts.
- All turbo variants (L1/L1O/LF1/LF2/LF3/BEST) are distance-semantics: LPIPS
  0.18–0.34 measures *difference from the 40-step baseline*, **not** quality.
  The binding three-tier verdict (无感 / 略降 / 不可接受) requires the owner's
  eyes on `quality-BEST/quality_grid.png` + `quality_crops_poster.png`
  (typography at native resolution). **No quality gate is marked passed
  without it.**
- If the owner rejects the turbo schedule for the typography prompt, the
  fastest quality-approved fallback in this lab's data is F2 (INT8 40-step,
  near-lossless fidelity, 51.1 s, −6.7 GB VRAM) — that decision is the
  owner's, explicitly not pre-made here.

## Reproduce

```bash
python -m lab run --experiment BEST --date <tag>
```
