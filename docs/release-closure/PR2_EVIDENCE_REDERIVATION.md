# PR #2 — Phase P1: Independent Evidence Rederivation

Machine-readable companion: `docs/release-closure/PR2_EVIDENCE_REDERIVATION.json`, produced by
`scripts/release-closure/recompute_evidence.py` (reads **only** raw probe JSONs, conditioning
safetensors, per-request pipeline/e2e timestamps; never docs/ summaries).

Classification legend: **[RR]** independently reproduced from raw evidence · **[DR]** verified only
against committed derived records · **[HL]** supported only by historical host logs · **[NV]** not
independently verifiable · **[CT]** contradicted · **[OD]** outdated.

## 1. Encoder CPU A/B (raw: `baseline/probe_cpu.json`, `ov/run_*/probe_*.json`)

Warm medians recomputed from raw rep rows (C0 n=1 warm/prompt, OV arms n=2 warm/prompt):

| Arm | p1 (66 tok) | p2 (51 tok) | p3 (78 tok) | negative (22→8 tok) | RSS peak |
|---|---:|---:|---:|---:|---:|
| C0 PyTorch CPU (BF16 weights, FP32 manual cast) | 176.02 s | 136.81 s | 207.45 s | 61.51 s | 34.5 GB |
| C1a OV FP32 | 0.779 s* | 0.648 s* | 0.875 s* | 0.398 s* | 50.5 GB |
| C1b OV BF16 deployment (FP16-stored IR) | 0.532 s | 0.285 s | 0.555 s | 0.187 s | 30.0 GB |
| C2 OV INT8 | 0.361 s | 0.249 s | 0.406 s | 0.144 s | 16.8 GB |

\* from committed derived record (values cross-checked against PR comment); all other cells recomputed from raw rows this session.

Speedups vs C0 (median-warm ratio): C1b deployment **330.9 / 480.4 / 373.5 / 328.5×** — the
"331–480×" claim **[RR]**. C2 487.5–549.4×, C1a 154.6–237.2× — **[RR]**.
Sample-size caveat: C0 n=1, OV n=2 per prompt, single session, single host → the *order of
magnitude* is solid (2–3 orders), specific ratios carry wide uncertainty. This is the P1-D gap.

Baseline name check: C0 is **ComfyUI PyTorch CPU with BF16-stored weights and FP32 manual-cast
compute** (probe records `torch.set_num_threads(16)`, 16-thread parity) — **[RR]**; docs must not
generalize the ratio to "OpenVINO vs optimized PyTorch CPU".

## 2. Conditioning numerics (raw: `*/cond/*.safetensors`, recomputed in float64)

| Pair | p1 relL2 | p2 | p3 | negative | Gate (threshold) |
|---|---:|---:|---:|---:|---|
| G_vs_C0 | 7.38e-6 | 6.93e-6 | 6.68e-6 | 3.10e-5 | reference **[RR]** |
| C1a_vs_C0 | 3.52e-6 | 3.44e-6 | 3.03e-6 | 5.43e-6 | PASS (≤1e-4, cos≥0.99999) **[RR]** |
| C1b_deployment_vs_C0 | 1.253e-2 | 1.152e-2 | 1.091e-2 | **1.915e-2** | PASS (≤2e-2, cos≥0.999) **[RR]** |
| C1b_uncompressed_vs_C0 | 1.315e-2 | 1.238e-2 | 1.158e-2 | **6.634e-2** | FAIL **[RR]** |
| C2_vs_C1a | 6.38e-2 | 6.36e-2 | 5.62e-2 | **2.825e-1** (cos 0.9608) | FAIL **[RR]** |

All committed verdicts in `alignment/conditioning_alignment.json` are byte-consistent with
independent recomputation from the raw tensors. Determinism (identical cond SHA256 across reps)
reproduced for every arm/prompt **[RR]**. Long-sequence stress (180–526 tok) re-verified from
committed NPZ conditioning: C1b 2.2–10.8% relL2, C2 7.5–18.4%, growing with length — descriptive
only, not a product gate **[RR]**.

## 3. Token contract / empty-negative (raw probe ids)

- Production `clip.tokenize("")` → 22 ids `[…, 151644, 872, 198, 151645, 198, 151644, 77091, 198]` — no `220`.
- Manual diagnostic (`prevent_empty_text=True`) → same ids **plus `220` inserted at index 17** (23 ids) — the literal word "empty".
- OV encoder (`prevent_empty_text=False`) → identical to production for all 4 prompts **[RR]**.
- Pinned ComfyUI `comfy/text_encoders/qwen_image21.py:18` default is `prevent_empty_text=False` → OV matches the runtime that actually conditions generation; the *diagnostic* deviated. `token_ids_self_consistent=false` for G/C0 is therefore a **diagnostic artifact**, not a semantic mismatch; the correct production-contract flag must compare like-for-like (fix in P0-C).

## 4. Pipeline four-request campaign (raw: `pipeline/pipeline_ab.json` per-request timestamps)

Recomputed from per-request walls/timestamps:

- Arm A (GPU-native sequential): request walls `26.065, 13.033, 12.792, 13.272` s; sum **65.162 s**;
  reported batch total 66.615 s (includes ~1.45 s inter-request overhead) — both **[RR]**.
  First-request penalty **26.065 s**; steady median **13.032 s**; steady-4 estimate 52.128 s.
- Arm C (OV BF16 deployment worker): gen walls `13.029, 12.086, 12.154, 12.216` s; total
  **50.046 s**; gen median **12.185 s**; CPU-encode per request `0.40, 0.77, 0.44, 0` s (the
  shared empty-negative is encoded once — req4 reuses it); proven overlap Σ=**1.621 s** **[RR]**.
- Arm D (INT8): total 49.749 s, gen median 12.131 s, overlap Σ=1.401 s **[RR]**.
- Arm B (PyTorch CPU TE): total 680.004 s, fill 176.17 s, encode median 147.14 s, overlap Σ=37.09 s —
  CPU-bound, matches committed summary **[RR]**.
- C vs A: **1.302×** (sum-of-walls) / **1.331×** (batch totals) cold-inclusive; **1.042×**
  steady — but the steady figure is an *estimate* (A steady median × 4 vs C measured total), and
  A's "cold" is dominated by a **recurring** ~26 s first-request GPU CLIP reload (the separate
  warm session `e2e/warm_gpu_native.json` again shows 26.06 s on its first request **[RR]**).
  → P1-E: a genuinely balanced warm A/C comparison was NOT measured; the "≈4% steady gain" is
  currently an estimate, honestly labeled in docs but worth a direct experiment (P4).
- No measurable GPU slowdown from concurrent 16-thread CPU encoding (B/C/D gen medians within
  ~1 s of A steady) **[RR]**.

## 5. E2E latency / VRAM (raw: `e2e/e2e_ab.json` per-request records)

- G_native wall median 13.031 s (3×cfg1), VRAM peak median 34.15 GB, residency 33.09 GB **[RR]**.
- C1b_OV_BF16 precomputed wall median 12.029 s, peak 16.77 GB, residency 16.29 GB **[RR]**.
- Full-pipeline VRAM peak saving: **17.26 GB** (committed, warm-session native peak 34.03 GB vs
  precomputed 16.77 GB) **[DR+RR]** — recomputation across request sets gives 16.8–17.4 GB
  depending on which requests form the median; the 17.26 GB figure corresponds to the documented
  warm-session definition and is reproducible from the named records. Scope: turbo BF16 DiT,
  8 steps, 1024², cfg=1 — **not** extrapolable to 40-step or other resolutions.
- E2E wall delta (precomputed vs native, same session): ≈ +1.0 s *faster* for precomputed
  (12.03 vs 13.03 s median) because GPU TE (~0.6 s) plus its allocation effects are removed;
  serialized-with-encode estimate ≈ 12.58 s **[DR]**.

## 6. Automated image metrics (raw: `e2e/e2e_ab.json` quality block; images on disk)

- C1b vs G: PSNR 35.57–47.03 dB, SSIM 0.9876–0.9972, LPIPS(Alex) 0.0019–0.0080 **[RR]**.
- C1a comparable; C2 worst (p1 PSNR 27.37 dB, LPIPS 0.0378) **[RR]** — consistent with the C2
  negative-conditioning failure; C2 must remain non-default.
- C0-native p1 LPIPS 0.0090 (n=1 prompt) **[DR]**.
- **Human semantic-quality verdict: PENDING_HUMAN_REVIEW — unchanged.** No automated metric
  substitutes for it.

## 7. Provenance addendum (source attribution of the 2026-11-05 campaign)

See `PR2_P0_BASELINE.md` §4. Records written 08:09–10:34 UTC carry `source_head=20a7f7e`
(bootstrap) because the lab implementation was still uncommitted working-tree code (committed
10:39 as `840c484`); the 10:56 formal CLI run is a post-commit replay at `9283778` (committed as
`6aacd6f`) and reproduces the encoder arm. Historical records were not modified; this addendum is
the authoritative interpretation. All NEW measurements in this audit carry their true HEAD.

## 8. Claim scoreboard

| Claim | Class | Note |
|---|---|---|
| C0 136.81–207.45 s (3 positives) | **[RR]** | n=1 warm/prompt |
| C1b 0.285–0.555 s | **[RR]** | n=2 warm/prompt |
| Encoder speedup 331–480× | **[RR]** | order-of-magnitude solid; exact ratios low-n |
| C1b positive relL2 1.09–1.25% | **[RR]** | |
| C1b deployment negative relL2 1.91% PASS | **[RR]** | FP16-stored IR |
| C1b uncompressed negative 6.63% FAIL | **[RR]** | FP32-stored IR, BF16 hint |
| C2 negative 28.25% FAIL | **[RR]** | cos 0.9608 |
| VRAM peak saving ≈17.26 GB | **[RR]/[DR]** | 16.8–17.4 GB across definitions; scope turbo/8-step/1024²/cfg1 |
| 4-req C1b 50.05 s vs A 66.62 s (1.33×) | **[RR]** | 1.30× by wall-sum; A cold = recurring CLIP reload |
| Warm steady gain ≈4% | **[CT→estimate]** | derived estimate, not a measured balanced warm A/C pair — P4 re-measures |
| Overlap proven (B 37.1 s / C 1.62 s / D 1.40 s) | **[RR]** | |
| No GPU slowdown under CPU contention | **[RR]** | within ~1 s; single session |
| pytest 53 passed CPU | to re-verify in P7 | CI green at HEAD **[DR]** |
| Human image quality | **PENDING** | not automated, by design |
