# GPU Performance Matrix — physical W7900 qualification campaign

**GPU:** AMD Radeon PRO W7900 (gfx1100, 48 GB), single device, Radeon Cloud ·
**ROCm** hip 7.2.53211 · **PyTorch** 2.9.1+gitff65f5b · **Campaign run prefix** `q1-075553-*` ·
all numbers below are **MEASURED on this host in this session** unless marked `HISTORICAL` (frozen campaign) or `DERIVED`.

## Full-protocol A/B (3 prompts × 3 seeds, warmup+3 measured each, fresh-prompt arm; cold separated)

| Comparison | Baseline run | Candidate run | Median B→C (s) | Speedup (measured) | Reference (HISTORICAL) | VRAM Δ (device peak) | Verdict |
|---|---|---|---|---|---|---|---|
| M3 less-computation B0 vs L1 (Turbo 8 steps) | `q1-075553-B0` | `q1-075553-L1` | 55.096 → 12.029 (n=9 each, formal) | **4.58×** `FORMAL_PROTOCOL_MEASURED` | 4.55× (55.206/12.132) | −0.5 MB (both ~38.9 GB) | PASS (not the naive 5×; confound: model+schedule+steps) |
| M4 attention B0 vs F1 (ck backend) | `q1-075553-B0` | `q1-075553-F1` | 55.096 → 43.247 | **1.274×** | 1.225× | −3 MB (≈38.9 GB) | PASS — ck backend verified ACTIVE from server-log evidence (Comfy-Kitchen + AOTriton efficient attention) |
| M4 quantization B0 vs F2 (INT8 40-step) | `q1-075553-B0` | `q1-075553-F2` | 55.096 → 50.104 | **1.100×** | 1.081× | **−6.66 GB** (38.90 → 32.24 GB) | PASS — the INT8 win is mainly VRAM, exactly as taught |
| M6 combined B0 vs BEST (Turbo+INT8) | `q1-075553-B0` | `q1-075553-BEST` | 55.096 → 9.022 | **6.107×** | 6.118× | **−6.66 GB** (38.90 → 32.24 GB) | PASS — ≠ F1×F2×L1 multiplication |

Per-arm spreads (max−min of 9 warm): B0 0.015 s · L1 2.006 s (one outlier run) · F1 0.884 s · F2 1.007 s · BEST 0.006 s.

## Profiling (M2) — `q1-075553-profile` (rocprofv3 1.1.0, traced server)

- Pipeline (from own records): total 55.096 s = DiT 53.463 (97.0%) + VAE 0.897 + TE-cached 0 + other 0.736 (DERIVED by subtraction); fresh TE 0.554 s; cold 75.21 s; steady step 1.333 s/40 steps.
- Kernel trace (DiT window): 95,600 kernels, **GPU busy 97.9%** — **GEMM 61.2% · attention 32.5% · elementwise 4.7% · norm 1.3%** (frozen campaign: 61.3/32.4/97.8 — reproduced within 0.1 pp).
- Kernel trace (TE window): 99,445 kernels, busy 81.7%.
- DERIVED Amdahl ceiling from measured g=0.612: infinitely fast GEMM caps kernel-layer speedup at 1/(1−g) = **2.58×** (kernel-busy scope only, not end-to-end).
- Traced runs include interception overhead — composition analysis only, never a performance claim (recorded in the analysis JSON).

## Quality gates (this campaign's own paired images, LPIPS(alex))

| Experiment | Semantics | per-prompt median LPIPS | Human verdict |
|---|---|---|---|
| F2 vs B0 | fidelity (numerics variant) | 0.0207 / 0.0093 / 0.0049 | PENDING_HUMAN_REVIEW (grids + native-res poster crops produced) |
| L1 vs B0 | distance (model+schedule) | 0.2984 / 0.1865 / 0.2124 | PENDING_HUMAN_REVIEW |
| BEST vs B0 | distance (combined) | 0.3016 / 0.1900 / 0.2094 | PENDING_HUMAN_REVIEW — the historical M6 BEST independent review also remains PENDING_HUMAN_REVIEW; no auto-approval anywhere |

## M5 heterogeneous results

See `runs/q1-075553-*` records (H1/H2, m51-te-cpu/cuda, H3_OV_BF16, m51 pipeline, fairab) — filled in `campaign_status.json` and summarized in `gpu_ab_summary.json`.

- Frozen reference (HISTORICAL, cleanroom campaign): PyTorch CPU TE ≈ 154–480× slower than GPU TE on the same 16-thread host; OpenVINO BF16 recovers ~331–480× vs PyTorch CPU; INT8 encoder numerical **FAIL stays FAIL**; fair 2×2 factorial: architecture effect ≈1.02–1.03× under matched fast clients, client-monitoring effect up to ≈4.9%.
- Current-campaign values: see `gpu_ab_summary.json` (kept separate from the historical ones; never pooled).

## Limitations

1. Single physical host, single W7900, one session per arm; no cross-instance claim.
2. Speedups are ratios of the declared populations only; classroom populations are labelled `BOUNDED_PHYSICAL_COMPARISON`.
3. GPU-utilization/VRAM peaks are periodic samples (lower bounds on transients).
4. The pipeline layer of `lab profile` pools this repo's committed `acceptance-b0-B0` warm records (same experiment family, prior session on a W7900) with the campaign's own — medians agree to <0.2%, noted for transparency.
5. Quality verdicts remain human (three-tier rubric on grids); LPIPS triages, never approves.
