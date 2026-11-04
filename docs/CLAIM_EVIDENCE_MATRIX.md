# Claim → Evidence Matrix (Linux W7900 Lab)

Every performance/quality claim made by this workstream, with the exact
evidence file that substantiates it. Claims without a row here must not be
made. All evidence under `evidence/results/` unless noted.

| # | Claim | Value | Evidence | Caveats |
|---|---|---|---|---|
| 1 | Frozen baseline warm latency (B0) | 55.21 s (median of 9; two sessions 55.24/55.11) | `p1-baseline-B0/`, `p1-baseline-r2-B0/*.json` | warm = same-prompt node-cache hit; fresh +0.3–0.6 s TE |
| 2 | Baseline steady step time | 1.345 s (min 1.336 / max 1.367) | same records, `steady_step_median_s` | step[0] excluded (protocol v1.3) |
| 3 | Baseline pipeline split | DiT 53.88 s (97.6%) · VAE 0.90 · TE 0.56 fresh · other 0.43 | `p2-profile-profile/pipeline_breakdown.json` | derived from 18 warm records |
| 4 | Kernel composition of one warm gen | GEMM 61.3% / attention 32.4% / elementwise 4.7% / norm 1.3% of GPU busy 97.8% | `p2-profile-profile/rocprof-dit/analysis.json` (+ raw CSVs, not committed) | traced run: composition only, interception overhead ~1%; TE phase outside window (measured at pipeline layer instead) |
| 5 | Host/transfer overhead ≈ 0 | 197 memcpy APIs (~36 µs each), 1 sync per gen | same trace, `hip_api` analysis | models resident; no PCIe shuffling |
| 6 | Turbo 8-step speedup (L1) | 12.13 s = 4.55× vs B0 warm | `p34-L1-L1/*.json` | schedule change → distance semantics; not lossless |
| 7 | Official Turbo schedule (L1O) | 12.07 s; identical step cost; distances 0.23–0.34 | `p3-L1O-L1O/*.json` | ManualSigmas from Comfy-Org template (source cited in P3) |
| 8 | ck attention gain (F1) | 45.09 s = 1.22×; DiT 1.25×; step 1.345→1.077 s | `p34-F1-F1/*.json` + server log "Using Comfy Kitchen attention" | backend evidence re-scraped into records after pattern fix |
| 9 | INT8 DiT (F2) | 51.09 s = 1.08×; VRAM 37.7→31.0 GB; weights −49% | `p34-F2-F2/*.json` | memory-efficiency improvement; speed modest, as warned |
| 10 | INT8 fidelity near-lossless | LPIPS 0.005/0.009/0.021 (p3/p2/p1) | `quality-F2/quality.md` + grids | fidelity semantics; noise floor 0.0000 |
| 11 | Combinations (LF1/LF2/LF3) | 10.03 / 9.02 / 9.02 s | `p34-LF*/` | measured end-to-end; ratios never multiplied |
| 12 | ck adds nothing on int8 turbo | LF2 = LF3 = 9.02 s | `p34-LF2-LF2`, `p34-LF3-LF3` | both sessions, 9 runs each |
| 13 | BEST config & independent rerun | 9.02 s warm; rerun 9.020/9.023/9.020 s | `p6-BEST-rerun-BEST/*.json` | fresh process, fresh server |
| 14 | BEST speedup vs B0 | 6.12× (55.21/9.02) | claims 1 + 13 | same environment, paired protocol |
| 15 | BEST cold start | 32.4 s | `p34-LF2-LF2` cold record | first-gen model load included; server boot 7.0 s separate |
| 16 | CPU TE: 152–391× slower, cond parity cosine ≥ 0.999999, VRAM −16.8 GB | see values | `p5-H1-r2-H1-cpu-te/H1_RESULTS.json` | probe + server-integrated variant |
| 17 | Pipeline A/B: 51.1 s vs 737.1 s; overlap 50.4 s proven | 4 images | `p5-H2-H2-pipeline/H2_RESULTS.json` | timestamps per request in the same file |
| 18 | CPU contention slows GPU gens ~1.5× | 12–13 s → 19.1 s under CPU TE load | same H2 file, `gen_wall_s` | single-GPU host contention |
| 19 | Determinism noise floor | LPIPS 0.0000 (9/9) | `noise_floor.json` | B0 r1 vs r2 (cross-restart); file hashes differ, perceptually identical |
| 20 | GPU is compute-bound (not launch/transfer bound) | GPU busy 97.8% | claim 4 evidence | — |
| 21 | torch.compile rejected | NOT_RUN | P4 §F4 rationale (ceiling <6% of DiT from P2 trace) | decision-with-evidence, no measurement claimed |
| 22 | Weights integrity | 6/6 files SHA256-match pinned digests | `evidence/results/doctor.json` + P0 §3 | anchor = in-repo pinned table |
| 23 | ComfyUI pinned | v0.43.0 @ b5a4321, 1329/1329 files size-verified | P0 §2 + `ComfyUI/.linux_lab_pinned_commit` | git transport blocked; tree-API verification |
| 24 | Quality verdicts (turbo family) | **PENDING_HUMAN_REVIEW** | `quality-*/quality_grid.png`, `quality_crops_poster.png` | no PASS claimed; grids committed for the owner |
| 25 | Historical Windows numbers | background only | `bench/results/2026-*` (untouched) | never substituted for Linux measurements |

**Forbidden inferences (unchanged from the project methodology):** LPIPS low
⇒ high quality; distance high ⇒ bad; human verdict on 9 pairs ⇒ all users;
any claim about prompts never run.
