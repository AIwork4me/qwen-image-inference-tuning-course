# M4 — Faster Computation

**Problem.** The workload is now minimal (8 steps), but each step still
executes the same arithmetic. Two independent levers remain: *choose
better kernels* (attention backend) and *shrink the numbers themselves*
(INT8 weights). Each is a clean A/B against the same baseline — and their
combination is where the course's most-quoted lesson lives: independent
gains never multiply.

**Concepts (beginner glossary).**

| Term | Plain meaning |
|---|---|
| **attention backend** | the implementation behind the attention math — same equations, different kernels (pytorch SDPA vs Comfy Kitchen "ck") |
| **SDPA / flash attention** | PyTorch's fused scaled-dot-product attention; "flash" = a memory-efficient tiling strategy |
| **INT8 quantization** | storing weights as 1-byte integers instead of 2-byte bf16 — halves weight memory, but compute must dequantize on use |
| **convrot** | the INT8 variant's rotation-based scheme (ConvRot) used by the official Qwen-Image release |
| **A/B experiment** | change exactly ONE variable vs a paired reference; everything else identical |
| **dequantize-on-use** | converting int8 → bf16 (or feeding mixed GEMMs) at execution time — why int8 doesn't automatically double throughput |

**Run.**

```bash
# classroom (~3 min GPU each): mechanics + attention-log evidence
.venv-lab/bin/python -m lab run --experiment F1 --smoke --date m4-smoke
.venv-lab/bin/python -m lab run --experiment F2 --smoke --date m4-smoke

# formal (F1 ~13 min, F2 ~15 min GPU): quotable medians
.venv-lab/bin/python -m lab run --experiment F1 --date m4-f1
.venv-lab/bin/python -m lab run --experiment F2 --date m4-f2
.venv-lab/bin/python -m lab run --experiment LF2 --date m4-lf2   # ~4 min
.venv-lab/bin/python -m lab run --experiment LF3 --date m4-lf3   # ~4 min

# quality gate for the quantization (uses YOUR raw images, ~1 min GPU)
.venv-lab/bin/python -m lab quality --experiment F2
```

Models: F1 needs bf16 (already present from M1); F2 needs the int8 DiT
(7.3 GB); LF2/LF3 reuse the turbo-int8 from your first smoke.

**Observe (reference campaign).**

F1 — attention backend (software choice, same hardware):

| Metric | B0 (pytorch) | F1 (ck) |
|---|---|---|
| Warm total | 55.21 s | **45.09 s (1.22×)** |
| Steady step | 1.345 s | 1.077 s (1.25×) |
| Peak VRAM | 37.7 GB | 37.7 GB |

F2 — INT8 weights (precision choice):

| Metric | B0 (bf16) | F2 (int8 convrot) |
|---|---|---|
| Warm total | 55.21 s | 51.09 s (**1.08×**) |
| Weight file | 14.23 GB | **7.26 GB (−49%)** |
| Peak VRAM | 37.7 GB | **31.0 GB (−6.7 GB)** |
| LPIPS fidelity vs B0 | — | **0.005–0.031 (near-lossless)** |

Combinations — measured, never multiplied:

| Config | Warm | vs B0 |
|---|---|---|
| L1 turbo | 12.13 s | 4.55× |
| LF1 turbo+ck | 10.03 s | 5.50× |
| LF2 turbo int8 | **9.02 s** | **6.12×** |
| LF3 turbo int8+ck | **9.02 s** | **6.12×** |

**Explain.** F1 cross-check with M2: attention was 32.4% of busy ≈ 17.2 s
(measured, trace); the DiT saved 10.7 s ⇒ attention now ≈ 6.5 s (**derived
by subtraction** — no trace of the ck path was taken; the estimate is
consistent, not observed). The Amdahl ceiling was 1.46×; measured 1.22×.
**The backend actually in use is proven from server logs** ("Using Comfy
Kitchen attention"), recorded per run — flags alone are not evidence.

F2: INT8 here is a **memory optimization first**. "Theoretical 2× int8
throughput" did not materialize at 1024² — the dequantize-on-use path
keeps most GEMMs bf16-bound. This gap between hardware peak and measured
latency is the module's central lesson. On the 24 GB Windows card, this
same memory saving was the *only* reason 2048² fit at all.

The combination table is the course's flagship: the "product of parts"
(4.55 × 1.08 × 1.25 ≈ 6.1×) happens to land near the measured 6.12× — but
the *composition* is different: **ck adds exactly nothing on the int8
turbo pipeline** (LF2 = LF3 to the millisecond). After int8, the
attention path already runs the fast kernels; after the step cut, the
remaining attention time is too small to matter. Levers interact —
re-measure every combination, and re-profile: the bottleneck migrates.

The rejected experiment: torch.compile was **NOT_RUN — rejected with
evidence**: elementwise+norm is 6.0% of busy and GPU is 97.8% busy, so the
ceiling is < 0.4 s end-to-end, against real compile cost +
shape-specialization risk. Saying "no" with a number is an optimization
decision too.

**Think (answer keys in `<details>`).**

1. What fraction of F1's DiT saving is explained by the M2 attention
   share? What's the residual?
   <details><summary>answer</summary>M2 attention ≈ 17.2 s of busy;
   F1 saved 10.7 s ⇒ ~62% "explained", residual ~6.5 s plausibly the
   ck-vs-flash gap on this stack plus fusion side-effects — i.e. the
   profile predicted the *scale* of the opportunity, not the exact
   outcome. (And 17.2→6.5 remains a derived estimate — see M2.)</details>
2. Your manager wants INT8 "because it's 2× on paper". Write the reply
   (150 words) using F2 data.
   <details><summary>answer</summary>Key points: measured 1.08× at
   1024² (dequantize-on-use keeps GEMMs bf16-bound); the REAL win is
   −6.7 GB VRAM / −49% weights (enables smaller cards / bigger
   resolutions); fidelity is near-lossless (LPIPS 0.005–0.031 vs 0.0000
   noise floor); paper peak ≠ deployed latency — only the paired A/B
   counts.</details>
3. Run LF2 and LF3 and verify they tie. Explain why in terms of where
   time goes after turbo+int8.
   <details><summary>answer</summary>After int8, attention already runs
   through the fast path; after cutting steps 5×, the remaining attention
   time (~0.3 s scale) is too small a fraction of 9 s for the backend
   choice to register. Evidence: identical warm medians 9.023 vs 9.022 s
   with n=9 each.</details>
4. Why does the lab refuse to benchmark on a server whose attention
   backend it cannot verify from logs?
   <details><summary>answer</summary>Because the A/B would be
   mislabeled: the record would claim "ck vs pytorch" while both arms ran
   the same path. `attention_evidence` (scraped log lines, per record)
   is the difference between measuring and storytelling.</details>

**Acceptance.**
- [ ] F1/F2 smoke or formal records in `runs/`, with `attention_evidence`
      quoted from YOUR F1 record
- [ ] The three A/B tables rebuilt from your records (or the reference
      records, clearly labeled as such)
- [ ] The INT8 essay + the LF2=LF3 explanation written in your notebook
- [ ] The quality grid for F2 opened
      (`runs/quality/F2/quality_grid.png` on your host, or the shipped
      `evidence/quality/F2/`)
