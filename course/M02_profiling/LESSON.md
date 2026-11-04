# M2 — Profiling: Where the Time Actually Goes

**Problem.** You have a baseline (55 s) and a budget of GPU hours, not
weeks. Which optimization do you build first? Guessing wastes credits;
profiling ranks the opportunities *before* you spend anything — and tells
you what kind of problem you even have (compute-bound? memory-bound?
scheduling-bound?).

**Concepts (beginner glossary).**

| Term | Plain meaning |
|---|---|
| **profiling** | recording *where* time goes — which pipeline stage, which GPU kernel |
| **DiT (diffusion transformer)** | the denoising network doing the 40 steps — the actual image "painter" |
| **kernel** | one unit of work launched on the GPU (a matmul, an attention pass) |
| **GEMM** | GEneral Matrix-Multiply — the arithmetic heart of transformers |
| **attention** | the "look at everything, weight by relevance" operator; cost grows with sequence length |
| **GPU-busy %** | fraction of wall time the GPU was executing kernels (vs idle/waiting) |
| **Amdahl's law** | if a part is fraction *f* of the time and you speed it up by *s*, total speedup = 1 / ((1−f) + f/s) — the fixed part caps you |
| **measured vs derived** | a number an instrument reported vs a number computed from other numbers — always label which |

**Run.**

```bash
# classroom (~2 min GPU): Layer A only — pipeline timing from your B0 records
.venv-lab/bin/python -m lab profile --layers pipeline --date m2-pipe

# full session (~12 min GPU): adds Layer B — bounded rocprofv3 kernel traces
.venv-lab/bin/python -m lab profile --date m2-full
```

GPU-credit note: Layer B launches the server *under the profiler* twice
(DiT window + TE window, 90 s collection each). Traced wall times are for
composition analysis only — interception adds ~1% overhead, so a traced
run is NEVER a performance claim.

**Observe (reference campaign, 2026-11-03; your numbers go in
`runs/<tag>-profile/`).**

Layer A — pipeline timing (from the B0 records' websocket node events):

| Stage | Time | Share | Method |
|---|---|---|---|
| DiT denoise (40 steps) | 53.88 s | **97.6%** | measured (node timing) |
| VAE decode | 0.90 s | 1.6% | measured |
| Text encoder (fresh) | 0.56 s | ~1% | measured (fresh-prompt arm) |
| Other (queue/save/client) | 0.43 s | 0.8% | **derived** (subtraction) |

Layer B — kernel trace (rocprofv3 on the real server process, bounded
90 s window aligned to one warm generation; the trace contains exactly
40.0 steps = 95,600 kernel dispatches):

| Kernel class | Dispatches | GPU time | Share of busy |
|---|---|---|---|
| GEMM (`Cijk_Alik_Bljk…` hipBLASLt) | 8,021 | 32.54 s | **61.3%** |
| Attention (`attn_fwd` flash) | 1,248 | 17.23 s | **32.4%** |
| Elementwise/activation | 83,067 | 2.49 s | 4.7% |
| Norm (adaln) | 2,573 | 0.71 s | 1.3% |
| **GPU busy** | — | 53.11 / 54.31 s | **97.8%** |

Plus: 197 memcpy API calls (~36 µs each — per-step sigmas), **1 sync**.

Two tooling lessons baked into the lab code (both cost a failed run):
- **SIGINT the rocprofv3 wrapper**, or you get unconverted `.dat` buffers
  (reproduced twice before the fix);
- the ~0.5 s TE phase fell outside the first trace window — its cost is
  measured at Layer A instead. **Know what your profiler did not capture.**

**Explain.** Lesson one: the DiT is everything. Optimizing VAE decode from
0.9 s to 0.0 s improves end-to-end by 1.6%.

What 97.8% GPU-busy does — and does not — establish. A high busy
percentage rules out *idle time*: there are no large scheduling gaps, no
"GPU waiting for the host" rescue. It does **not** by itself establish
that the workload is *compute-bound*: a GPU can be 100% busy executing
memory-bandwidth-stalled kernels. Distinguishing compute-bound from
memory-bound needs another instrument (e.g. FLOP utilization vs hardware
peak, or roofline analysis) — which we did not run. What we CAN say from
the two layers together: the time is concentrated in GEMM + attention
kernels (93.7% of busy time) with almost no copies or syncs, and the
per-step times are stable — so the practical conclusion stands (no
scheduling/transfer rescue exists; the levers are less/faster kernel
work), while the compute-vs-bandwidth question remains open. State
exactly as much as the instrument shows.

Amdahl as a planning tool (not a result — a bound is a *prediction*, only
the measurement is a *result*):

- attention → 0 instantly: e2e ≤ 55.2 × (1 − 0.324×0.976) ≈ **1.46×**
- steps 40 → 8 (equal step cost): ≈ **4.6×** ceiling
- GEMM 2× faster: ≈ 1.43×

These bounds ranked the experiments (turbo ≫ int8 > attention) — and the
later measurements confirmed them within ~5% (4.55× measured vs 4.6×
bound).

Charts generated from the data (never hand-edited):
[`evidence/results/chart_warm_latency.png`](../../evidence/results/chart_warm_latency.png)
·
[`evidence/results/chart_waterfall.png`](../../evidence/results/chart_waterfall.png)
— rebuildable with `python -m lab report --write` (read-only by default;
they must match SUMMARY.json
byte-for-byte in the numbers they display; a disagreeing chart is a bug).

**Think (answer keys in `<details>`).**

1. Compute the Amdahl ceiling for "elementwise+norm → 0". Why did the
   case study reject torch.compile on this basis?
   <details><summary>answer</summary>elementwise+norm ≈ 6.0% of busy ≈
   5.9% of e2e ⇒ ceiling ≈ 1/(1−0.059) ≈ 1.063× ⇒ < 3.2 s on 55 s, before
   compile warmup cost and shape-specialization risk. Numbers at:
   `evidence/results/p2-profile-profile/rocprof-dit/analysis.json`
   (by_class shares).</details>
2. Your trace shows 97.8% GPU busy. A colleague proposes "overlap
   host-side PNG saving with the next generation". Estimate the ceiling
   from Layer A alone.
   <details><summary>answer</summary>"Other" (queue/save/client) is 0.8%
   of e2e ⇒ even perfect overlap buys ≤ 0.43 s on 55 s (~1.008×). Also
   ComfyUI already saves asynchronously — the idea costs more in analysis
   time than it can ever return.</details>
3. The kernel table says attention = 17.23 s. M4's F1 experiment saves
   10.7 s of DiT time by swapping the attention backend. Why is
   "attention cost dropped 17.2 → 6.5 s" labeled a DERIVED estimate, not
   a measurement?
   <details><summary>answer</summary>17.2 s comes from the *trace
   classifier* (measured, on the reference host); 6.5 s is inferred by
   subtraction from a DIFFERENT instrument (A/B node timing). No trace of
   the ck path was taken at that time, so the 6.5 s is an estimate
   consistent with the data, not a direct observation. (This is exactly
   the measured-vs-derived discipline: `pipeline_breakdown.json` now
   carries a per-key `method` map.)</details>

**Acceptance.**
- [ ] Layer A ran on YOUR baseline records (smoke evidence accepted if
      GPU credits are tight: reuse `runs/<tag>-profile/pipeline_breakdown.json`)
- [ ] Your by-class table reproduced from `analysis.json` with top-3
      kernels (name, count, total ms)
- [ ] One Amdahl calculation shown step by step, with the fraction's
      source file cited
- [ ] You can point to which numbers in the tables above are measured
      vs derived
