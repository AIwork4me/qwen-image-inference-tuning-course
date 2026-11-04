# M2 — Profiling: Where the Time Actually Goes

**Goal:** replace opinions with two layers of evidence — pipeline timing and
GPU kernel traces — then use Amdahl's law to rank opportunities *before*
writing any optimization.

## Lesson

### Layer A: pipeline timing (cheap, always on)

The websocket monitor timestamps every node: text-encode / KSampler / VAE
decode, plus per-step progress events. From the B0 records:

| Stage | Time | Share |
|---|---|---|
| DiT denoise (40 steps) | 53.88 s | **97.6%** |
| VAE decode | 0.90 s | 1.6% |
| Text encoder (fresh) | 0.56 s | ~1% |
| Other (queue/save/client) | 0.43 s | 0.8% |

Lesson one: **the DiT is everything.** Optimizing VAE decode from 0.9 s to
0.0 s would improve end-to-end by 1.6%.

### Layer B: kernel tracing (rocprofv3 on the real server process)

```bash
rocprofv3 --runtime-trace --output-format csv -P 230:90:1 -- python lab/comfyui_launch.py ...
```

A bounded 90-second window (never a full-session trace), aligned by a driver
client so it captures exactly one warm generation — verifiably: the trace
contains exactly 40.0 steps = 95,600 kernel dispatches.

| Kernel class | Dispatches | GPU time | Share of busy |
|---|---|---|---|
| GEMM (`Cijk_Alik_Bljk…` hipBLASLt) | 8,021 | 32.54 s | **61.3%** |
| Attention (`attn_fwd` flash) | 1,248 | 17.23 s | **32.4%** |
| Elementwise/activation | 83,067 | 2.49 s | 4.7% |
| Norm (adaln) | 2,573 | 0.71 s | 1.3% |
| **GPU busy** | — | 53.11 / 54.31 s | **97.8%** |

Plus: 197 memcpy API calls (~36 µs each — per-step sigmas), **1 sync**.
The GPU is compute-bound; there is no scheduling/transfer rescue.

Two tooling lessons baked into the lab code (both cost us a failed run):
- **SIGINT the rocprofv3 wrapper**, or you get unconverted `.dat` buffers;
- the ~0.5 s TE phase fell outside the window — its cost is measured at
  Layer A instead. Know what your profiler did *not* capture.

### Amdahl as a planning tool (not a result)

With DiT = 97.6% of e2e and attention = 32.4% of busy:

- attention → 0 instantly: e2e ≤ 55.2 × (1 − 0.324×0.976) ≈ **1.46×**
- steps 40 → 8 (equal step cost): ≈ **4.6×** ceiling
- GEMM 2× faster: ≈ 1.43×

These bounds ranked the experiments (turbo ≫ int8 > attention) — and the
later measurements confirmed them within ~5% (4.55× measured vs 4.6× bound).
A bound is a *prediction*; only the measurement is a *result*.

## Exercise

1. `python -m lab profile --date <your-tag>` (two bounded sessions, ~12 min).
2. From `rocprof-dit/analysis.json`: reproduce the by-class table; list the
   top-3 kernels by total time with counts and means.
3. Compute the Amdahl ceiling for "elementwise+norm → 0". Why did the case
   study reject torch.compile on this basis? (see P4 §F4)
4. Discussion: your trace shows 97.8% GPU busy. A colleague proposes
   "overlap host-side PNG saving with the next generation". Estimate the
   ceiling of that idea from Layer-A numbers alone.

**Acceptance:** your by-class table + one Amdahl calculation shown step by step.
