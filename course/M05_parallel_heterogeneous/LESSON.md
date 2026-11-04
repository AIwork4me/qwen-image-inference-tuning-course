# M5 — Parallel & Heterogeneous Computing (on ONE GPU)

**Goal:** learn what single-GPU parallelism can and cannot do — with two
measured negative results and one proof of genuine overlap.

## Lesson

### The mental model

Diffusion sampling is inherently sequential (step *k+1* needs step *k*'s
output). What CAN move: the text encoder (a separate model) to another
processor, and request-level pipelining (encode request *i+1* while the GPU
denoises request *i*). This is **not** multi-GPU tensor parallelism —
one W7900, always.

### H1: move the 8B text encoder to the CPU?

ComfyUI natively supports `CLIPLoader device=cpu`, and the lab runs the
identical graph via `comfy.sd.load_clip`:

| | GPU | CPU |
|---|---|---|
| Encode (3 prompts) | 0.35 / 0.61 / 1.17 s | **136.8 / 177.8 / 207.1 s** |
| VRAM allocated | 16.8 GB | **0** |
| Conditioning agreement | — | cosine ≥ 0.999999, max\|Δ\| ≤ 0.0074 |

Correctness: **pass** (numerically equivalent conditioning). Latency:
**152–391× slower**. The 8B encoder is GEMM-bound — exactly the workload
GPUs are built for. CPU TE is a VRAM trade (−16.8 GB), never a speed play
on this class of host.

### H2: overlap CPU TE with GPU denoising?

4 requests, turbo 8-step. Arm A: sequential (GPU TE → gen, one at a time).
Arm B: a CPU worker encodes the *next* request while the GPU denoises the
current one (conditioning injected through a small custom node).

| | A sequential | B pipelined |
|---|---|---|
| 4-image time | **51.1 s** | 737.1 s |
| Images/min | **4.69** | 0.33 |
| Proven overlap | — | **50.4 s** (timestamps) |
| GPU gen under CPU load | 12–13 s | **19.1 s** |

Three separate lessons in one table:

1. **The overlap is real** — 50.4 s of concurrent CPU+GPU execution, proven
   by per-request timestamps, not inferred from utilization graphs.
2. **It still loses 14×** — the CPU side of the pipeline costs 160–221 s
   per request; Amdahl applies to pipelines too (the slowest stage gates).
3. **Parallelism has a price** — the 128-thread CPU matmuls starve the
   server's kernel-launch threads; the *GPU* gens slowed 1.5×. Two
   processors sharing one host contend.

**Throughput ≠ latency**: neither design changes a single image's 9–12 s.
"Parallel makes it faster" needs a noun — faster *what*?

### What would actually help (untested = open challenge)

Sequential fast generations (BEST ≈ 6.6 img/min with fresh prompts) is the
measured frontier. The one untested direction: **batching** multiple images
into a single denoise (batch_size > 1) — see M7 challenge #1.

## Exercise

1. `python -m lab run --experiment H1` (~15 min) — read `H1_RESULTS.json`:
   quote the GPU/CPU encode times and the cosine agreement.
2. `python -m lab run --experiment H2` (~15 min) — from `H2_RESULTS.json`,
   compute the break-even CPU-encode time *t\** below which pipelining wins
   (derive it from A's per-request time). How far is this host from *t\**?
3. Discussion: on a 4× faster CPU (or an int8 CPU TE), would B win? What
   else would you have to re-measure before claiming it?
4. Explain to a colleague why "GPU util was 100% during the overlap" is not
   evidence the pipeline helped.

**Acceptance:** the break-even derivation + the H2 numbers that support it.
