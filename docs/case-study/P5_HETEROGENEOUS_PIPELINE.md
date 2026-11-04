# P5 — Parallel & Heterogeneous Computing (Linux W7900)

**Date:** 2026-11-03 · **Runs:** `p5-H1-r2-H1-cpu-te`, `p5-H2-H2-pipeline`
Single W7900. This phase is explicitly **not** multi-GPU tensor parallelism.

## H1 — CPU text-encoding feasibility

Qwen3-VL 8B BF16 TE via ComfyUI's own loader (`comfy.sd.load_clip`,
`CLIPType.QWEN_IMAGE`, `load_device=cpu`) — the identical graph the server runs.
Measured on the EPYC 9334 host (128 threads visible, 55 GiB cgroup).

| Metric | GPU | CPU | Δ |
|---|---|---|---|
| Model load (cold) | 21.5 s | 17.9 s | — |
| Encode p1-zh-scene (52 tok) | 1.17 s | **177.8 s** | 152× slower |
| Encode p2-en-scene (37 tok) | 0.35 s | **136.8 s** | 391× slower |
| Encode p3-poster-text (64 tok) | 0.61 s | **207.1 s** | 340× slower |
| VRAM allocated by TE | 16.8 GB | **0** | −16.8 GB |

**Conditioning parity (correctness):** cosine similarity GPU vs CPU =
1.000001–1.000003 on all 3 prompts; max |Δ| ≤ 0.0074, mean |Δ| ≈ 4×10⁻⁵
(bf16-path rounding). The CPU TE is numerically equivalent.

**Server-integrated variant** (`CLIPLoader device=cpu` in the workflow):
full turbo generation 261.4 s vs 47.1 s with GPU TE — the ~214 s CPU encode
dominates. OpenVINO on AMD CPU was not attempted: the gap to close is 2–3
orders of magnitude; no evidence suggests an OpenVINO EPYC build closes it,
and graph support for Qwen3-VL is unvalidated (per the "do not assume"
discipline this is recorded as not attempted, not as impossible).

**Verdict:** CPU TE = correctness PASS, VRAM win (−16.8 GB), **latency
catastrophe**. Viable only for memory-starved batch scenarios, never for
interactive use on this host.

## H2 — Multi-request pipeline (4 requests, turbo 8-step)

Arm A (sequential): server-side GPU TE then generation, one request at a time.
Arm B (pipelined): CPU TE worker process encodes request *i+1* while the GPU
denoises request *i*; conditioning injected via the lab-owned
`LinuxLabCondLoader` custom node (`ComfyUI/custom_nodes/tuning_lab/ (installed by `python -m lab run --experiment H2`)`).

| Metric | A sequential | B pipelined |
|---|---|---|
| 4-image completion | **51.15 s** | 737.13 s |
| Images / minute | **4.69** | 0.33 |
| Per-request wall | 12.1–13.0 s | 12.1 (fill) + 3×19.1 s gens |
| CPU TE per prompt (overlapped) | — | 160–221 s |
| **Proven CPU–GPU overlap** | — | **50.4 s** (every gen fully inside a TE window) |
| GPU gen slowdown under CPU load | — | 12–13 s → 19.1 s (≈1.5× host contention) |

Overlap is **proven by timestamps** (generation windows ⊂ CPU-TE worker
windows; `H2_RESULTS.json` carries per-request intervals), not inferred.

**Verdict — a valid negative result:** with this host's CPU, request-level
CPU-TE pipelining is **14× worse** in throughput. Two independent causes,
both measured: (1) CPU TE costs 160–221 s vs the GPU's 0.3–0.6 s; (2) the
128-thread CPU matmuls starve the server's kernel-launch threads, slowing the
GPU itself by ~1.5×. Single-image latency is untouched by pipelining either
way — throughput and latency are different quantities (lecture slide 5).

## What WOULD help on one GPU (engineering conclusions, not claims)

- Keep TE on the GPU and accept its 0.3–0.6 s (it is ~5% of even a turbo gen).
- Multi-request throughput on one W7900 is best served by *sequential* fast
  generations (BEST ≈ 6.6 img/min incl. fresh TE) — batching inside one
  workflow (batch_size > 1) is the untested next candidate (see OPEN_ISSUES).
- H3 (concurrent GPU requests): not attempted — a single ComfyUI server
  serializes its queue, VRAM headroom after int8 (~31 GB peak of 48 GB) is
  real but the server model is one-prompt-at-a-time; marked NOT_RUN with
  this reason rather than guessed.

## Reproduce

```bash
python -m lab run --experiment H1   # direct probe + integrated
python -m lab run --experiment H2   # sequential vs pipelined
```
