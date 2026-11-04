# M1 — Measure First: The Frozen Baseline

**Goal:** establish the reference every later claim is measured against —
and learn how easy it is to measure the wrong thing.

## Lesson

### What "baseline" means here

B0 = the **unoptimized but fully functional** configuration:
BF16 weights, 40 steps, euler/simple, CFG 1, the server's normal attention
path. We do **not** force a slow fallback to exaggerate later gains — the
Windows parent project had to force math-attention as a *bug workaround*
(their AOTriton flash path was broken); on our Linux stack the normal path
works, so the baseline is the normal path. The server log proves which one
ran ("Using pytorch attention") — trust logs, not flags.

### The protocol (why it is shaped like this)

```
cold (seed+1M) -> warmup (seed+2M) -> 3x measured (frozen seeds) -> fresh-prompt arm
```

- **cold** pays model loading once, so it never pollutes steady-state numbers;
- **warmup** uses a *different* seed than cold — otherwise the text-encoder
  node cache turns your warmup into a no-op (a real bug this project hit);
- **measured** runs cycle the frozen seeds [42, 1337, 20261030];
- the **fresh-prompt arm** (" ." appended) busts the TE cache so the real
  text-encode cost becomes visible (warm runs have TE = cache hit = 0 s).

### The measured story (W7900, 2026-11-03)

| Metric | Value |
|---|---|
| Warm total (median, n=9) | **55.21 s** (two independent sessions: 55.24 / 55.11) |
| Steady step time | 1.345 s (spread 1.336–1.367) |
| Fresh-prompt total | 56.10 s (TE adds 0.31–0.60 s) |
| Full cold start | 74.2 s (server boot 7.0 s separate) |
| Peak VRAM (device, sampled) | 37.7 GB |

Two full sessions were run — not because one was doubted, but because
**reproducibility is a measured property, not an assumption**.

### Discussion questions

1. Why is the warm median not "the user experience"? (What is?)
2. The two sessions differ by 0.13 s. Is that noise? How would you decide?
3. Why do we record the *attention backend actually used* instead of the
   flag we passed?

## Exercise

1. `python -m lab baseline --date <your-tag>` (≈16 min on a W7900-class GPU).
2. Open one `measure` record JSON; identify: `runs[0].total_s`,
   `steady_step_median_s` (why is step[0] excluded?), `attention_evidence`,
   `vram_peak_mb` vs `vram_device_peak_mb`, `image_sha256`.
3. Compute your warm median and spread; compare to the case study.
4. Write down: what would break comparability if you "improved" the runner
   mid-session? (e.g., changing polling intervals, prompts, server flags)

**Acceptance:** your SUMMARY row for B0 + a paragraph on which latency
definition a real user would experience.
