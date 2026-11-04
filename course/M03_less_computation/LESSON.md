# M3 — Less Computation

**Goal:** the biggest lever in the whole course — do less work — measured
honestly, including what it does to output identity.

## Lesson

### The experiment

Swap the 40-step base DiT for the **Turbo distilled 8-step** checkpoint
(same TE, VAE, prompts, seeds, attention, server lifecycle):

| Metric | B0 | L1 | Δ |
|---|---|---|---|
| Warm total | 55.21 s | **12.13 s** | **4.55×** |
| DiT denoise | 53.88 s | 10.78 s | 5.00× |
| Steady step time | 1.345 s | 1.337 s | ≈ 1.0× |
| Fresh-prompt total | 56.11 s | 13.03 s | 4.31× |
| Cold start | 74.2 s | 48.4 s | 1.53× (load-bound) |

**Attribution is clean because per-step cost is unchanged**: the entire gain
is the 32 removed steps. Nothing about *execution* got faster.

### Why isn't it 5×? (the exam question)

DiT was 97.6% of the total; the other ~1.4 s (VAE + TE + fixed costs) is now
a hard floor:

```
ceiling = 55.2 / (55.2·0.976·(8/40) + 1.4) ≈ 4.6×   measured: 4.55×
```

Amdahl's law, live. Also note the fresh-prompt speedup (4.31×) is *lower*
than warm (4.55×) — the 0.5 s TE is a bigger fraction of a 12 s total.

### Two schedules, one compute budget

The official Comfy-Org Turbo template samples with a ManualSigmas schedule
(`1.0, 0.978453, …, 0.414568, 0`) instead of `simple`. L1O measured it:
**12.07 s vs 12.13 s** — identical (same FLOPs), but the *images* differ
(LPIPS distance 0.23–0.34 vs 0.19–0.30 against the same baseline). Same
step count, different numerics: schedule choice is a **quality** decision,
not a speed decision. This is why the lab keeps them as two labeled
experiments instead of silently switching.

### The honesty clause

Turbo is **not** lossless and never claimed to be: outputs legitimately
differ (distance semantics). Whether 8-step images are *acceptable* is a
human judgment on the grids — [M6](../M06_combined_quality/LESSON.md).

## Exercise

1. `python -m lab run --experiment L1 --date <tag>` then `--experiment L1O`.
2. Verify from your records: per-step cost unchanged; DiT total ÷ 5.
3. Compute the 8-step Amdahl ceiling *before* looking at your measured
   number. Compare.
4. Your colleague says "we got 5× because steps went 40→8". Write the
   two-sentence correction.
5. (CPU-only) Read `lab/workflows.py`: what changes between the simple and
   manual-sigmas workflows? Why can't KSampler express the official schedule?

**Acceptance:** B0-vs-L1 table from your own records + the Amdahl derivation.
