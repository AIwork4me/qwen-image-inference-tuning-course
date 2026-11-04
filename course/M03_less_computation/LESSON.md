# M3 — Less Computation

**Problem.** Everything so far kept the workload identical and asked "how
do we run it faster?" The bigger question: *does the workload itself need
to be this big?* The 40-step sampling schedule is a quality choice, not a
law of physics. Distillation-trained "Turbo" models do 8 steps — the
biggest single lever in this course. But a speedup that changes the output
is a *tradeoff*, and tradeoffs get judged, not just timed.

**Concepts (beginner glossary).**

| Term | Plain meaning |
|---|---|
| **diffusion steps** | denoising iterations from pure noise to image — 40 here, 8 for turbo |
| **distillation / Turbo** | a student model trained to reach image quality in few steps that the teacher needed many for |
| **step schedule (sigmas)** | the noise levels visited across steps (`simple` vs the official ManualSigmas list) |
| **per-step cost** | seconds per step — if it is unchanged after a change, your gain was *purely* "fewer steps" |
| **quality tradeoff** | different output that may be acceptable or not — a human decides (M6) |
| **LPIPS distance** | a perceptual difference metric between two images — for different schedules it measures *difference*, not *badness* |

**Run.**

```bash
# classroom (~2 min GPU each): one prompt, one measured run — mechanics only
.venv-lab/bin/python -m lab run --experiment L1  --smoke --date m3-smoke
.venv-lab/bin/python -m lab run --experiment L1O --smoke --date m3-smoke2

# formal (~5 min GPU each): full protocol, quotable medians
.venv-lab/bin/python -m lab run --experiment L1  --date m3-l1
.venv-lab/bin/python -m lab run --experiment L1O --date m3-l1o
```

Model note: L1/L1O need the turbo-bf16 DiT (14.2 GB — `start.sh
--all-models` if you staged only the smoke set).

**Observe (reference campaign).**

| Metric | B0 | L1 | Δ |
|---|---|---|---|
| Warm total | 55.21 s | **12.13 s** | **4.55×** |
| DiT denoise | 53.88 s | 10.78 s | 5.00× |
| Steady step time | 1.345 s | 1.337 s | ≈ 1.0× |
| Fresh-prompt total | 56.11 s | 13.03 s | 4.31× |
| Cold start | 74.2 s | 48.4 s | 1.53× (load-bound) |

**Explain.** Attribution is clean *because per-step cost is unchanged*:
the entire gain is the 32 removed steps; nothing about execution got
faster. This is what separates "less computation" (M3) from "faster
computation" (M4) — check the step time first, always.

Why isn't it 5×? (the exam question). DiT was 97.6% of the total; the
other ~1.4 s (VAE + TE + fixed costs) is now a hard floor:

```
ceiling = 55.2 / (55.2·0.976·(8/40) + 1.4) ≈ 4.6×   measured: 4.55×
```

Amdahl's law, live. Also note the fresh-prompt speedup (4.31×) is *lower*
than warm (4.55×) — the 0.5 s TE is a bigger fraction of a 12 s total.

Two schedules, one compute budget: the official Comfy-Org Turbo template
samples with a ManualSigmas schedule (`1.0, 0.978453, …, 0.414568, 0`)
instead of `simple`. L1O measured it: **12.07 s vs 12.13 s** — identical
(same FLOPs), but the *images* differ (LPIPS distance 0.23–0.34 vs
0.19–0.30 against the same baseline). Same step count, different
numerics: schedule choice is a **quality** decision, not a speed decision.
The lab keeps them as two labeled experiments instead of silently
switching.

The honesty clause: Turbo is **not** lossless and never claimed to be.
Outputs legitimately differ (distance semantics). Whether 8-step images
are *acceptable* is a human judgment on the grids — M6.

Paired images (same prompt, same seed, only the schedule changed):
[`evidence/samples/B0/p1-zh-scene-42-r0.png`](../../evidence/samples/B0/p1-zh-scene-42-r0.png)
vs
[`evidence/samples/BEST/p1-zh-scene-42-r0.png`](../../evidence/samples/BEST/p1-zh-scene-42-r0.png)
— look at both before reading any number.

**Think (answer keys in `<details>`).**

1. Your colleague says "we got 5× because steps went 40→8". Write the
   two-sentence correction.
   <details><summary>answer</summary>"Steps fell 5× but only the DiT
   fraction (~97.6%) scales with steps; the ~1.4 s of VAE/TE/fixed cost
   doesn't. Amdahl caps the end-to-end gain at ≈4.6×, and we measured
   4.55×."</details>
2. Verify from your own L1 records that per-step cost is unchanged. Which
   field proves it, and why is step[0] excluded from that field?
   <details><summary>answer</summary>`steady_step_median_s` (1.337 vs
   1.345 s). Step 0 carries TE + weight-restore overhead on top of its
   compute, so the steady median uses steps 1+ (`lab/core.py:
   steady_step_rate`).</details>
3. L1 and L1O have identical speed but different images. Give one product
   scenario where you would ship L1O over L1 regardless of speed.
   <details><summary>answer</summary>When the official template's schedule
   is the compatibility contract (e.g. matching Comfy-Org reference
   outputs for regression testing), you accept its numerics as the
   standard even at equal speed.</details>
4. (CPU-only, no GPU needed) Read `lab/workflows.py`: what changes
   between the simple and manual-sigmas workflows? Why can't KSampler
   express the official schedule?
   <details><summary>answer</summary>ManualSigmas switches to
   RandomNoise + CFGGuider + KSamplerSelect + SamplerCustomAdvanced,
   passing the explicit sigma list. KSampler only takes a scheduler
   *name* and synthesizes sigmas from step count — it cannot accept an
   arbitrary 9-value list (8 steps + terminal 0).</details>

**Acceptance.**
- [ ] Your L1 (and ideally L1O) records in `runs/` — smoke acceptable for
      mechanics, formal for quotable numbers
- [ ] B0-vs-L1 table built from YOUR records
- [ ] The 8-step Amdahl ceiling computed *before* reading your measured
      number, then compared
- [ ] The two-sentence 5×-myth correction written in your notebook
