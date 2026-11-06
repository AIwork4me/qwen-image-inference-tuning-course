# M1 — Measure First: The Frozen Baseline

**Learning objectives (you will be able to):**
1. Design and run a reproducible baseline protocol and explain the purpose of each of its four phases (cold / warmup / measured / fresh);
2. Compute your own warm median and spread from your records and compare them against the reference campaign;
3. Decompose fresh − warm = 0.90 s precisely (TE 0.56 s + scheduling variance ~0.34 s) and refuse to over-explain it;
4. Argue, with two-session data, that reproducibility is a measured property, not an assumption.

中文版：[LESSON.zh-CN.md](LESSON.zh-CN.md)

**Problem.** "It feels slow" is not an engineering statement. Before any
optimization you need one number that is (a) defined precisely enough to
re-measure, (b) reproducible, and (c) honest about what "fast" means for a
user. That number is the baseline B0 — and most performance mistakes are
made *here*, before any optimization exists.

**Concepts (beginner glossary).**

| Term | Plain meaning |
|---|---|
| **baseline (B0)** | the unoptimized but *fully functional* config: bf16 weights, 40 denoising steps, euler/simple, CFG 1, the server's normal attention path |
| **warm run** | a repeated generation with everything (weights, text-encoder output) already cached — measures steady-state throughput |
| **cold run** | the first-ever generation: pays model loading from disk |
| **fresh-prompt run** | a new prompt text: busts the text-encoder cache so the real TE cost is visible |
| **median & spread** | the middle value of sorted repeats, and how far apart they are — spread tells you whether the median is trustworthy |
| **node cache** | ComfyUI memoizes identical computations; same prompt+seed = instant replay (a measurement trap, see below) |

B0 is deliberately NOT a slow fallback: the parent Windows project had to
force math-attention as a *bug workaround* (their flash path was broken);
on Linux the normal path works, so the baseline is the normal path. The
server log proves which one ran ("Using pytorch attention") — trust logs,
not flags.

**Run.**

```bash
# classroom smoke (bounded, ~3 min GPU): one prompt, one measured run
.venv-lab/bin/python -m lab baseline --smoke --date m1-smoke

# formal session (~16 min GPU on a W7900-class card): the full protocol
.venv-lab/bin/python -m lab baseline --date m1-$(date +%m%d)
```

GPU-credit note: the formal protocol is 16 runs (1 cold + 1 warmup + 3
measured per prompt × 3 prompts + 3 fresh-prompt) ≈ 16 min. The smoke is
fine for learning the mechanics; only the formal session produces a
median you may quote.

**Observe (real output, live Radeon Cloud W7900-class instance,
2026-11-04, full session — the very session this course ships with).**

```
  cold          p1-zh-scene seed=1000042 total=95.583s ks=56.478s te=1.421s vae=1.256s steps=40 vrampeak=17305.8MB dev=17757.5MB
  warmup        p1-zh-scene seed=2000042 total=55.155s ks=53.815s te=0s    vae=0.901s steps=40 vrampeak=15809.2MB dev=17812.0MB
  measure       p1-zh-scene seed=42      total=55.188s ks=53.824s te=0s    vae=0.898s steps=40 vrampeak=15809.2MB dev=17812.0MB
  measure       p2-en-scene seed=42      total=55.098s ks=53.687s te=0s    vae=0.899s steps=40 vrampeak=33139.4MB dev=35105.3MB
  ...
  fresh_prompt  p1-zh-scene seed=42      total=55.709s ks=53.911s te=0.512s ...
```

Three things to notice before any optimization exists: `te=0` on warm runs
(text-encoder node cache hit — hence the fresh-prompt arm); the cold run
is **95.6 s vs the reference host's 74.2 s** — the cloud instance loads
weights from loop-mounted persistent storage, slower than the lab's local
disk, a *platform* property your records capture honestly; and the VRAM
columns differ between prompts (ComfyUI offloaded the 17.5 GB text
encoder after the cold run and re-resident-ed it at p2 — memory
management is a variable, which is why the summary reports per-record
peaks and their max).

The protocol shape, and why:

```
cold (seed+1M) -> warmup (seed+2M) -> 3x measured (frozen seeds) -> fresh-prompt arm
```

- **cold** pays model loading once, so it never pollutes steady-state numbers;
- **warmup** uses a *different* seed than cold — otherwise the TE node
  cache turns your warmup into a no-op (a real bug this project hit);
- **measured** runs cycle the frozen seeds [42, 1337, 20261030];
- the **fresh-prompt arm** (" ." appended) busts the TE cache so the real
  text-encode cost becomes visible.

**Explain.** The reference campaign's measured story (2026-11-03):

| Metric | Value |
|---|---|
| Warm total (median, 18 runs across two independent sessions) | **55.21 s** (sessions: 55.24 / 55.21) |
| Steady step time | 1.345 s (spread 1.336–1.367) |
| Fresh-prompt total | 56.10 s |
| — of which text encode (measured on the fresh arm) | 0.56 s median (per-run range 0.31–0.60 s) |
| — of which residual (queue/scheduling variance, not TE) | ~0.34 s |
| Full cold start | 74.2 s (server boot 7.0 s separate) |
| Peak VRAM (device, sampled) | 37.7 GB |

Read the fresh decomposition precisely: **fresh − warm = 0.90 s**, of which
the *measured* TE contribution is 0.56 s (median) — the remaining ~0.34 s
is run-to-run scheduling variance, not text encoding. Saying "fresh adds
the TE cost" alone under-explains the delta by a third.

Two full sessions were run — not because one was doubted, but because
**reproducibility is a measured property, not an assumption**. Your
instance should land near these numbers on a W7900-class GPU (the live
cloud run above: 55.17–55.24 s); a different Radeon will legitimately
differ — your records capture your actual GPU, which is what makes them
honest.

Where your records land: `runs/<tag>-B0/*.json` (records) and
`runs/raw/<tag>-B0/*.png` (images, stay on your host — their SHA256 is in
the records). Aggregate YOUR numbers:

```bash
.venv-lab/bin/python -m lab report --scope student --write
```

which writes `runs/SUMMARY.md` — the frozen campaign's SUMMARY under
`evidence/results/` is never touched by this.

**Think (answer keys in `<details>`).**

1. Why is the warm median not "the user experience"? What is?
   <details><summary>answer</summary>A user usually pays cold-ish costs
   (server warm but prompt new): the fresh-prompt total is the closest
   single number, and cold+boot matters for first-request SLAs. Warm is
   the right number only for steady-state throughput of repeated
   work.</details>
2. The two reference sessions differ by ~0.03–0.13 s. Is that noise? How
   would you decide?
   <details><summary>answer</summary>Compare with the *within-session*
   spread (1.336–1.367 s per step ⇒ ~0.03 s on the total is within run
   noise). A session gap of the same magnitude as within-session spread =
   consistent measurement. Also: identical seeds produce bit-identical
   images (noise floor 0.0000), so timing variance is scheduler/host
   noise, not model noise.</details>
3. Why do we record the *attention backend actually used* instead of the
   flag we passed?
   <details><summary>answer</summary>Flags can silently no-op (backend
   unavailable → fallback). The scraped log line is the execution truth;
   the flag is only an intention. Evidence > intent.</details>
4. What breaks comparability if you "improve" the runner mid-session?
   <details><summary>answer</summary>Everything downstream: polling
   intervals change measured overheads, new prompts/seeds change the
   workload, server flags change the thing measured. Frozen protocol =
   comparable A/Bs; changes go in a NEW experiment with its own
   records.</details>

**Student exercise (tiered).**
- Beginner: run the smoke session, locate one of your records under `runs/`, and point at
  `total_s`, `steady_step_median_s`, `attention_evidence`, `vram_peak_mb`. Acceptance: one
  sentence per field (and why step[0] is excluded — it carries TE + weight-restore costs).
- Intermediate: aggregate your numbers with `report --scope student --write` and write a
  paragraph comparing your median/spread vs the reference 55.21 s / 1.336–1.367 s; decide
  whether your host belongs to the same measurement population. Submit: the paragraph plus
  the SUMMARY excerpt.
- Advanced: using only your own records, reproduce the three-way decomposition of
  fresh − warm = 0.90 s (TE, scheduling residual, uncertainty) and state which part cannot
  be separated on your host. Submit: the decomposition table + method note.

**Research extension (testable hypothesis).** Hypothesis: on this stack, lowering the
measurement poll interval to 50 ms would push measurement bias on warm totals below 0.05 s,
making <1% real improvements detectable. Derive the quantization-error bound of the current
interval first, then design an n≥9 experiment, and predict the new noise floor.

**Acceptance.**
- [ ] Your B0 smoke or formal session completed with schema-valid records
      in `runs/`
- [ ] You can point to `runs[0].total_s`, `steady_step_median_s`
      (why is step[0] excluded? — it carries TE + weight-restore costs),
      `attention_evidence`, `vram_peak_mb` vs `vram_device_peak_mb`, and
      `image_sha256` in one of YOUR records
- [ ] Your warm median + spread computed and compared to the case study
- [ ] One paragraph: which latency definition would a real user
      experience, and which number in your records is closest to it
