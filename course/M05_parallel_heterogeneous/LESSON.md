# M5.0 — PyTorch CPU Text Encoding & Pipelining (on ONE GPU)

**Learning objectives (you will be able to):**
1. Derive the pipelining break-even CPU-encode time t\* from the H2 measurements and state how far the reference host is from it;
2. Prove overlap exists from per-request timestamps while explaining why the pipeline still loses 14× (slowest-stage gating + contention cost);
3. Separate throughput from latency notions and demand a noun in "parallel is faster — faster what?";
4. Argue that a negative result with complete evidence is first-class science, citing exactly where the evidence lives.

中文版：[LESSON.zh-CN.md](LESSON.zh-CN.md)

**Next:** [M5.1 — CPU Text Encoder accelerated by OpenVINO](M5_1_OPENVINO_CPU_TE.md) treats this PyTorch CPU implementation as the baseline. **Same-stack W7900 measurements are now COMPLETED (2026-11-05):** encoder 331–480× faster on positive prompts (328× on the empty negative), warm E2E service latency 1.081× / actual 4-request batch throughput 1.029× better, 17.26 GB full-pipeline peak saved. See [the measured lesson](M5_1_OPENVINO_CPU_TE.md) and [M51_RESULTS](../../docs/M51_RESULTS.md).

**Problem.** Marketing parallelism is easy: "just move work to the CPU and
overlap!" Engineering parallelism is measured. This module takes the two
seductive ideas for a single-GPU image service — offload the text encoder
to the CPU, pipeline CPU work under GPU denoising — and measures both to
the point of proof, one yielding a 14× *loss* with the most instructive
evidence in the course.

**Concepts (beginner glossary).**

| Term | Plain meaning |
|---|---|
| **heterogeneous computing** | using different processor types (GPU + CPU) for different parts of one workload |
| **text encoder (TE)** | the 8B-parameter Qwen3-VL model that turns your prompt into conditioning tensors (~17.5 GB in bf16) |
| **pipelining** | starting stage *i+1*'s work while stage *i* still runs (here: encode request *n+1* on CPU while the GPU denoises request *n*) |
| **overlap** | time during which two processors genuinely execute concurrently — *provable only with timestamps*, not utilization graphs |
| **throughput vs latency** | images per minute vs seconds per image — different metrics, different winners |
| **contention** | two processors sharing one host fight for memory bandwidth and OS threads — parallelism has a price |
| **negative result** | a measurement showing an approach does NOT work — first-class science when the evidence is complete |

**Run (budget warning — read first).**

The H1/H2 experiments are **slow CPU-bound sessions (~15 min each)** and
are **taught from the committed physical evidence by default** — the
committed records below are complete (timings, agreement stats, overlap
timestamps) and re-running them buys no new information on the same
hardware class. They remain available as opt-in research tasks after
budgeting:

```bash
# opt-in research reruns (~15 min GPU each; H1 spends most of it on the CPU)
.venv-lab/bin/python -m lab run --experiment H1
.venv-lab/bin/python -m lab run --experiment H2
```

Committed evidence:
`evidence/results/p5-H1-H1-cpu-te/` + `p5-H1-r2-H1-cpu-te/` (probe with
reproducibility arm) and `evidence/results/p5-H2-H2-pipeline/H2_RESULTS.json`.

**Observe (reference campaign — the shipped evidence).**

H1 — move the 8B text encoder to the CPU? (identical graph via
`comfy.sd.load_clip`):

| | GPU | CPU |
|---|---|---|
| Encode (3 prompts) | 0.35 / 0.61 / 1.17 s | **136.8 / 177.8 / 207.1 s** |
| VRAM allocated | 16.8 GB | **0** |
| Conditioning agreement | — | cosine ≥ 0.999999, max\|Δ\| ≤ 0.0074 |

H2 — overlap CPU TE with GPU denoising? (4 requests, turbo 8-step; arm B
injects CPU-precomputed conditioning through a lab-owned custom node):

| | A sequential | B pipelined |
|---|---|---|
| 4-image time | **51.1 s** | 737.1 s |
| Images/min | **4.69** | 0.33 |
| Proven overlap | — | **50.4 s** (timestamps) |
| GPU gen under CPU load | 12–13 s | **19.1 s** |

**Explain.** H1: correctness **passes** (numerically equivalent
conditioning) while latency **fails** (152–391× slower). The 8B encoder
is GEMM-bound — exactly the workload GPUs are built for. CPU TE is a VRAM
trade (−16.8 GB), never a speed play on this class of host.

H2 packs three separate lessons into one table:

1. **The overlap is real** — 50.4 s of concurrent CPU+GPU execution,
   proven by per-request timestamps, not inferred from utilization
   graphs. (Why timestamps: utilization can be 100% while a pipeline
   still loses — the numbers below are the proof.)
2. **It still loses 14×** — the CPU side of the pipeline costs 160–221 s
   per request; Amdahl applies to pipelines too (the slowest stage gates
   the whole thing).
3. **Parallelism has a price** — the 128-thread CPU matmuls starve the
   server's kernel-launch threads; the *GPU* generations slowed 1.5×
   (12–13 s → 19.1 s). Two processors sharing one host contend.

**Throughput ≠ latency**: neither design changes a single image's 9–12 s.
"Parallel makes it faster" needs a noun — faster *what*?

What would actually help (untested = open challenge): sequential fast
generations (BEST ≈ 6.6 img/min with fresh prompts) is the measured
frontier. The one untested direction: **batching** multiple images into a
single denoise (batch_size > 1) — see the M7 challenges.

**Think (answer keys in `<details>`).**

1. From `H2_RESULTS.json`, derive the break-even CPU-encode time *t\**
   below which pipelining wins. How far is the reference host from *t\**?
   <details><summary>answer</summary>In steady state, arm B's cycle time
   is max(GPU gen, CPU encode) + handoff; A's is TE_gpu + gen ≈ 12.8 s.
   Pipelining wins when max(gen, t_cpu) < 12.8 s ⇒ t\* ≈ 12 s (given
   gen ≈ 12 s unperturbed — and note contention *raises* gen). The
   reference host's CPU encode is 160–221 s: 13–18× too slow, before
   even counting contention.</details>
2. On a 4× faster CPU (or an int8 CPU TE), would B win? What else would
   you have to re-measure before claiming it?
   <details><summary>answer</summary>At t_cpu ≈ 40–55 s, still above
   t\*; at ≈ 8× you approach break-even. But contention must be
   re-measured (thread starvation scales with CPU load), and the
   *quality* path of CPU-produced conditioning must re-pass the
   agreement check. The claim needs a fresh paired A/B — never an
   extrapolation.</details>
3. Explain why "GPU util was 100% during the overlap" is not evidence the
   pipeline helped.
   <details><summary>answer</summary>Utilization says the GPU was busy —
   not that the *total* shrank. In arm B the GPU was busy *slower*
   (gens 19.1 s vs 12–13 s) while the CPU crawled at its own pace:
   100% busy and 14× worse. Throughput is measured in images/minute
   end-to-end, and the timestamps prove where the time went.</details>
4. Which single number in the H1 table would you cite to kill the
   "just offload the TE" proposal — and which caveat must accompany it?
   <details><summary>answer</summary>136.8–207.1 s vs 0.35–1.17 s
   (152–391× slower). Caveat: it *frees 16.8 GB of VRAM* — on a
   memory-constrained card that trade can still be right, at a latency
   cost stated up front.</details>

**Student exercise (tiered).**
- Beginner: write the break-even derivation (cycle formula → t\*) with each number's source
  field cited. Acceptance: complete derivation citing specific paths in `H2_RESULTS.json`.
- Intermediate: restate the three H2 lessons in one sentence each, attaching the exact
  evidence number from the table. Submit: three sentences + three numbers.
- Advanced: substitute M5.1's OpenVINO BF16 CPU encode (0.44 s/request) into arm B's cycle
  model, recompute the 4-request total, compare with M5.1's measured arm C (50.05 s), and
  attribute the difference (fill, contention, handoff). Submit: recomputed table +
  attribution paragraph.

**Research extension (testable hypothesis).** Hypothesis: on a cloud instance with 2× CPU
quota (32 threads), arm B's CPU encode time roughly halves but GPU contention raises
generation time ≥15%, so pipelining still does not break even. Write the prediction
formulas and design a minimal A/B (fixed prompts and seeds) testing both predictions.

**Acceptance.**
- [ ] The break-even derivation written out with the H2 numbers cited
- [ ] The three lessons of the H2 table restated in your own words
- [ ] One paragraph on why a negative result with timestamps is worth
      more than an optimistic guess — referencing where the overlap
      evidence lives (`H2_RESULTS.json` → `B_pipelined.overlap_s_total`)
- [ ] (Opt-in, budgeted) your own H1/H2 rerun records in `runs/`,
      compared against the shipped ones
