# The Optimization Research Methodology

**How this course thinks: Measure → Model → Hypothesize → Optimize → Validate → Explain → Reproduce → Contribute.**

*This is the methodology chapter that binds every module (M0–M7) together. Read
it once before M1 and again after M6. Every claim referenced here is registered
in [`data/verified_claims.json`](data/verified_claims.json) with its evidence
path and scope.*

中文版：[METHODOLOGY.zh-CN.md](METHODOLOGY.zh-CN.md)

---

## 1. The one-sentence philosophy

> A performance number without a run record is not a result — and a speedup
> without a quality verdict is not a delivery.

Optimization is not a hunt for impressive numbers. It is an engineering
discipline of **bottleneck identification, controlled change, and honest
measurement**, where negative results are as valuable as positive ones —
sometimes more, because they are rarer and better documented.

---

## 2. The optimization research loop (ask this every module)

For every optimization you study, propose, or review, answer all ten
questions in order — before looking at the next section's measured result:

| # | Question | Where the course answers it |
|---|---|---|
| 1 | **What real problem are we solving?** (For whom is latency/throughput/memory the binding constraint?) | M1 baseline motivation; M5 deployment story |
| 2 | **What operation dominates cost?** (Not what is fashionable — what is on the critical path?) | M2: DiT 97.6% of pipeline; GEMM 61.3% / attention 32.4% of GPU-busy |
| 3 | **Which measurements support that diagnosis?** | M2 pipeline + kernel traces; M5 request timestamps |
| 4 | **What change do we propose — and which of the three strategies is it?** | §3 below |
| 5 | **What speedup do we PREDICT before implementing?** (Amdahl bound; write it down first) | M3: predicted ≤4.63×; M5.1: predicted ≤1.07× |
| 6 | **Which variables are held constant?** (One variable at a time; paired A/B) | Every module's controls table |
| 7 | **How exactly do we measure the result?** (Warm/cold, median/mean, n, workload) | M1 protocol; verdict vocabulary |
| 8 | **Did the target metric actually improve beyond noise?** | Every module's observation table |
| 9 | **What happened to quality, memory, and resource cost?** | M6 quality gate; M4 VRAM; M5 CPU/RAM |
| 10 | **How would someone else reproduce and verify this?** (Run IDs, seeds, evidence paths, PR) | M0 evidence culture; M7 contribution |

A hypothesis that cannot be falsified by a measurement is not an engineering
hypothesis. Prediction-before-measurement (question 5) is what turns a lab
notebook into science: write the bound, then check the bound.

---

## 3. The three (and only three) ways to make inference faster

Every optimization in this course — and most in the field — is one of:

### ① Less Computation — remove work

Don't accelerate an operation; delete it. Step distillation (40 → 8 denoising
steps) removed 80% of the DiT work: **4.55× warm** (M3). The catch: the
schedule change legitimately changes the output, so the gain is purchased with
a **quality decision** a human must judge — and Amdahl caps it below 5×
because the DiT is only 97.6% of the pipeline, and per-step cost is unchanged
(1.345 → 1.337 s: proof the gain is purely removed work, not faster kernels).

### ② Faster Computation — same work, faster kernels

Attention backend swap (ck): **1.22×** (F1). INT8 weight-only quantization:
**1.08×** at 1024² but **−6.7 GB VRAM** — a memory-first win (F2). The
discipline: hardware-usage percentages are not latencies; only paired
end-to-end measurement counts, and **independent gains never multiply**
(4.55 × 1.08 × 1.25 "predicts" 6.14×; measured 6.12× — and ck adds *nothing*
on the int8-turbo path: LF2 vs LF3 = 9.023 vs 9.022 s, fully overlapping
distributions).

### ③ Parallel & Heterogeneous Computing — overlap or relocate work

When one accelerator is saturated, move work to a different engine and overlap
it. This course measured the seductive version (CPU text encoding under GPU
denoising) and **lost 14×** (M5.0) — then fixed the implementation with
OpenVINO (M5.1) and recovered to a **warm service gain of 1.081×** with a
**17.26 GB VRAM saving**, while actual batch throughput improved only
**1.029×** because orchestration overhead — not encoding — became the new
bottleneck. Heterogeneous computing rewards you exactly to the degree that
the moved work was on your critical path, and charges you for every handoff
you add.

**The strategies compose only through measurement.** M6's BEST (6.12×) is a
measured combination of ①+②, never a product of ratios.

---

## 4. Core performance concepts (each with the course's own numbers)

### 4.1 Computational complexity vs measured cost

Complexity (e.g. attention O(n²·d)) tells you how cost *scales*, not what you
will measure. The course's kernel trace (M2) showed GEMM at 61.3% of busy time
— the theory-directed search found the hotspot, but only the trace quantified
it. Rule: complexity argues, measurement decides.

### 4.2 Memory capacity vs memory bandwidth

Capacity is GB; bandwidth is GB/s. F2's INT8 halves weight bytes → capacity
problem solved (−6.7 GB), but at 1024² the DiT stays compute-bound, so latency
barely moves (1.08×). When a workload becomes bandwidth-bound (large
resolutions, long sequences), the same change would pay much more. Know which
regime you are in before predicting.

### 4.3 Compute-bound vs memory-bound

One number: GPU-busy was **97.8%** during B0 (M2) — the W7900 was starved for
work 2.2% of the time at most, so host-side micro-optimizations had a ceiling
of ~1.02×. That is why torch.compile was rejected (NOT_RUN) — its ceiling
(<6% of DiT) could not repay its engineering cost *for this workload on this
stack*.

### 4.4 Occupancy and kernel launch costs

Many small kernels → launch/sync overhead can dominate. M5.0's CPU encoder
starved the GPU's launch threads: under CPU-TE load, GPU generations slowed
~1.5× (12–13 s → 19.1 s) — a *contention* effect invisible in any
per-component benchmark.

### 4.5 Amdahl's law — the course's most-used formula

$$S(n) = \frac{1}{(1-p) + \frac{p}{n}}$$

Speedup is capped by the fraction p of time the improvement touches, no matter
how large the improvement n. Course instances:

- Steps 40→8 (n=5) with p=0.976 (DiT share): cap = 1/(0.024+0.1952) = **4.63×**; measured 4.55×.
- Encoder 331–480× faster (n≈331+) with p≈0.04 (encoder share): cap ≈ **1.04–1.07×**; measured warm service 1.081× (within noise of the bound once handoffs are counted).

Amdahl is why question 5 (predict first) is always answerable.

### 4.6 Latency vs throughput — and why they diverge

Latency is what one requester waits; throughput is images per unit time
across a batch. M5.1's sharpest lesson: warm per-request service latency
improved **1.081×** (13.031 → 12.060 s) while actual four-request batch
throughput improved only **1.029×** (4.42 → 4.55 img/min). The difference is
inter-request orchestration (~0.54 s/request extra in the precomputed arm).
Any "images/min" from summing service times is an upper bound and must be
labeled naive.

### 4.7 Cold start vs warm start

Cold = first request after idle (model load, caches empty): BEST cold is
32.4 s vs warm 9.02 s. Warm = models resident, caches hot. Never average the
populations; never compare a cold arm against a warm arm. M5.1's cold-inclusive
1.33× is mostly *avoiding a 16.8 GB GPU CLIP reload* — a session-start effect,
not a steady-state gain.

### 4.8 Fresh-prompt latency

A "warm" server still pays a fresh prompt's text encode: B0 fresh total is
+0.90 s over warm (TE 0.56 s + ~0.34 s scheduling). M5.1's serialized
fresh-prompt C1b estimate is 12.58 s (12.03 precomputed + 0.55 encode).
Fresh-prompt latency is the honest number for a one-shot API user.

### 4.9 Pipeline fill and drain

A pipeline of k stages must fill before it saturates and drains before it
finishes. M5.1's arm C pays a 0.56 s fill (first CPU encode) per batch; arm
B's fill was 176 s (first PyTorch CPU encode). Short batches amortize fill
poorly — a 4-request batch is 4/4 affected; a 100-request batch is 4%.

### 4.10 CPU/GPU overlap

Overlap is real only if proven by intervals: M5.0 proved 50.4 s of overlap
with per-request timestamps — and still lost 14×, because overlap hides CPU
time only when the CPU stage is not the slowest producer. B's CPU stage
(147 s median) was 12× slower than the GPU stage (12.07 s): the pipeline ran
at CPU speed.

### 4.11 Queueing and orchestration overhead

Every handoff costs: HTTP submission, queue wait, workflow parse, file write,
completion polling. M5.1 measured ~0.54 s/request of extra orchestration in
the precomputed arm — small next to 12 s generations, but it consumed
two-thirds of the service gain at batch level. Orchestration is a real,
measurable component of system performance (and the subject of the course-v2
overhead study).

### 4.12 Kernel speedup vs end-to-end speedup

A kernel 1.25× faster (ck on DiT) yields 1.22× end-to-end (VAE + TE don't
accelerate). An encoder 331–480× faster yields 1.081× service. The gap is
always Amdahl + handoffs. Report end-to-end or report nothing.

### 4.13 Quantization: storage vs arithmetic

INT8 here means INT8 *stored weights* with mixed execution — not INT8
arithmetic everywhere. F2 (INT8 DiT) is fidelity-near-lossless (LPIPS
0.005–0.031 vs 0.0000 noise floor). M5.1's C2 (INT8 encoder) fails the
empty-negative conditioning gate at 28.25% relL2: quantizing *different
components* has wildly different blast radii. Never say "INT8" without saying
*what* is quantized and *which* gate judges it.

### 4.14 Fidelity vs perceptual similarity

Fidelity = distance from the same arm's own reference (INT8 vs BF16, same
schedule): low LPIPS means *quantization is faithful*. Similarity = distance
between *different* output distributions (turbo vs 40-step): LPIPS 0.19–0.30
means *legitimately different images*, neither better nor worse by
construction. Confusing the two semantics is the most common analysis error
the course sees.

### 4.15 Experimental noise and confidence

Report n, median, and spread; never a single run. The course's noise floor:
9/9 identical-seed LPIPS 0.0000 (deterministic within a session), yet B0
sessions differ 55.24 vs 55.21 s (0.06%) — so sub-1% "improvements" on this
stack are noise. When ranges overlap (LF2 9.023 vs LF3 9.022), the honest
conclusion is "no measurable difference," not "LF3 wins."

### 4.16 Break-even and bottleneck migration

For every optimization there is a workload where it stops paying: the 0.53 s
CPU encoder is 50% of the cycle only when denoise is ~0.53 s (it is ~12 s).
And every successful optimization *migrates* the bottleneck: M5.1 moved it
from the encoder to orchestration + GPU denoise; the next win must attack
*those*, or it wins nothing.

---

## 5. Negative results are first-class science

The course's most-taught chain of results is a *failure → diagnosis → partial
recovery → system-level lesson*:

1. **M5.0 (negative):** PyTorch CPU TE offload frees 16.8 GB VRAM but costs
   152–391× encoder latency and a 14× throughput loss with proven overlap.
2. **M5.1 (recovery):** a better CPU implementation (OpenVINO, persistent
   worker, BF16-execution IR) makes CPU encoding viable: 331–480× faster on
   positive prompts, conditioning PASS, 17.26 GB VRAM saved.
3. **M5.1 (the ceiling):** encoder gain ≠ system gain. Warm service 1.081×,
   actual batch 1.029×, cold-inclusive 1.33× — three different questions,
   three different numbers, one Amdahl ceiling.
4. **Still failing (kept honest):** C2 INT8 empty-negative gate **FAIL**
   (28.25%) and the FP32-stored-IR variant **FAIL** (6.63%) — no acceptance
   can promote them; the 40-step sweep stays **NOT_RUN**.

This arc is why the course exists: it teaches that *architecture decisions
are hypotheses*, implementations are treatments, and systems keep score in
units you must define yourself.

---

## 6. Verdict vocabulary and evidence rules

- Statuses: `PASS / FAIL / BLOCKED / NOT_RUN / PENDING` — and a FAIL can only
  be cleared by re-running the failed gate, never by a subjective acceptance
  or an adjacent success.
- Claims ≤ evidence: every number cites a run record path; every run record
  has a source HEAD; the registry
  ([`data/verified_claims.json`](data/verified_claims.json)) is the machine-
  readable index, validated by `scripts/course-quality/check_claims.py`.
- Scope discipline: different run populations (cold vs warm), memory
  definitions (campaign peak 17.26 GB vs balanced-audit 16.18 GB), and
  precision configurations (FP16-stored-BF16-exec vs FP32-stored) are never
  interchangeable.
- Human-quality verdicts are recorded through an explicit mechanism with a
  reviewer and scope (see `docs/release-closure/PR2_OWNER_QUALITY_VERDICT.md`),
  and never propagate beyond their declared scope.

---

## 7. From student to contributor

The loop ends in contribution: Issue → Hypothesis → Reproducer → Code →
Tests → Evidence → PR → Review. M7 grades exactly that pathway; the PR
template (`.github/PULL_REQUEST_TEMPLATE/optimization.md`) asks for the
bottleneck, the change, the constants, the before/after numbers, the quality
effect, the resource cost, reproducibility, and the claims that remain
unverified. If you can fill that template honestly, you have done — and can
defend — real optimization engineering.
