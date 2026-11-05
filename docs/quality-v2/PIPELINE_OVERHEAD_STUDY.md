# Course-v2 Pipeline Overhead Study (P2)

**Run IDs:** `overnight-course-v2-a7fa69a-p2profile-140000` (profiling),
`overnight-course-v2-a7fa69a-p2ab-152000` (controlled A/B — the quoted numbers).
**Host:** the physical W7900-class gfx1100 campaign host (16-CPU/55 GiB cgroup),
2026-11-05/06. **Module:** `lab/m51_pipeline_profile.py` (experiment-mode only;
the frozen production harness `lab/m51_pipeline_warm.py` is untouched).

## 1. Research question

The M5.1 release-closure audit measured warm per-request **service latency**
1.081× (A 13.031 s vs C 12.060 s, n=8/8 non-overlapping) but actual
four-request **batch throughput** only 1.029× (4.42 → 4.55 img/min). Where
does the precomputed arm's ~0.54 s/request of extra orchestration time go,
and is any of it client-controllable without touching model, precision,
tokenizer, conditioning tensors, seed, latent, DiT, VAE, scheduler, steps,
or output quality?

**Pre-registered exploratory target:** ≥5% additional batch-throughput
improvement over C. (Target used for exploration only — not a success
threshold to be fabricated.)

## 2. Step A — instrumented profiling (fine-grained per-request timeline)

Each request was instrumented with a same-process monotonic timeline:
post → first-node executing → per-node durations → ws completion → history
visible → image fetched → file written → next post. Cross-process CPU-encode
intervals come from the worker's wall-clock replies (interpreted, per the
repo's clock-semantics convention). Batches interleaved
`poll, ws, poll, ws` in one warm session (completion detection being the
only variable between poll and ws).

**Findings (run -140000, confirmed by -152000):**

| Candidate overhead | Median (poll mode) | Median (ws mode) | Verdict |
|---|---:|---:|---|
| Completion detection (poll quantization after done) | 0.001–0.3 s | 0.008–0.14 s | **NOT the bottleneck** — poll/ws batch ratio 1.004 |
| Queue→first-node | 0.002–0.005 s | same | negligible |
| Image fetch (HTTP GET ~1.5 MB localhost) | 0.007 s | 0.005 s | negligible |
| File write + sha256 | 0.001–0.002 s | same | negligible |
| Workflow build | 0.0002 s | same | negligible |
| **Per-request RunMonitor teardown** (between requests: `loop_head`) | **0.69–0.99 s** | same | **THE overhead** |

**Root cause:** the production request path (`lab.m51_e2e.run_one`, used
unchanged by `lab.m51_pipeline_warm`) creates a fresh `RunMonitor` per
request and tears it down inside the batch gap: `RunMonitor.stop()` joins a
websocket receive loop whose socket timeout is 1.0 s and a VRAM poller that
waits up to 0.5 s — a phase-dependent 0.4–1.0 s stall per request, in BOTH
arms (arm A too). The audit's "C carries ~0.54 s/request more than A"
decomposes as: pipeline fill (~0.11–0.15 s/req amortized) + this shared
teardown cost landing at systematically worse phases in the precomputed arm.

Also observed (both profile runs): ComfyUI evicted the 16.8 GB CLIP after
the A-session warmup, so each A session's first measured request paid an
~11.5 s `CLIPLoader` reload (25.1–25.3 s walls vs 13.04 steady) — the same
session-start mechanism as the campaign's cold-inclusive first request
(26.07 s). Documented as known behavior; it does not affect C-arm
comparisons (no CLIP in that session).

## 3. Step B — one bounded optimization

**C-opt ("fast" mode), single variable:** keep ONE persistent RunMonitor
for the whole batch (websocket events sliced per request for correct node
timing); keep the existing next-conditioning prefetch; everything else —
workflows, worker, seeds, image fetch, file write — unchanged. Code:
`run_one_fast`/`batch_fast` in `lab/m51_pipeline_profile.py` (~90 lines,
experiment-mode; the frozen harness is not modified).

Rejected as unjustified by profiling: reduced client polling (already
~1 ms), request-submission orchestration (2 ms), redundant serialization
(0.2 ms), persistent model residency (out of scope: changes server state).

## 4. Step C — controlled A/B (interleaved, warm session)

Arms: `C[poll]` (production path) vs `C[fast]` (C-opt), interleaved
poll,fast,poll,fast in ONE warm server session with the same worker,
prompts, seeds, and cgroup budget. Arm A measured in its own warm session
for narrative reference. Correctness contract: conditioning-file SHA256
per request identical across modes (`cond_sha256_stable_within_request:
req1–req4 all true`); identical seeds/workflows; image bytes differ only by
the stack's documented run-to-run kernel nondeterminism (noise-floor claim:
perceptually identical, LPIPS 0.0000).

| Batch | Mode | Batch wall (4 requests) | loop_head median |
|---|---|---:|---:|
| C1 | poll | 53.218 s | 0.993 s |
| C2 | **fast** | **50.157 s** | **0.001 s** |
| C3 | poll | 52.304 s | 0.993 s |
| C4 | **fast** | **50.483 s** | **0.001 s** |

- **Measured batch-throughput improvement: 52.761 → 50.320 s = 1.0485×
  (+4.85%)**, consistent across both interleaved pairs, batch-wall
  distributions non-overlapping; the mechanism (teardown elimination,
  ~3×0.99 s ≈ 2.98 s predicted vs 2.44 s measured after fill/drain) is
  directly attributable.
- Service walls unchanged (poll 12.09 s vs fast 12.19 s medians — GPU path
  untouched, as designed).
- VRAM/RSS: unchanged (client-side only).
- **Relative to arm A** (warm service 13.037 s median, actual throughput
  4.42 img/min): the fast client lifts C's actual throughput to
  4/50.32×60 = **4.77 img/min → 1.079×**, i.e. with harness overhead gone,
  the batch-throughput ratio nearly matches the service-latency ratio
  (1.081×). The audit's "missing" throughput reappears — confirming the
  diagnosis.

## 5. Step D — decision (honest, pre-registered)

The +4.85% measured improvement is **real and attributable** (non-overlapping
batches, exact mechanism, correctness contract intact), but it lands **just
below the pre-registered 5% exploratory target**. Per the pre-registration
discipline:

1. **The production implementation is kept unchanged.**
   `lab/m51_pipeline_warm.py`, `lab/m51_e2e.py` and every historical M5.1
   measurement stay byte-identical; the optimization lives only in the
   experiment module behind its explicit mode, for students to reproduce.
2. **Reported as a near-miss positive with a proven mechanism**, not as a
   success. The interesting scientific result is not the 4.85% — it is that
   the service-vs-batch gap (1.081× vs 1.029×) is *client-harness overhead,
   not deployment-architecture cost*, and that eliminating it closes the gap
   to 1.079×.
3. **Educational framing:** a benchmark harness is part of the measured
   system. If your client pays 1 s/request of teardown, your "architecture
   comparison" silently measures your client. This is now a teaching example
   in the M5.1 lesson's exercise and the lecture's Q5.

### Student exercise (also in M5.1 lesson)

Given the table in §4: (a) predict the batch-wall saving from the
loop_head column before reading §4's walls; (b) explain why the saving is
3×0.99 s ≈ 2.98 s rather than 4×0.99 s; (c) explain why service walls are
unchanged; (d) state the correctness contract and why image hashes may
still differ. *(Answers: teardown after the final request falls outside the
batch wall; the GPU path is untouched; conditioning SHA256 + seed +
workflow identity; ROCm run-to-run kernel nondeterminism — the course's
noise-floor claim.)*

## 6. Limitations

- n=2 interleaved batches per mode (bounded GPU budget); walls non-overlapping
  but the sample is small — stated, not hidden.
- Single host, single session; the monitor-teardown cost is
  phase-dependent (0.4–1.0 s), so other sessions will land differently.
- Arm A paid a documented session-start CLIP reload on its first request in
  both profile runs; A numbers here are for narrative only, and the C-vs-C
  comparison is unaffected.
- The fast client changes the *measurement harness*, not the served system;
  any production deployment should use a persistent-connection client as
  standard practice (this study quantifies what that practice is worth).

## 7. Research extension (open hypotheses)

1. Does a persistent monitor also lift arm A's actual throughput (its gaps
   carry the same teardown)? Predicted: A → ~4×13.04+0.5 ≈ 52.7 s ≈
   4.56 img/min, preserving the ~1.05× A/C batch ratio — i.e. the ratio is
   architecture-bound once both arms use equal clients. (Untested; a clean
   follow-up A/B.)
2. Would overlapping image fetch/write with the next request's submission
   recover anything further? (Profiling says ≤ 10 ms/request — predicted
   no; a one-line falsification experiment.)

## 8. Evidence index

- Raw per-request timelines, batch records, attribution summary:
  `runs/overnight-course-v2-a7fa69a-p2ab-152000/pipeline_profile/pipeline_profile.json`
  (+ `raw/*.png`, server logs, worker cond files with SHA256 in records).
- First profiling run (poll/ws modes):
  `runs/overnight-course-v2-a7fa69a-p2profile-140000/pipeline_profile/pipeline_profile.json`.
- Module: `lab/m51_pipeline_profile.py` (CPU-import-safe; `attribute_request`
  unit-tested in `tests/test_course_quality.py`).
- GPU budget used: ~23 minutes wall across both runs (well under the 3 h
  mission ceiling).
