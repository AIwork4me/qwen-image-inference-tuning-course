# M0 — Setup & the Discipline

**Goals:** get the lab running, and internalize *why* every measurement in
this course is gated, pinned, and recorded.

## Lesson

### The stack

| Layer | What we use | Why it's pinned |
|---|---|---|
| GPU | one AMD Radeon PRO W7900 (gfx1100, 48 GB) | single-GPU claims only |
| OS / stack | Linux, ROCm 7.2.1, torch 2.9.1+rocm | the validated env; never "upgraded" mid-study |
| Server | ComfyUI v0.43.0 @ commit `b5a4321` | a moving server invalidates every comparison |
| Weights | Comfy-Org/Qwen-Image-2.1 @ `aacf31a6`, SHA256-pinned | integrity anchored in-repo (`lab/weights.py`) |
| Protocol | 3 prompts × 3 seeds, euler/simple/CFG 1 | frozen, so all A/Bs are comparable |

### The discipline (read this twice)

1. **Fail-closed preflight** — `python -m lab doctor` refuses to measure on
   the wrong GPU, a drifted ComfyUI, or a corrupted weight file. A benchmark
   that silently falls back to CPU produces confident nonsense.
2. **Evidence, not anecdotes** — every run writes a schema-valid JSON record
   (timing, VRAM samples, attention-path evidence scraped from the server
   log, image SHA256, env snapshot). Records are append-only.
3. **Latency vocabulary** — cold / fresh / warm / boot are *different
   numbers* and never mixed (a warm number is not "user latency").
4. **VRAM is sampled, not exact** — peaks from 0.5–1 s polls are lower
   bounds; the course records both torch-level and device-level (rocm-smi).

### Case-study anchor

Read [docs/case-study/P0_PREFLIGHT.md](../../docs/case-study/P0_PREFLIGHT.md):
the audit found a healthy W7900 stack *and* real blockers (egress proxy,
one environment-specific test failure) — all recorded, none hidden. The
verdict vocabulary you'll see everywhere: **PASS / FAIL / BLOCKED / NOT_RUN**.

## Exercise

1. Clone this repo; create `.venv-lab`; install ComfyUI requirements into it
   (see [docs/REPRODUCTION_GUIDE.md](../../docs/REPRODUCTION_GUIDE.md) §1).
2. Fetch ComfyUI @ `b5a4321` and the pinned weights (digests in `lab/weights.py`).
3. Run `python -m lab doctor`. **It must print `verdict: PASS`.**
4. Open `evidence/results/doctor.json` — list the 7 checks and what each
   protects against.
5. Answer: why does the doctor re-hash every weight file instead of trusting
   file sizes? (Hint: what attack does a size-only check miss?)

**Acceptance:** doctor PASS screenshot/output + your answers in the notebook.
