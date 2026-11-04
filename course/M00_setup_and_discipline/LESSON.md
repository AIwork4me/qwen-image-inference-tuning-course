# M0 — Setup & the Discipline

**Problem.** You just got a cloud GPU and you want to make an image model
faster. The professional's first move is the opposite of what you expect:
*prove the environment is trustworthy before touching anything*. A
benchmark that silently falls back to CPU, or a server that quietly
updated itself mid-study, produces confident nonsense — and you would
publish it.

**Concepts (beginner glossary first).**

| Term | Plain meaning | Why it matters here |
|---|---|---|
| **GPU / VRAM** | the accelerator card and its own memory (a W7900-class card: 48 GB) | models must *fit*; weights + activations both count |
| **ROCm** | AMD's GPU software stack (driver + libraries), the counterpart of CUDA on NVIDIA | "torch+rocm" means PyTorch built for AMD GPUs |
| **gfx1100** | the GPU's architecture generation (RDNA3, Navi31) | kernel behavior differs per generation; our evidence is gfx1100 |
| **ComfyUI** | a node-graph server that runs diffusion models over HTTP | we benchmark *through* it, like a real deployment |
| **pinned commit / SHA256 pin** | freezing a software version / a file's cryptographic fingerprint | a moving server or a corrupted weight invalidates every comparison |
| **fail-closed** | if a check cannot run, the tool REFUSES to continue | the alternative (fail-open) measures nothing and reports numbers |
| **warm / cold / fresh** | repeated / first-ever / new-prompt runs | different latencies; mixing them is a classic bug (M1) |

The frozen stack:

| Layer | What we use | Why it's pinned |
|---|---|---|
| GPU | one AMD Radeon gfx1100-class (W7900 48 GB in the reference campaign) | single-GPU claims only |
| OS / stack | Linux, ROCm 7.2.1, torch 2.9.1+rocm | the validated env; never "upgraded" mid-study |
| Server | ComfyUI v0.43.0 @ commit `b5a4321` | a moving server invalidates every comparison |
| Weights | Comfy-Org/Qwen-Image-2.1 @ `aacf31a6`, SHA256-pinned | integrity anchored in-repo (`lab/weights.py`) |
| Protocol | 3 prompts × 3 seeds, euler/simple/CFG 1 | frozen, so all A/Bs are comparable |

**Run (classroom, ~15 min one-time + 2 min daily).**

```bash
bash scripts/cloud/start.sh          # one-command setup (M0's practical core)
.venv-lab/bin/python -m lab doctor --cloud   # staged preflight
.venv-lab/bin/python -m lab doctor           # strict preflight (formal runs)
```

Budget: the first `start.sh` downloads ~25.5 GB (10–20 min on the audited
cloud network); re-runs verify in seconds. `doctor` costs well under a
minute of GPU time (it launches one tiny attention kernel).

**Observe (real output from a live Radeon Cloud instance, 2026-11-04).**

```
PASS gpu_amd_single_working: AMD Radeon Graphics / gfx1100 / 49136MB
PASS torch_rocm: torch 2.9.1+gitff65f5b hip 7.2.53211-e1a6bc5663
PASS comfyui_pinned: v0.43.0 @ b5a432157de6
PASS weights_stage1_integrity: {...per-file verdicts...}
PASS prompts_frozen: 3 prompts x 3 seeds, ids intact
PASS flash_sdpa_probe: flash attention launches + syncs on gfx1100
PASS rocprofv3: version: 1.1.0
verdict: PASS  evidence: runs/doctor.json
```

And a real failure, because failure teaches more than success — a
corrupted (size-mismatched) weight file:

```
FAIL weights_stage1_integrity: {'per_file': [
  {'rel': 'diffusion_models/qwen_image_2.1_turbo_int8_convrot.safetensors',
   'ok': False, 'reason': 'size-mismatch',
   'local_size': 5968230400, 'expected_size': 7256787829, ...}]}
```

The doctor's output goes to `runs/doctor.json` — the shipped
`evidence/results/doctor.json` belongs to the frozen campaign and is never
overwritten by your runs.

**Explain.** The discipline, read twice:

1. **Fail-closed preflight** — doctor refuses to measure on the wrong GPU,
   a drifted ComfyUI, or a corrupted weight. Read the failure, fix the
   cause, re-run.
2. **Evidence, not anecdotes** — every run writes a schema-valid JSON
   record (timing, VRAM samples, attention-path evidence scraped from the
   server log, image SHA256, env snapshot). Records are append-only and
   land in `runs/`, physically separated from the frozen reference under
   `evidence/`.
3. **Latency vocabulary** — cold / fresh / warm / boot are *different
   numbers* and never mixed.
4. **VRAM is sampled, not exact** — peaks from 0.5–1 s polls are lower
   bounds; we record both torch-level and device-level (rocm-smi) views.

How SHA256 verification actually behaves (read precisely — the docs claim
exactly this): the first check of each file hashes it completely; a
successful hash is stamped (size + mtime + digest) and later checks with
an unchanged file return `stamp-cache-hit` without re-hashing; ANY file
change invalidates the stamp and forces a fresh full hash. A wrong *size*
fails instantly without hashing. What does the size-only shortcut miss
that the full hash catches? A file rewritten with the same length but
different bytes — e.g. a torn download reassembled at the right size
(this exact failure killed the parent project's E1 turbo bench: the
safetensors header gate in `lab/download_weights.py` exists because of it).

Case-study anchor: [docs/case-study/P0_PREFLIGHT.md](../../docs/case-study/P0_PREFLIGHT.md)
— the audit found a healthy W7900 stack *and* real blockers (egress
proxy, one environment-specific test failure), all recorded, none hidden.
The verdict vocabulary everywhere: **PASS / FAIL / BLOCKED / NOT_RUN**.

**Think (answer keys in `<details>`).**

1. Why does the doctor hash the whole file instead of trusting sizes?
   <details><summary>answer</summary>A size collision is trivial to
   produce (torn downloads, partial writes). SHA256 pins make the in-repo
   table the trust anchor: any byte-level corruption is caught regardless
   of length.</details>
2. Your cloud instance shows `gfx1100` but a colleague's shows `gfx1201`
   and doctor prints a WARN, not a FAIL (cloud mode). May you both take
   the course? May you pool your numbers into one median?
   <details><summary>answer</summary>Yes to learning — the methodology is
   hardware-agnostic. No to pooling: different architectures are different
   measurement populations; each set of numbers is valid only for its own
   host (that is why `report --scope student` never mixes with the frozen
   reference).</details>
3. Why is the doctor's own report written to `runs/`, not `evidence/`?
   <details><summary>answer</summary>`evidence/` is the immutable record
   of the official campaign; anything your machine produces is *your*
   evidence. Appending would contaminate the published medians — the exact
   bug class this repo guards against in code.</details>

**Acceptance.**
- [ ] `start.sh` completed and printed your first-image path (or you
      documented the actionable blocker it reported)
- [ ] `doctor --cloud` verdict is PASS (strict PASS too once you have all
      six models)
- [ ] You can name the 7 strict-doctor checks and what each protects
      against (open `runs/doctor.json`)
- [ ] You explained the stamp-cache behavior and the attack it still
      catches, in your own words
