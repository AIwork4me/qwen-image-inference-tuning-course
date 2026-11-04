# Radeon Cloud Quickstart — from zero to your first image

**Target reader:** a student who just got a Radeon Cloud instance approved.
Total time on the audited instance: **~10–20 minutes** — dominated by the
25.5 GB staged model download, which measured ~0.4 GB/s aggregate on the
audited cloud network (hf-mirror, 16-way ranged). **Your network will
vary**: at a typical 20–40 MB/s the same download takes 11–22 minutes...
at 2 MB/s it is ~3.5 hours — start it early and let the resumable
downloader work. Re-runs are minutes (everything verifies and skips).

> Before you start: apply for free Radeon Cloud GPU compute time
> (platform: <https://developer.amd.com.cn/radeon>, docs:
> <https://amd-aim.github.io/radeon-cloud-docs/zh-cn/>). Choose a
> **Linux Notebook image with PyTorch-ROCm preinstalled** and a
> **single GPU**. Quotas (free hours, GPU model, storage) are shown in the
> platform console — they change over time, so this guide never assumes a
> number.

## The three commands

Open a terminal in your Notebook and run:

```bash
cd /workspace
git clone https://github.com/AIwork4me/qwen-image-inference-tuning-course.git
cd qwen-image-inference-tuning-course
bash scripts/cloud/start.sh
```

That single script (all steps idempotent — re-run it any time):

1. **Diagnoses your instance** — GPU, ROCm, PyTorch, disk, proxy. No
   secrets, no telemetry; everything is printed to you only.
2. **Fetches ComfyUI at the pinned commit** `b5a4321` (v0.43.0) — via git,
   or a tarball if the cloud egress proxy blocks git (it does, sometimes).
3. **Builds `.venv-lab`** that *borrows* the preinstalled ROCm PyTorch
   through a read-only bridge — your working torch is never reinstalled
   or upgraded.
4. **Stages the minimum model set for a first image** (~25.5 GB:
   turbo-int8 DiT + 8B text encoder + VAE), resumable 16-way download from
   the mirror your network allows, then verifies every file's **pinned
   SHA256**.
5. **Runs the cloud doctor** — fail-closed checks (GPU, stack, digests,
   prompts, SDPA probe, rocprofv3).
6. **Generates ONE bounded smoke image** and prints its path plus your
   next lesson.

Expected tail of a healthy first run (measured on the audited instance,
W7900-class):

```
== 6/6 First image (bounded smoke, BEST config = turbo-int8 8-step) ==
[BEST] starting fresh server (flags=[], sdp=none) ...
  cold          p1-zh-scene seed=1000042 total=33.067s ...
  measure       p1-zh-scene seed=42      total=9.023s  ...
== YOUR FIRST IMAGE ==
  image  : /workspace/qwen-image-inference-tuning-course/runs/raw/<tag>-BEST-smoke/p1-zh-scene-42-r0.png
  record : /workspace/qwen-image-inference-tuning-course/runs/<tag>-BEST-smoke/p1-zh-scene-42-r0.json
```

Your measurements live under `runs/` and can **never** modify the frozen
reference evidence under `evidence/` — that separation is enforced by the
code itself, not by convention.

## Options

| Flag | Effect |
|---|---|
| `--all-models` | fetch all six pinned variants (~61 GB; checks disk budget first) — needed for B0/F1/F2/L1/L1O/LF1/H1/H2 |
| `--skip-smoke` | stop after the doctor (no image) |
| `--smoke-only` | re-run just the smoke (setup steps verify themselves) |
| `--hf-base URL` | model source override (default `https://hf-mirror.com`; use `https://huggingface.co` where reachable) |

## After the first image: the staged path through the course

```bash
bash scripts/cloud/start.sh --all-models     # unlock the bf16 baseline (+14.2 GB)
.venv-lab/bin/python -m lab baseline         # M01: the frozen B0 reference (~16 min)
```

GPU-time budget per FULL protocol session (W7900-class; smoke variants
cost 1–3 min each and are what the classroom uses):

| Experiment | Full session | Smoke |
|---|---|---|
| B0 baseline (40 steps) | ~16 min | ~3 min |
| L1 / L1O (turbo 8-step) | ~5 min | ~2 min |
| F1 (ck attention) | ~13 min | ~3 min |
| F2 (INT8 40-step) | ~15 min | ~3 min |
| LF1–LF3, BEST | ~4 min | ~2 min |
| H1 (CPU TE probe) | **~15 min, slow CPU work — opt-in research, taught from the committed evidence by default** |
| H2 (pipeline A/B) | **~15 min — opt-in research, same policy** |

**Save your GPU credits:** stop (or destroy) the instance whenever you are
not actively measuring. `/workspace` — with the repo, weights, `.venv-lab`
and your `runs/` — persists across stops, so you lose nothing.

## Tested recovery guide (every failure below was reproduced on the live
instance)

### `git clone` fails with `CONNECT tunnel failed, response 503`
The cloud egress proxy intermittently refuses git's HTTP/2 tunnels.
`start.sh` already retries with HTTP/1.1. If cloning the COURSE repo still
fails:

```bash
git config --global http.version HTTP/1.1
git clone https://github.com/AIwork4me/qwen-image-inference-tuning-course.git
```

### A model download was interrupted (instance stop, network drop)
Just re-run `bash scripts/cloud/start.sh`. The downloader keeps per-chunk
part files and resumes byte-exact; completed files are detected and
skipped by size + SHA256 stamp.

### `digest verification FAILED` in step 4
One file is corrupt (torn download). Delete it and re-run:

```bash
rm weights/comfyui-split/<the-file-listed>.safetensors
bash scripts/cloud/start.sh
```

### `BLOCKED: <N> GiB free ... needs ~25 GiB`
Your persistent workspace cannot hold even the staged set. Either free
space (`du -sh /workspace/*`), attach/expand the platform's persistent
storage, or pick an instance flavor with ≥ 30 GB workspace (≥ 65 GB for
`--all-models`). This is a provisioning problem — no code change will fix
a full disk.

### pip step fails (`Could not find a version ...`)
The image's default pip mirror (Tsinghua) sometimes lags fresh packages.
`start.sh` already retries against official PyPI. Manual equivalent:

```bash
.venv-lab/bin/pip install --index-url https://pypi.org/simple/ \
  -r ComfyUI/requirements.txt -r requirements-dev.txt
```

### The strict doctor fails on `gpu_gfx1100_single` but the cloud doctor passes
You are on a different Radeon (arch/VRAM). That is fine for learning — the
cloud doctor (`--cloud`) warns instead of failing — but your numbers are an
independent measurement, not a reproduction of the frozen reference, and
bf16 experiments may not fit in VRAM (check the WARN text).

### Instance restarted / reconnected
Everything under `/workspace` survived. Re-run `bash scripts/cloud/start.sh`
— it verifies each stage (marker files, digests, venv importability) and
only repairs what is missing. If `/tmp` was wiped, nothing course-related
was stored there.

### Server died during boot (`ComfyUI died during boot`)
Read the tail of the run's `server.log`:

```bash
tail -40 runs/<tag>-BEST-smoke/server.log
```

The two boot-killers seen in the wild: a corrupt weights file (fix above)
and the historical stdlib-`profile` shadowing bug (already fixed in this
repo — if you see `ImportError: attempted relative import with no known
parent package` mentioning `lab/profile.py`, `git pull`).

## Where to go next

1. [course/M01_baseline/LESSON.md](../course/M01_baseline/LESSON.md) —
   measure the frozen baseline and read your first run record.
2. [docs/RADEON_CLOUD_READINESS.md](RADEON_CLOUD_READINESS.md) — what the
   audited instance looked like and every platform quirk encoded in
   `start.sh`.
3. [course/SYLLABUS.md](../course/SYLLABUS.md) — the full M00–M07 path.
