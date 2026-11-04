# Reproduction Guide — Linux W7900 Lab

Everything below was executed on the physical W7900 lab (ROCm 7.2.1,
torch 2.9.1+rocm7.2.1, ComfyUI v0.43.0 @ b5a4321, weights @ aacf31a6,
SHA256-pinned), and re-validated 2026-11-04 end-to-end on a live Radeon
Cloud instance (same stack). Expect your absolute numbers to differ on
other hardware or stacks — the *methodology* is what reproduces.

## 0. Preflight (fail-closed)

```bash
bash scripts/cloud/start.sh        # one-command setup (see docs/RADEON_CLOUD_QUICKSTART.md)
.venv-lab/bin/python -m lab doctor --cloud   # staged mode (first-image set)
.venv-lab/bin/python -m lab doctor           # strict mode (formal runs) — must print verdict: PASS
```

The strict doctor checks: exactly one AMD gfx1100-class GPU with ~48 GB,
torch/ROCm versions, ComfyUI pinned-commit marker, SHA256 of every weight
file (full hash on first check; size+mtime-keyed stamp cache afterwards —
any file change forces a fresh full hash), frozen prompts, a live
flash-SDPA probe, and rocprofv3 availability. Output lands in
`runs/doctor.json`; the shipped `evidence/results/doctor.json` belongs to
the frozen campaign and is never overwritten.

## 1. Environment

`start.sh` builds everything. Manual equivalent, if you need it:

- `.venv-lab` created from the active python (on Radeon Cloud:
  `/opt/venv`, which already carries the ROCm torch), with a read-only
  `.pth` bridge to the system site-packages — **torch is never
  reinstalled over a working ROCm build**;
- ComfyUI's `requirements.txt` + `requirements-dev.txt` installed into
  `.venv-lab` (fall back to `--index-url https://pypi.org/simple/` if the
  configured mirror lags a package);
- ComfyUI at `ComfyUI` (pinned commit `b5a4321`; tarball fallback when
  the egress proxy blocks git), weights at `weights/comfyui-split/`
  (pinned digests; on hostile networks `lab/download_weights.py` — 16-way
  ranged resume; on the audited cloud network huggingface.co is blocked
  and hf-mirror.com is the allowed source).

## 2. Frozen baseline (Phase 1)

```bash
.venv-lab/bin/python -m lab run --experiment B0 --date <tag>
```

Protocol: 3 prompts × seeds [42, 1337, 20261030]; cold (seed+1M), warmup
(seed+2M, cache-defeating), 3 measured per prompt, fresh-prompt arm (" ."
suffix) — 16 runs, fresh server. **Records land in `runs/<tag>-B0/`
(schema-valid JSON) and raw PNGs in `runs/raw/<tag>-B0/`.** The frozen
campaign's own two reproducibility sessions live under
`evidence/results/p1-baseline-B0/` and `p1-baseline-r2-B0/`: warm medians
55.24 vs 55.21 s from the committed records (pooled 18-run median 55.21 s;
the per-session figures in the verbatim P1 journal reflect the parent
project's original reporting).

## 3. Profiling (Phase 2)

```bash
.venv-lab/bin/python -m lab profile --date <tag>
```

Layer A aggregates YOUR B0 records (falls back to the frozen reference
records only if you have none) into the pipeline breakdown
(`pipeline_breakdown.json`, with an explicit measured-vs-derived `method`
map). Layer B launches the real ComfyUI server under `rocprofv3
--runtime-trace --output-format csv -P 230:90:1` with a driver client
that aligns one warm generation inside the bounded window; shutdown is a
SIGINT **to the rocprofv3 wrapper** (SIGTERM anywhere leaves unconverted
`.dat` buffers — reproduced twice before the fix). Outputs:
`runs/<tag>-profile/rocprof-{dit,te}/analysis.json` + raw CSVs (~85 MB
each, not committed).

## 4. Experiments (Phases 3–5)

```bash
for EXP in L1 L1O F1 F2 LF1 LF2 LF3 BEST; do
  .venv-lab/bin/python -m lab run --experiment $EXP --date <tag>-$EXP
done
# opt-in research reruns (slow CPU-bound sessions; taught from the
# committed evidence by default — see M5's budget note):
.venv-lab/bin/python -m lab run --experiment H1   # CPU TE probe (~15 min)
.venv-lab/bin/python -m lab run --experiment H2   # pipeline A/B (~15 min)
```

Every run: fresh server, health gate, schema-valid records (under
`runs/`), attention-path evidence scraped from the server log, image
SHA256, torch + rocm-smi VRAM peaks, GPU utilization samples. The runner
pre-checks the manifest's required weight files and prints the exact
download commands when one is missing (staged model policy).

## 5. Quality gates (Phase 6)

```bash
.venv-lab/bin/python -m lab quality --experiment F2     # fidelity semantics
.venv-lab/bin/python -m lab quality --experiment BEST   # distance semantics
```

The command resolves the latest raw image dirs (yours under `runs/raw/`,
or an explicit run via `--date <run-id>`), derives semantics from the
manifest's `quality_semantics` field, and writes grids + LPIPS to
`runs/quality/<exp>/` — the frozen campaign's grids under
`evidence/quality/` are never overwritten. Direct use of the underlying
tool remains available:

```bash
.venv-lab/bin/python -I lab/quality_gate.py \
  --baseline runs/raw/<tag>-B0 \
  --experiment runs/raw/<tag>-LF2 \
  --experiment-name LF2 --semantics distance \
  --baseline-desc "<describe the baseline>" \
  --noise-a <same-config img A> --noise-b <same-config img B> \
  --out runs/quality-LF2
```

Semantics: `fidelity` for same-model numerics variants (INT8), `distance`
for different schedules (Turbo). The noise floor comes from two
independent same-config runs (measured 0.0000 on 9/9 pairs). The human
three-tier verdict on `quality_grid.png` + `quality_crops_poster.png` is
the binding gate — prepare grids, don't fabricate judgments.

## 6. Report

```bash
.venv-lab/bin/python -m lab report                    # frozen reference scope (default)
.venv-lab/bin/python -m lab report --scope student    # your runs/ only
```

The reference scope aggregates ONLY the run dirs listed in
`lab/reference_runs.json` — smoke, duplicate and student data can never
contaminate the published medians (B0 pools its two reproducibility
sessions, n=18; every other experiment n=9).

## Known reproduction hazards (measured, all in P0/P2 + the cloud audit)

| Hazard | Workaround used |
|---|---|
| Egress proxy kills git-clone / large single responses | repo via retry + `http.version=HTTP/1.1`; ComfyUI via codeload tarball; weights via 16-way ranged download from the allowed mirror |
| Platform `curl` silently rewrites huggingface.co → hf-mirror.com (masks the real egress policy) | the lab always names `hf-mirror.com` explicitly (`--hf-base` to override) |
| `rglob` unreliable on the ext4 loop fs | explicit `iterdir`/`glob` in analysis code |
| Filesystem mtime_ns frozen across rapid writes | stamp-cache re-hashes on any change; if a cache hit looks stale, delete `weights/comfyui-split/.digest_cache.json` |
| rocprofv3 + SIGTERM → no CSVs | always SIGINT the rocprofv3 wrapper |
| `rocm-smi -g` shows clocks, not utilization | use `-u --csv` for GPU use %, `--showmeminfo vram --csv` for device VRAM (column layouts differ across ROCm releases — parsed by header) |
| pip mirror (Tsinghua) lags fresh packages | retry with `--index-url https://pypi.org/simple/` |
| `lab/` on the server sys.path shadowed stdlib `profile` | `comfyui_launch.py` strips it (fixed in this repo) |
