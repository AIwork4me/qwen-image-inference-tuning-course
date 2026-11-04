# Reproduction Guide — Linux W7900 Lab

Everything below was executed on the physical W7900 lab (ROCm 7.2.1,
torch 2.9.1+rocm7.2.1, ComfyUI v0.43.0 @ b5a4321, weights @ aacf31a6,
SHA256-pinned). Expect your absolute numbers to differ on other hardware or
stacks — the *methodology* is what reproduces.

## 0. Preflight (fail-closed)

```bash
python -m lab doctor
```

Must print `verdict: PASS`. It checks: exactly one AMD gfx1100 GPU with ~48 GB,
torch/ROCm versions, ComfyUI pinned-commit marker, SHA256 of every weight file,
a live flash-SDPA probe, and rocprofv3 availability. Evidence is written to
`evidence/results/doctor.json`.

## 1. Environment

- Python 3.12 venv (`.venv-lab`), ComfyUI's `requirements.txt` installed
  into it, borrowing the validated ROCm torch via a read-only `.pth` bridge
  (never reinstall torch over a working ROCm build).
- ComfyUI at `ComfyUI` (pinned commit), weights at
  `weights/comfyui-split/` (pinned digests; on hostile networks use
  `lab/download_weights.py` — 16-way ranged resume).

## 2. Frozen baseline (Phase 1)

```bash
python -m lab run --experiment B0 --date <tag>
```

Protocol: 3 prompts × seeds [42, 1337, 20261030]; cold (seed+1M), warmup
(seed+2M, cache-defeating), 3 measured per prompt, fresh-prompt arm (" ."
suffix) — 16 runs, fresh server. Records land in
`evidence/results/<tag>-B0/` (schema-valid JSON) and raw PNGs in
`.../raw/<tag>-B0/`. Two independent sessions were run for reproducibility
(`p1-baseline-B0`, `p1-baseline-r2-B0`): warm medians 55.24 vs 55.11 s.

## 3. Profiling (Phase 2)

```bash
python -m lab profile --date <tag>
```

Layer A aggregates the B0 records into the pipeline breakdown
(`pipeline_breakdown.json`). Layer B launches the real ComfyUI server under
`rocprofv3 --runtime-trace --output-format csv -P 230:90:1` with a driver
client that aligns one warm generation inside the bounded window; shutdown is
a SIGINT **to the rocprofv3 wrapper** (SIGTERM anywhere leaves unconverted
`.dat` buffers — reproduced twice before the fix). Outputs:
`rocprof-{dit,te}/analysis.json` + the raw CSVs (~85 MB each, not committed).

## 4. Experiments (Phases 3–5)

```bash
for EXP in L1 L1O F1 F2 LF1 LF2 LF3 BEST; do
  python -m lab run --experiment $EXP --date <tag>-$EXP
done
python -m lab run --experiment H1   # CPU TE probe (~15 min)
python -m lab run --experiment H2   # pipeline A/B (~15 min)
```

Every run: fresh server, health gate, schema-valid records, attention-path
evidence scraped from the server log, image SHA256, torch + rocm-smi VRAM
peaks, GPU utilization samples.

## 5. Quality gates (Phase 6)

```bash
.venv-lab/bin/python -I lab/quality_gate.py \
  --baseline evidence/results/raw/<tag>-B0 \
  --experiment evidence/results/raw/<tag>-LF2 \
  --experiment-name LF2 --semantics distance \
  --baseline-desc "<describe the baseline>" \
  --noise-a <same-config img A> --noise-b <same-config img B> \
  --out evidence/results/quality-LF2
```

Semantics: `fidelity` for same-model numerics variants (INT8), `distance`
for different schedules (Turbo). The noise floor comes from the two
independent B0 sessions (measured 0.0000 on 9/9 pairs). The human
three-tier verdict on `quality_grid.png` + `quality_crops_poster.png` is the
binding gate — prepare grids, don't fabricate judgments.

## 6. Report

```bash
python -m lab report     # SUMMARY.json + SUMMARY.md
```

## Known reproduction hazards (measured, all in P0/P2)

| Hazard | Workaround used |
|---|---|
| Egress proxy kills git-clone / large single responses | repo via mirror with per-file size verification; weights via 16-way ranged download |
| `rglob` unreliable on the ext4 loop fs | explicit `iterdir`/`glob` in analysis code |
| Filesystem mtime_ns frozen across rapid writes | 1 pre-existing CPU test (`stamp cache`) fails locally; not a code regression |
| rocprofv3 + SIGTERM → no CSVs | always SIGINT the rocprofv3 wrapper |
| `rocm-smi -g` shows clocks, not utilization | use `-u --csv` for GPU use %, `--showmeminfo vram --csv` for device VRAM |
