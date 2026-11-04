# Attribution & boundaries

## What this course derives from

- **[AIwork4me/qwen-image-rocm](https://github.com/AIwork4me/qwen-image-rocm)** (MIT)
  — the parent community benchmark project. Its
  `feat/linux-w7900-optimization-lab` branch provided:
  - the lab code (adapted into `lab/`): protocol core, ComfyUI client,
    runner, profiler, pipeline experiments, quality gate, weight pins;
  - every measurement in `evidence/` (records, grids, charts), produced on
    the lab Radeon PRO W7900 on 2026-11-03;
  - the P0–P6 engineering journals (verbatim in `docs/case-study/`).
- **[ComfyUI](https://github.com/comfyanonymous/ComfyUI)** (GPL-3.0) —
  cloned at pinned commit `b5a4321` (v0.43.0) at setup time; none of its
  code is copied into this repository.
- **Comfy-Org/Qwen-Image-2.1** weight release (revision `aacf31a6`) —
  downloaded by each user, SHA256-pinned in `lab/weights.py`, never committed.
- **Qwen-Image-2.1 / Qwen3-VL models** — Alibaba (Qwen) upstream licenses
  apply to the weights and the images generated with them.

## Boundaries

- Independent educational material — **not** an official AMD, Comfy.Org or
  Alibaba/Qwen project.
- All performance numbers describe the specific measured hardware/stack
  (single W7900, ROCm 7.2.1, torch 2.9.1+rocm, 2026-11-03). They are
  reproducible methodology, not universal product claims.
- The pending human quality verdict (M6) means no shipped configuration is
  quality-approved yet.

## Sensitive-content audit

Evidence batches are pattern-scanned for tokens/keys before commit; no
credentials, model weights, or massive traces are committed (the rocprof
raw CSVs and raw generation PNGs stay on the measurement host; their
analyses and hashes are committed).
