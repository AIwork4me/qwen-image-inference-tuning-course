<!--
Optimization PR template — answer honestly. "Claims ≤ evidence" is the
course's first law; empty sections are better than optimistic ones.
-->

## Bottleneck identified

What did you profile, what instrument produced the evidence, and which
fraction of the critical path does the bottleneck occupy? (Cite the record
path under runs/ or evidence/.)

## What changed

One sentence describing the single variable you changed. If you changed
more than one thing, split the PR.

## What was held constant

Model, precision, tokenizer, conditioning tensors, seeds, scheduler, steps,
hardware, cgroup budget, measurement protocol — list what stayed identical
so the A/B is fair.

## Before / after numbers

| Metric (state its semantics: cold/warm/service/batch) | Before | After | n | Spread |
|---|---|---|---|---|

## Quality effect

Which gate applies (fidelity / distance / human three-tier), what the
metrics or grids show, and what remains pending a human verdict.

## CPU/GPU resource cost

VRAM peak, CPU/RSS, contention effects, GPU-time per session — anything a
deployer must budget for.

## Reproducibility

Run-record directories (under runs/), exact commands, seeds, and the
environment facts (GPU arch, VRAM, ROCm, stack). Could the owner re-run this
tomorrow?

## Claims that remain unverified

Negative results, NOT_RUN corners, single-session numbers, anything you
suspect but did not measure. (Honesty here is worth full credit; fabrication
anywhere is grounds for rejection.)
