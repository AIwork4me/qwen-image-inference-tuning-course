# Lab results summary (auto-generated)

Generated: 2026-11-04T04:49:14+00:00  ·  scope: **student**  ·  records: 74

Scope `reference` = the frozen official W7900 campaign only (lab/reference_runs.json). Student runs live under `runs/` and are aggregated separately (`python -m lab report --scope student`).

| Exp | Category | warm med (s) | step med (s) | fresh (s) | TE (s) | cold (s) | VRAM peak (MB) | util% p50 | attn | n | quality |
|---|---|---|---|---|---|---|---|---|---|---|---|
| B0 | baseline | 55.099 | 1.34 | 56.108 | 0.572 | 95.583 | 37709.0 | 100.0 | pytorch-sdpa-default | 9 | REFERENCE |
| BEST | final | 9.024 | 0.922 | 10.026 | 0.557 | 25.576 | 31026.8 | 100.0 | pytorch-sdpa-default | 9 | RECORDED_OR_PENDING_HUMAN_REVIEW |
| F1 | faster-computation | 43.375 | 1.049 | None | None | 80.211 | 37710.5 | 100.0 | ck | 9 | RECORDED_OR_PENDING_HUMAN_REVIEW |
| F2 | faster-computation | 50.285 | 1.222 | None | None | 72.149 | 30925.0 | 100.0 | pytorch-sdpa-default | 9 | RECORDED_OR_PENDING_HUMAN_REVIEW |
| L1 | less-computation | 12.119 | 1.338 | 13.03 | 0.548 | 49.765 | 37708.5 | 100.0 | pytorch-sdpa-default | 9 | RECORDED_OR_PENDING_HUMAN_REVIEW |
