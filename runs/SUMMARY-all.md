# Lab results summary (auto-generated)

Generated: 2026-11-04T06:04:20+00:00  ·  scope: **all**  ·  kinds: **formal**  ·  records: 216

Scope `reference` = the frozen official W7900 campaign only (lab/reference_runs.json). Student runs live under `runs/` and are aggregated separately (`python -m lab report --scope student`); smoke runs are excluded from formal medians by default (`--kind smoke` to inspect them).

## Reference campaign (frozen, formal only)

| Exp | Category | warm med (s) | step med (s) | fresh (s) | TE (s) | cold (s) | VRAM peak (MB) | util% p50 | attn | n | quality |
|---|---|---|---|---|---|---|---|---|---|---|---|
| B0 | baseline | 55.206 | 1.345 | 56.105 | 0.555 | 74.171 | 37686.8 | 100.0 | pytorch-sdpa-default | 18 | REFERENCE |
| BEST | final | 9.023 | 0.915 | 10.025 | 0.549 | 27.586 | 31026.8 | 100.0 | pytorch-sdpa-default | 9 | RECORDED_OR_PENDING_HUMAN_REVIEW |
| F1 | faster-computation | 45.086 | 1.077 | None | None | 84.577 | 37688.6 | 100.0 | ck | 9 | RECORDED_OR_PENDING_HUMAN_REVIEW |
| F2 | faster-computation | 51.093 | 1.22 | None | None | 80.257 | 31024.7 | 100.0 | pytorch-sdpa-default | 9 | RECORDED_OR_PENDING_HUMAN_REVIEW |
| L1 | less-computation | 12.132 | 1.337 | 13.029 | 0.553 | 48.439 | 37686.6 | 100.0 | pytorch-sdpa-default | 9 | RECORDED_OR_PENDING_HUMAN_REVIEW |
| L1O | less-computation | 12.074 | 1.337 | None | None | 50.324 | 37715.3 | 100.0 | pytorch-sdpa-default | 9 | RECORDED_OR_PENDING_HUMAN_REVIEW |
| LF1 | combined | 10.027 | 1.067 | None | None | 53.666 | 37710.5 | 100.0 | ck | 9 | RECORDED_OR_PENDING_HUMAN_REVIEW |
| LF2 | combined | 9.023 | 0.916 | None | None | 32.431 | 31026.8 | 100.0 | pytorch-sdpa-default | 9 | RECORDED_OR_PENDING_HUMAN_REVIEW |
| LF3 | combined | 9.022 | 0.906 | None | None | 30.414 | 31026.6 | 100.0 | ck | 9 | RECORDED_OR_PENDING_HUMAN_REVIEW |

## Student runs (formal)

| Exp | Category | warm med (s) | step med (s) | fresh (s) | TE (s) | cold (s) | VRAM peak (MB) | util% p50 | attn | n | quality |
|---|---|---|---|---|---|---|---|---|---|---|---|
| B0 | baseline | 55.099 | 1.34 | 56.108 | 0.572 | 95.583 | 37709.0 | 100.0 | pytorch-sdpa-default | 9 | REFERENCE |
| BEST | final | 9.024 | 0.922 | 10.026 | 0.557 | 25.576 | 31026.8 | 100.0 | pytorch-sdpa-default | 9 | RECORDED_OR_PENDING_HUMAN_REVIEW |
| F1 | faster-computation | 43.375 | 1.049 | None | None | 80.211 | 37710.5 | 100.0 | ck | 9 | RECORDED_OR_PENDING_HUMAN_REVIEW |
| F2 | faster-computation | 50.285 | 1.222 | None | None | 72.149 | 30925.0 | 100.0 | pytorch-sdpa-default | 9 | RECORDED_OR_PENDING_HUMAN_REVIEW |
| L1 | less-computation | 12.119 | 1.338 | 13.03 | 0.548 | 49.765 | 37708.5 | 100.0 | pytorch-sdpa-default | 9 | RECORDED_OR_PENDING_HUMAN_REVIEW |

