# Linux W7900 lab — results summary (auto-generated)

Generated: 2026-11-03T15:45:01+00:00  ·  records: 146

| Exp | Category | warm med (s) | step med (s) | fresh (s) | TE (s) | cold (s) | VRAM peak (MB) | util% p50 | attn | n | quality |
|---|---|---|---|---|---|---|---|---|---|---|---|
| B0 | baseline | 55.22 | 1.345 | 56.105 | 0.555 | 74.171 | 37686.8 | 100.0 | pytorch-sdpa-default | 21 | REFERENCE |
| BEST | final | 9.023 | 0.915 | 10.025 | 0.549 | 27.586 | 31026.8 | 100.0 | pytorch-sdpa-default | 9 | PENDING_HUMAN_REVIEW |
| F1 | faster-computation | 45.086 | 1.077 | None | None | 84.577 | 37688.6 | 100.0 | ck | 9 | PENDING_HUMAN_REVIEW |
| F2 | faster-computation | 51.093 | 1.22 | None | None | 80.257 | 31024.7 | 100.0 | pytorch-sdpa-default | 9 | PENDING_HUMAN_REVIEW |
| L1 | less-computation | 12.132 | 1.337 | 13.029 | 0.553 | 48.439 | 37686.6 | 100.0 | pytorch-sdpa-default | 9 | PENDING_HUMAN_REVIEW |
| L1O | less-computation | 12.074 | 1.337 | None | None | 50.324 | 37715.3 | 100.0 | pytorch-sdpa-default | 9 | PENDING_HUMAN_REVIEW |
| LF1 | combined | 10.027 | 1.067 | None | None | 53.666 | 37710.5 | 100.0 | ck | 9 | PENDING_HUMAN_REVIEW |
| LF2 | combined | 9.023 | 0.916 | None | None | 32.431 | 31026.8 | 100.0 | pytorch-sdpa-default | 9 | PENDING_HUMAN_REVIEW |
| LF3 | combined | 9.022 | 0.906 | None | None | 30.414 | 31026.6 | 100.0 | ck | 9 | PENDING_HUMAN_REVIEW |
