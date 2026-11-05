# PR #2 — Phase P5: Human Image-Quality Review Package (PENDING)

**Status: `PENDING_HUMAN_REVIEW` — no human semantic-quality verdict has been performed.**
Automated metrics below are pixel/statistical distances only; they cannot approve or reject
semantic quality. Nothing in this file may be read as a human verdict.

## 1. What the owner should review (all frozen, seed-42, turbo BF16 DiT 8-step 1024² cfg=1 unless noted)

**Fast path — committed review sheets (`docs/release-closure/review-sheets/`, built by
`scripts/release-closure/build_review_sheets.py`, full-res sources pinned by SHA256 in
`review-sheets/INDEX.json`):**

- `campaign_p1-zh-scene_G_vs_C1b.jpg`, `campaign_p2-en-scene_G_vs_C1b.jpg`,
  `campaign_p3-poster-text_G_vs_C1b.jpg` — side-by-side G_native vs C1b deployment.
- `crop_poster_title_G_native.jpg` / `crop_poster_title_C1b.jpg` — 100% zoom of the p3
  Chinese poster title block (typography fidelity).
- `warmsteady_req1..4_A_vs_C.jpg` — audit warm balanced-session A vs C (steady-state
  equivalence check; full-res originals stay on the audit host, hash-pinned).

**Full-resolution originals:**

| Prompt | Reference (committed in git) | C1b deployment (committed in git) |
|---|---|---|
| p1 中文场景 | `runs/m51-20261105-cleanroom-01/e2e/raw/G_native__p1-zh-scene__cfg1.png` | `…/C1b_OV_BF16__p1-zh-scene__cfg1.png` |
| p2 English scene | `…/G_native__p2-en-scene__cfg1.png` | `…/C1b_OV_BF16__p2-en-scene__cfg1.png` |
| p3 中文海报排版 | `…/G_native__p3-poster-text__cfg1.png` | `…/C1b_OV_BF16__p3-poster-text__cfg1.png` |

Also committed: C1a FP32 and C2 INT8 full-res pairs in the same directory, plus
side-by-side grids `runs/m51-20261105-cleanroom-01/e2e/grids/*.jpg`.

- Chinese poster typography crop (worst-case text rendering):
  `runs/m51-20261105-cleanroom-01/e2e/typography/p3_poster_zh_native_crop.jpg` — request an
  equivalent C1b crop during review if desired (raw images are in `e2e/raw/`).
- Worst-case differences (automated): C2 p1 (PSNR 27.37 dB, LPIPS 0.0378) and every arm's
  empty-negative behavior under cfg=2 (` *__cfg2.png` for G/C0/C1b/C2 on p1).
- CFG>1 examples (conditioning error amplification): `e2e/raw/*__p1-zh-scene__cfg2.png`
  for G_native, C0_precomputed, C1b_OV_BF16, C2_OV_INT8.

## 2. Automated metrics (independently recomputed in phase P1; not a verdict)

| Arm vs G_native (cfg=1) | PSNR (dB) | SSIM | LPIPS(Alex) |
|---|---|---|---|
| C1b deployment BF16 | 35.57 – 47.03 | 0.9876 – 0.9972 | 0.0019 – 0.0080 |
| C1a FP32 | comparable to C1b | — | — |
| C2 INT8 | **27.37 – 38.77** | 0.9483 – 0.9941 | up to **0.0378** |
| C0 (PyTorch CPU) precomputed | — | — | p1 LPIPS 0.0090 (n=1) |

Conditioning-tensor distances: see `PR2_EVIDENCE_REDERIVATION.md` §2 (C1b deployment max
1.91% relL2 on the empty negative; INT8 28.25% — the declared FAIL).

## 3. Decisions requested from the owner

1. **C1b deployment acceptability**: are C1b images visually equivalent to G_native for the
   three course prompts, including Chinese poster typography? (gate scope: M5.1 default arm)
2. **C2 exploratory scope**: confirm C2 stays non-default (its negative-conditioning gate
   already FAILs numerically — the human check is about whether even its positives are
   acceptable for demonstrations).
3. **Course-wide BEST configuration** (6.12× Turbo/INT8 from M6) is a SEPARATE, earlier
   quality gate (`runs/quality/BEST/quality.md`); its review status must not be inferred from
   this M5.1 package or vice versa.

## 4. How to record the verdict (no automation may set it)

Write a JSON record with an explicit reviewer, then re-run any H3 experiment to observe the
propagated gate (see `lab/m51_contract.py::settle_human_quality`):

```json
{"reviewer": "<owner>", "verdict": "PASS | FAIL", "date": "…", "scope": "M5.1 C1b vs G_native"}
```

Until such a record exists, every generated result honestly reports
`human_quality: PENDING` and `release_status: PENDING_HUMAN_REVIEW`.
