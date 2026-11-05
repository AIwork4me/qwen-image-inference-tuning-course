# M5.1 Claim → Evidence Matrix

Every row is a claim made by this work. Historical upstream numbers are labelled; no current-machine claim depends on them.

| # | Claim | Evidence | Status |
|---:|---|---|---|
| 1 | The host has one working gfx1100 AMD GPU with ~48 GiB VRAM and ROCm PyTorch 2.9.1 | `environment.json`; `rocminfo`; `rocm-smi` | PASS |
| 2 | The course bootstrapped from the pinned ComfyUI 0.43.0 tree and SHA256-pinned weights | `environment.json`; `runs/doctor.json`; `lab/weights.py` | PASS |
| 3 | The M5.0 CPU baseline reproduces the historical 177.8/136.8/207.1 s within 1% | `baseline/probe_cpu.json` (176.0/136.8/207.4 s) | PASS |
| 4 | GPU and CPU PyTorch produce the same conditioning (token IDs exact; rel L2 ~7e−6) | `alignment/conditioning_alignment.json` | PASS |
| 5 | The OpenVINO FP32 backend aligns with the PyTorch CPU path | rel L2 3–5e−6, cos 1.0 | PASS |
| 6 | BF16 deployment (FP16-stored IR, BF16 execution) preserves conditioning | rel L2 1.09–1.25% positive, 1.91% negative | PASS |
| 7 | BF16 on the FP32-stored IR fails the negative gate | 6.63% on the 8-token empty negative | FAIL |
| 8 | INT8 positive prompts are within the declared envelope | 5.6–6.4%, cos ≥0.9979 | PASS |
| 9 | INT8 fails the negative-conditioning gate | 28.25%, cos 0.9608 | FAIL |
| 10 | OpenVINO CPU is 155–549× faster across the measured C1/C2 arms (C1b 328–480×) than the course PyTorch CPU path on the same CPU | `performance/encoder_ab.json` | PASS |
| 11 | The OpenVINO worker executes on CPU only | `EXECUTION_DEVICES=["CPU"]` in all probe JSONs | PASS |
| 12 | Offloading the GPU TE reduces full-pipeline VRAM peak, not just allocation | 34.15 → 16.77 GB peak; 16.80 GB residency saved | PASS |
| 13 | Generated images with OpenVINO conditioning are valid W7900 images | `e2e/raw/` (local), grids, typography crop | PASS |
| 14 | Image differences are measurable and BF16 is closer than INT8 | PSNR/SSIM/LPIPS table, `e2e_summary.json` | PASS |
| 15 | Faster TE gives only a small steady-state E2E gain | 13.03 → 12.03 s precomputed; ≈12.58 s serialized | PASS |
| 16 | Four-request CPU/GPU overlap is real | 37.09 s (B), 1.62 s (C), 1.40 s (D) overlap from timestamps | PASS |
| 17 | The PyTorch CPU pipeline is CPU-bound | 680.0 s; 147 s encode vs 12 s denoise; GPU idle 454.6 s | PASS |
| 18 | OpenVINO pipeline is GPU-bound | C 12.19 s denoise, 0.44 s encode, 0.0 s wait | PASS |
| 19 | No measurable GPU slowdown under 16-thread CPU encoding | B/C/D 12.07–12.19 s vs A 13.03 s (incl. TE) | PASS |
| 20 | Clean IRs run long sequences without the historical shape error | all arms 180–526 kept tokens; no patch | PASS |
| 21 | INT8's long-sequence error grows materially | 7.5–18.4% vs FP32 | PASS (descriptive) |
| 22 | The CPU test suite passes without a GPU | 53 passed | PASS |
| 23 | Original course evidence is unchanged | `git diff --stat` shows no `evidence/` modifications | PASS |
| 24 | Human image-quality review | none performed | PENDING_HUMAN_REVIEW |
| 25 | Universal AMD CPU speedup | not tested | NOT_RUN |
| 26 | 40-step BF16/CFG quality sweep | not run | NOT_RUN |
| 27 | INT8 is the production recommendation | contradicted by negative-path 28% | FAIL / not recommended |

## Historical numbers deliberately not reused as local evidence

- Zen5 Ryzen AI MAX+ PRO 395: OpenVINO INT8 0.187/0.377/0.687 s vs ComfyUI INT8 ConvRot 12.98/13.73/15.43 s (22–69×). Different CPU/GPU/weights.
- Zen4 EPYC 9334 reference: OV BF16 0.39–1.33 s, INT8 0.22–1.08 s, FP32 rel L2 7.7e−6. Different artifact lineage and thread budget.
- The current local C0 is ComfyUI PyTorch **FP32 manual-cast** compute, not a generic BF16 CPU GEMM baseline.
