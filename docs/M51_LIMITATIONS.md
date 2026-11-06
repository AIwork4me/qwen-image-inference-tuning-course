# M5.1 Limitations and Honest Boundaries

## Status vocabulary

- **PASS** — measured on this host and linked to an artifact.
- **FAIL** — measured and did not meet the declared gate.
- **BLOCKED** — not executable in the current environment.
- **NOT_RUN** — not attempted, with a reason and next action.
- **PENDING** — waiting for an external human step.

## Measured PASS

- Clean-room host audit, admission `GO_WITH_DEVIATIONS`.
- SHA256-pinned text encoder, turbo BF16 DiT, turbo INT8 DiT and VAE.
- M5.0 GPU/CPU PyTorch baseline; CPU timings reproduce historical M5.0 within 1%.
- Exact token IDs, output shape and pre-final-RMSNorm extraction contract.
- C1a FP32 conditioning gate: rel L2 3–5×10⁻⁶.
- C1b deployment BF16 gate: ≤1.25% positive, 1.91% negative.
- C1b/C2 conditioning tensor repeatability.
- OpenVINO CPU execution proof (`EXECUTION_DEVICES=["CPU"]`, CPU-only worker).
- W7900 image generation with OpenVINO-produced conditioning.
- Measured E2E latency, full-pipeline VRAM peak, residency, and four-request overlap.
- CPU-only test suite: **53 passed**.
- Original `evidence/` campaign left unchanged by the new runs (verify with `git diff --stat`).

## Measured FAIL

1. **C2 INT8 negative-conditioning gate.** Positive prompts rel L2 5.6–6.4% (pass), but the 8-token empty negative is **28.25%** (cos 0.9608), above the declared 8% gate. C2 is exploratory only.
2. **C1b-uncompressed artifact negative path.** FP32-stored IR + BF16 execution passes positive prompts but has **6.63%** negative error, above the 2% gate. The FP16-stored deployment artifact is the one that passes (1.91%); this is a precision/artifact distinction, not the same arm.

## BLOCKED

None for the required M5.1 gates on this host.

## NOT_RUN / PENDING

| Item | Status | Reason / next action |
|---|---|---|
| Human image-quality review | **PENDING_HUMAN_REVIEW** | No authorized reviewer. Use `e2e/grids/` and `e2e/typography/`; do not interpret PSNR/SSIM/LPIPS as semantic quality. |
| 40-step BF16 DiT + CFG sweep | **NOT_RUN** | The course H2/M5.1 E2E protocol uses turbo 8-step. A 40-step BF16 run costs ~16 min/image and changes the VRAM accounting. Next: add the non-turbo BF16 DiT and run a paired subset. |
| Full H1/H2 historical runner rerun | **NOT_RUN** | Replaced by a bounded, instrumented M5.1 baseline and pipeline that preserve the same semantics. The original H1/H2 evidence is frozen. |
| Multi-host / multiple AMD CPU generations | **NOT_RUN** | One EPYC 9334 host. No universal AMD speedup claim. |
| Explicit CPU affinity/NUMA pinning | **NOT_RUN** | cgroup quota 16; no `numactl` pinning. Both arms share the same constraint. |
| INT8 embedding-table quantization | **NOT_RUN** | Our C2 keeps BF16→FP32 embeddings; upstream's more aggressive int8 embedding variant is not reproduced. |
| 526-token production quality | **NOT_RUN** | Long-sequence stress is a compatibility/precision characterization only. |

## Boundary conditions

- **Turbo 8-step, cfg 1.0, 1024².** Negative conditioning is computed and aligned, but not consumed by the denoiser at cfg 1. The cfg=2 p1 probe is exploratory.
- **C0 native CPU TE:** only one prompt was run inside the server (238.14 s TE, 264.49 s wall). p2/p3 C0 images use conditioning from the isolated CPU probe. The p1 native and precomputed C0 images differ (LPIPS 0.0090 vs 0.0131); this is attributed to in-server threading/execution context, so do not treat them as bit-identical.
- **BF16 terminology:** C1b stores weights in an FP16-compressed IR and executes with `INFERENCE_PRECISION_HINT=bf16`. The “BF16” label refers to runtime precision, not BF16 weight storage. The FP32-stored IR + BF16 hint is reported separately.
- **GPU product identity:** the driver reports `AMD Radeon Graphics`, gfx1100, 48 GiB. It is W7900-class, not a literal `Radeon PRO W7900` string.
- **cgroup:** 16 CPU / 55 GiB, not the full 128-thread / 503 GiB host. Do not extrapolate absolute numbers to a full EPYC.
- **FP32 arm memory:** 50.5 GB RSS makes C1a unsuitable for co-residency under a 55 GiB cap; it is an alignment reference, not the deployment candidate.
- **First-request reload:** ComfyUI unloads models under memory pressure; arm A's first request after idle took 26.07 s vs 13.03 s steady. Pipeline totals include cold behavior; steady-state estimates are reported separately.
- **Metrics are differences:** PSNR/SSIM/LPIPS compare two valid images. They are not a quality PASS.
- **Noise:** C0 has n=1 warm sample per prompt. The two identical encodes differed by <0.3%, but the formal uncertainty is larger than C1's n=2. No p95 is claimed for C0.
- **Network:** git-over-proxy returned HTTP 503; the pinned ComfyUI tree was staged from codeload. Model downloads used hf-mirror. Reproduction may need retries.
- **Large artifacts:** the FP32/BF16/INT8 IRs are on `/opt/m51-artifacts` and symlinked into the run; they are not in git. Regenerate with `scripts/m51/provision_ov.sh`.
