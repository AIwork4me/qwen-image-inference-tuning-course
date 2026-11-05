# M5.1 End-to-End and Four-Request Pipeline

**Artifacts:** `runs/m51-20261105-cleanroom-01/e2e/e2e_summary.json`, `e2e/e2e_ab.json`, `pipeline/pipeline_summary.json`, `pipeline/pipeline_ab.json`.
**Figure:** `docs/figures/m51_pipeline_timeline.png`.

## 1. E2E protocol

- DiT: `qwen_image_2.1_turbo_bf16.safetensors` (unchanged), VAE: original BF16, 1024×1024, 8 steps, euler/simple, cfg 1.0, seed 42, same empty negative.
- Arm G: ComfyUI GPU `CLIPLoader` + `CLIPTextEncode` + GPU DiT/VAE.
- Arm C0 native: `CLIPLoader(device=cpu)` + GPU DiT/VAE (one prompt; expensive).
- C1a/C1b/C2 and C0-precomputed: conditioning precomputed by the corresponding encoder, injected through the **safe safetensors node**; no CLIP model is instantiated in the GPU process.
- G-precomputed is a bridge check: GPU-derived conditioning injected through the same node. It produced **pixel-identical** images to native G (PSNR ∞), proving the injection path is lossless.

## 2. Warm latency and memory

| Arm | Wall | TE | KSampler | VAE | Images/min | VRAM peak | VRAM residency |
|---|---:|---:|---:|---:|---:|---:|---:|
| **G native** | **13.03 s** | 0.60 s | 10.80 s | 0.88 s | 4.60 | **34.15 GB** | **33.09 GB** |
| G precomputed | 12.26 s | 0 | 10.97 s | 0.89 s | 4.89 | 16.88 GB | 16.28 GB |
| C1a OV FP32 | 12.03 s | 0* | 10.72 s | 0.88 s | 4.99 | 16.95 GB | 16.29 GB |
| **C1b OV BF16** | **12.03 s** | 0* | 10.73 s | 0.88 s | 4.99 | **16.77 GB** | **16.29 GB** |
| C2 OV INT8 | 12.04 s | 0* | 10.74 s | 0.88 s | 4.98 | 16.92 GB | 16.28 GB |
| C0 native | **264.49 s** | **238.14 s** | 11.07 s | 0.88 s | 0.23 | 33.10 GB | 16.36 GB |

\* Encoder time is outside the 12 s wall because conditioning is precomputed; add **0.19–0.56 s** for a serialized fresh prompt. Warm C1b serialized ≈ **12.58 s**.

**GPU memory saving:** full-pipeline peak **34.15 → 16.77 GB = 17.26 GB saved**; residency **33.09 → 16.29 GB = 16.80 GB saved**. This is measured for turbo 8-step 1024². It is **not** a claim that every 40-step configuration saves 16.8 GB of full-pipeline peak; a larger DiT peak changes the accounting.

## 3. Image-difference metrics vs native G

| Arm | PSNR median / worst | SSIM median / worst | LPIPS median / worst |
|---|---:|---:|---:|
| C1a OV FP32 | 34.87 / 34.75 dB | 0.99484 / 0.98847 | 0.00275 / 0.01057 |
| **C1b OV BF16** | **36.89 / 35.57 dB** | **0.99164 / 0.98757** | **0.00450 / 0.00795** |
| C2 OV INT8 | 29.96 / 27.37 dB | 0.97870 / 0.94832 | 0.01813 / 0.03779 |
| C0 native (p1) | 34.97 dB | 0.98874 | 0.00903 |
| C0 precomputed | 39.40 / 33.02 dB | 0.99343 / 0.98220 | 0.00346 / 0.01312 |

These metrics measure **image difference**, not semantic correctness or aesthetic quality. C1a and C0-native produce comparable differences from G; the tiny FP32 conditioning deltas are amplified by the diffusion trajectory. C2's worst case is visibly larger. **Human verdict: `PENDING_HUMAN_REVIEW`.**

Grids: `e2e/grids/{p1,p2,p3}_cfg1.jpg`; native-resolution Chinese typography crop: `e2e/typography/p3_poster_zh_native_crop.jpg`.

## 4. Four-request pipeline: measured timestamps

Arms: A GPU TE sequential; B persistent PyTorch CPU TE; C persistent OV BF16; D persistent OV INT8. Four distinct frozen prompts; each next conditioning is computed while the current image denoises.

| Arm | Total | Fill | CPU TE median | GPU denoise median | GPU idle/wait | Overlap proven | Images/min |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | 66.62 s | — | — | 13.03 s | — | — | 3.60 (steady est. 4.60) |
| B | **680.00 s** | 176.17 s | **147.14 s** | 12.07 s | **454.64 s** | 37.09 s | **0.35** |
| C | **50.05 s** | 0.56 s | 0.436 s | 12.19 s | 0 | 1.62 s | **4.80** |
| D | **49.75 s** | 0.40 s | 0.405 s | 12.13 s | 0 | 1.40 s | **4.82** |

A's first request after idle reloaded the GPU CLIP and took **26.07 s**; subsequent requests were **13.03 s**. Steady-state 4-request estimate = **52.13 s**, not 66.62 s. C/D do not reload a GPU CLIP because no CLIP is instantiated.

## 5. Break-even and Amdahl

```text
Sequential GPU cycle       = GPU_TE + GPU_denoise ≈ 0.60 + 12.43 = 13.03 s
Pipelined OV BF16 cycle    = max(OV_TE, GPU_denoise_under_load) + handoff
                           ≈ max(0.44, 12.19) + ~0.0 = 12.19 s
Measured steady gain       = 13.03 / 12.19 = 1.07× (pipeline C/A estimate 1.04×)
Encoder-only gain          = 176.02 / 0.532 = 331×
```

The encoder is ~4% of the sequential cycle. Even an infinitely fast encoder can only remove that 4% once denoising dominates. The large cold-inclusive 1.33× gain is mostly avoiding the 16.8 GB GPU CLIP reload, not the overlap itself.

**Contention:** B/C/D image generation stays at **12.07–12.19 s**, within ~1 s of A's 13.03 s (which includes 0.60 s TE). No measurable GPU slowdown from 16-thread CPU encoding was found. B is CPU-bound (GPU idle 124–196 s per request); C/D are GPU-bound.

## 6. Deployment recommendation

For this workload, use **C1b OpenVINO BF16 CPU + a persistent worker + a bounded request queue** when:
- the GPU must retain maximum headroom for a larger/40-step DiT or concurrent jobs;
- cold-start GPU CLIP load is expensive;
- BF16 conditioning fidelity (≤1.25% positive) is acceptable;
- the operator accepts a ~1.04–1.07× steady-state image-throughput gain, not a 300× E2E gain.

Do **not** deploy C2 INT8 as the default for cfg>1: its empty-negative conditioning error is 28%. It may be useful when a different valid image sample is acceptable and memory is tight.
