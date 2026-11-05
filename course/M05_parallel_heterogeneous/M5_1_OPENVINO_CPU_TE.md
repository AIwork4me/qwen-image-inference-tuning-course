# M5.1 — Make CPU Text Encoding Fast with OpenVINO

**M5.0 baseline:** [PyTorch CPU offloading & 4-request pipelining](LESSON.md)
**Experiment status:** **COMPLETED on a physical gfx1100 48 GiB W7900-class GPU, 2026-11-05.**
**Decision:** **GO** for OpenVINO BF16 CPU deployment; **INT8 is exploratory** (negative-conditioning gate fails).
**Full evidence:** [M51_RESULTS.md](../../docs/M51_RESULTS.md) · run `runs/m51-20261105-cleanroom-01/`.

**中文学习目标：** M5.0 中，把 Qwen3-VL-8B Text Encoder 搬到 CPU 可以释放约 16.8 GB GPU 显存，却让 PyTorch CPU 编码耗费 136.8–207.1 秒、四请求吞吐下降约 14 倍。M5.1 在同一块 EPYC 9334、同一批提示词、同一张 gfx1100 48 GB GPU、同一条 DiT/VAE 路径上，用 OpenVINO 重做 CPU 编码器。结论不是“端到端快 300 倍”，而是：**编码器快 200–500 倍，端到端只快约 3–8%，但显存节省是真实的 17 GB。**

---

## 1. Two real upstream studies (historical, not local measurements)

| Study | Tested hardware and result | What is reusable — and what isn't |
|---|---|---|
| [qwen3vl-cpu-benchmark](https://github.com/AIwork4me/qwen3vl-cpu-benchmark) | Ryzen AI MAX+ PRO 395, Zen5: OpenVINO **INT8** conditioning **0.187 / 0.377 / 0.687 s** vs ComfyUI **INT8 ConvRot PyTorch CPU product path** **12.98 / 13.73 / 15.43 s**, i.e. **22–69× encoder-only** | Reuse the [conditioning bridge](https://github.com/AIwork4me/qwen3vl-cpu-benchmark/blob/main/scripts/ov_encoder.py) and [custom node](https://github.com/AIwork4me/qwen3vl-cpu-benchmark/blob/main/custom_nodes/qwen3vl_openvino_cpu/__init__.py) as references; do **not** apply the 22–69× ratio to this course's **PyTorch FP32-manual-cast** M5.0 |
| [qwen-image21-openvino-alignment](https://github.com/AIwork4me/qwen-image21-openvino-alignment) | EPYC 9334 Zen4 + W7900D gfx1100: OV CPU BF16 **0.39–1.33 s**, INT8 **0.22–1.08 s** on 25–171-token workloads; clean OV FP32 vs PyTorch FP32 conditioning rel L2 **7.7e−6** | Reuse the [numeric alignment method](https://github.com/AIwork4me/qwen-image21-openvino-alignment/blob/main/CONCLUSIONS.md), [reproduction guide](https://github.com/AIwork4me/qwen-image21-openvino-alignment/blob/main/reports/REPRODUCTION.md), [limitations](https://github.com/AIwork4me/qwen-image21-openvino-alignment/blob/main/reports/LIMITATIONS.md); its artifact/ROCm stack differs |

**Important:** the Zen5 result is an **INT8-vs-INT8** backend comparison; this course's C0 is ComfyUI PyTorch **FP32 manual-cast compute** with BF16-stored weights. The local M5.1 run does not reuse those historical ratios as its own.

---

## 2. The controlled experiment

| Arm | Text Encoder | Hardware | Main comparison |
|---|---|---|---|
| **G — reference** | ComfyUI / PyTorch | GPU | GPU encoding plus GPU DiT/VAE |
| **C0 — M5.0 baseline** | ComfyUI / PyTorch (BF16 storage, FP32 manual cast) | **CPU** | Frozen CPU PyTorch reference |
| **C1a — alignment** | OpenVINO **FP32** | **same CPU** | Backend + high-precision reference |
| **C1b — main candidate** | OpenVINO **BF16 execution** on FP16-stored deployment IR | **same CPU** | **Primary C1 vs C0 comparison** |
| **C2 — optional** | OpenVINO **INT8 weight compression** | same CPU | Quantization A/B, separately gated |

Controls: frozen prompts and token IDs, same TE checkpoint lineage, 16-thread cgroup budget for both CPU arms, same DiT (turbo BF16), VAE, scheduler, 8 steps, cfg 1.0, seed 42, 1024².

### Conditioning contract

```text
ComfyUI QwenImage21Tokenizer (T2I system prefix, thinking=True)
        |
        +--> C0 PyTorch CPU (FP32 manual cast)
        +--> C1a/C1b/C2 OpenVINO CPU
        |
   layer-36 residual BEFORE final RMSNorm
   trim at the second <|im_start|>
   [1, seq_kept, 4096] FP32
        |
   LinuxLabCondSafetensors (safe, no pickle)
        |
   original W7900 GPU DiT -> original VAE -> PNG
```

No logits, no post-RMSNorm features, no generic chat template, no fake tensor.

---

## 3. Measured local results (2026-11-05)

### 3.1 Same-CPU encoder A/B

| Arm | p1 66 tok | p2 51 tok | p3 78 tok | empty neg | cold→first | CPU RSS |
|---|---:|---:|---:|---:|---:|---:|
| C0 PyTorch CPU | **176.02 s** | **136.81 s** | **207.45 s** | **61.51 s** | 179.4 s | 34.5 GB |
| C1a OV FP32 | 0.779 s | 0.648 s | 0.875 s | 0.398 s | 95.0 s | 50.5 GB |
| **C1b OV BF16** | **0.532 s** | **0.285 s** | **0.555 s** | **0.187 s** | **5.5 s** | **30.0 GB** |
| C2 OV INT8 | 0.361 s | 0.249 s | 0.406 s | 0.144 s | 7.2 s | 16.8 GB |

Speedup vs C0: **C1a 226/211/237/155×**, **C1b 331/480/374/328×**, **C2 488/549/511/427×** (p1/p2/p3/negative). The C0 numbers reproduce the historical M5.0 CPU evidence (177.8/136.8/207.1 s) within 1%.

### 3.2 Numerical gate

| Comparison | Positive rel L2 | Negative rel L2 | Status |
|---|---:|---:|---|
| C1a OV FP32 vs C0 | **3–5×10⁻⁶** | 5.4×10⁻⁶ | **PASS** |
| **C1b OV BF16 vs C0** | **1.09–1.25%** | **1.91%** | **PASS** |
| C2 OV INT8 vs C1a | **5.6–6.4%** | **28.25%** | positive PASS, **negative FAIL** |

Token IDs are exactly equal; shapes are `[1,52/37/64/8,4096]`; all tensors finite; repeated hashes identical within each arm.

### 3.3 Real images on the W7900 (turbo 8-step, 1024²)

| Arm | Image latency | PSNR vs G | SSIM vs G | LPIPS vs G | VRAM peak |
|---|---:|---:|---:|---:|---:|
| G native GPU TE | 13.03 s | — | — | — | 34.15 GB |
| C1a OV FP32 | 12.03 s* | 34.74–48.62 dB | 0.9885–0.9974 | 0.0015–0.0106 | 16.95 GB |
| **C1b OV BF16** | **12.03 s*** | **35.57–47.03 dB** | **0.9876–0.9972** | **0.0019–0.0080** | **16.77 GB** |
| C2 OV INT8 | 12.04 s* | 27.37–38.77 dB | 0.9483–0.9941 | 0.0081–0.0378 | 16.92 GB |
| C0 native CPU TE | **264.49 s** | 34.97 dB | 0.9887 | 0.0090 | 33.10 GB peak |

\* Precomputed conditioning; the encoder is outside the image wall. Serialized C1b fresh-prompt ≈ **12.58 s**.

**VRAM:** full-pipeline peak **34.15 → 16.77 GB (17.26 GB saved)**; residency **33.09 → 16.29 GB (16.80 GB saved)**. This is the measured full-pipeline effect for this configuration, not an assumption from the 16.8 GB TE allocation.

Image grids: `runs/m51-20261105-cleanroom-01/e2e/grids/`; native-resolution Chinese poster crop: `e2e/typography/p3_poster_zh_native_crop.jpg`. **Human review is PENDING.**

### 3.4 Four-request pipeline

| Arm | Total | Fill | CPU TE | GPU denoise | Images/min | Bound |
|---|---:|---:|---:|---:|---:|---|
| A GPU TE sequential | 66.62 s | — | — | 13.03 s | 3.60 (steady 4.60 est.) | GPU |
| B PyTorch CPU TE | **680.00 s** | 176.17 s | **147.14 s** | 12.07 s | **0.35** | **CPU** |
| **C OV BF16** | **50.05 s** | 0.56 s | 0.436 s | 12.19 s | **4.80** | GPU |
| **D OV INT8** | **49.75 s** | 0.40 s | 0.405 s | 12.13 s | **4.82** | GPU |

C vs A: **1.33× cold-inclusive**, **1.04× steady-state**. B is **0.098×** (10× slower). Overlap is proven from timestamps: B 37.09 s, C 1.62 s, D 1.40 s. No measurable GPU slowdown under 16-thread CPU load.

### 3.5 Why the end-to-end gain is small (Amdahl)

```text
Sequential cycle  = GPU_TE + GPU_denoise ≈ 0.60 + 12.43 = 13.03 s
Pipelined cycle   = max(OV_TE, GPU_denoise_under_load) + handoff
                  ≈ max(0.44, 12.19) ≈ 12.19 s
Encoder-only      = 176.02 / 0.532 = 331×
E2E steady gain   = 13.03 / 12.19 ≈ 1.07×
```

The encoder is ~4% of the image cycle. Once it is hidden, making it 300× faster cannot break the denoiser bound. The large cold-inclusive 1.33× comes mostly from **avoiding a 16.8 GB GPU CLIP reload**, not from overlap.

---

## 4. Implementation status

The lab now provides:

```bash
# Optional provisioning (NOT part of the default M0-M4 bootstrap)
bash scripts/m51/provision_ov.sh

# Bounded smoke and formal runs
.venv-lab/bin/python -m lab run --experiment H3_OV_BF16 --smoke
.venv-lab/bin/python -m lab run --experiment H3_OV_BF16
.venv-lab/bin/python -m lab run --experiment H3_OV_INT8 --smoke
```

The CLI checks the isolated `.venv-ov`, artifact manifests, disk/RAM budget and writes explicit `PASS`/`FAIL`/`BLOCKED` records. If the optional artifacts are absent, it prints the provisioning command and refuses to fake a run.

Implementation files: `lab/ov_export.py`, `lab/ov_compress.py`, `lab/ov_quantize.py`, `lab/ov_encoder.py`, `lab/ov_worker.py`, `lab/pt_worker.py`, `lab/custom_nodes/linuxlab_cond_safetensors.py`, `lab/m51_*.py`, `scripts/m51/provision_ov.sh`, `tests/test_m51_contract.py`.

---

## 5. Student exercise (beginner-friendly, reproducible)

**Question:** how much of a 300× encoder speedup survives to the image pipeline?

1. Read the measured tables above. Compute the encoder share of the sequential cycle: `TE / (TE + KSampler + VAE)`.
2. Use the pipeline JSON to verify that C's CPU encode interval lies inside the GPU denoise interval.
3. Compute the break-even: for what denoise time would a 0.53 s encoder be 50% of the cycle? *(Answer: ~0.53 s; this GPU denoise is ~12 s.)*
4. Compare C1b's 1.25% conditioning difference with C0-native's image-difference metrics. Why does a 7e−6 FP32 conditioning delta already produce LPIPS ~0.01? *(Diffusion amplifies small perturbations; image-difference metrics are not quality verdicts.)*
5. Explain why INT8's 28% negative-conditioning error matters at cfg>1 but not at cfg=1. *(At cfg=1 the negative branch is not consumed; at cfg>1 it is.)*

**Acceptance:** the student can explain the conditioning bridge, compute a fair ratio, reproduce the break-even, and refuse to call the 331× encoder gain a 331× image-generation gain.

---

## 6. Think — with measured answers

1. **Why does the Zen5 22–69× not answer this course's C0→C1 question?** It compares different CPUs, weights, thread budgets and an INT8-vs-INT8 path. Local C1b vs C0 is 331–480×, but against the course's FP32-manual-cast CPU path.
2. **If GPU denoise is 10.8 s, GPU TE 0.6 s, OV CPU TE 0.44 s and handoff ~0.2 s, what is the cycle speedup?** `(10.8+0.6)/(max(0.44,10.8)+0.2) ≈ 11.4/11.0 ≈ 1.04×`, not 20×. Measured: 1.04–1.07× steady.
3. **Why is `cosine ≥ 0.997` not automatic typography PASS?** Diffusion can amplify a small conditioning difference; inspect native-resolution crops and human review. Our p3 crop is evidence, not a PASS.
4. **BF16-vs-BF16 backend or BF16-vs-INT8?** C1b vs C0 changes backend plus BF16 execution; C2 vs C1a changes quantization. They are separate causal axes.

---

## 7. Limitations students must quote

- Turbo 8-step, cfg 1.0, one seed suite; human review PENDING.
- C2 INT8 fails the empty-negative gate (28.25%); do not call it production.
- C0 has n=1 warm sample per prompt; the two identical encodes differ <0.3%.
- The host is a 16-CPU/55 GiB cgroup of an EPYC 9334, not the full host.
- The GPU driver string is `AMD Radeon Graphics` gfx1100 48 GiB (W7900-class, not a literal product string).
- BF16 in C1b means **execution precision**, not BF16 weight storage.
- Do not quote the 22–69× Zen5 figure as a local result.
