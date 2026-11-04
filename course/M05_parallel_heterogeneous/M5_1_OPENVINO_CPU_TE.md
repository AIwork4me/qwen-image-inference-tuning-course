# M5.1 — Make CPU Text Encoding Fast with OpenVINO

**M5.0 baseline:** [PyTorch CPU offloading & 4-request pipelining](LESSON.md)  
**Experiment status:** **PLANNED / course W7900 OpenVINO A/B NOT_RUN**. The prior work below *was* physically measured, but the same-stack course experiment has **not** been run. Do not quote a new H3 speedup before measuring it.

**中文学习目标：** M5.0 中，把 Qwen3-VL-8B Text Encoder 搬到 CPU 可以释放约 16.8 GB GPU 编码器分配量，却让 PyTorch CPU 编码耗费 136.8–207.1 秒、四请求吞吐下降约 14 倍。现在保持 **同一块 CPU、同一批提示词、同一张 W7900、同一条 DiT/VAE 路径**，用 OpenVINO 优化 CPU 编码器，检验它能否从“省显存却极慢”变成“省显存且有系统价值”的方案。不能把编码器加速误说成端到端图像生成加速。

## 1. Start from two *real* upstream studies

| Study | Tested hardware and result | What is reusable — and what isn't |
|---|---|---|
| [qwen3vl-cpu-benchmark](https://github.com/AIwork4me/qwen3vl-cpu-benchmark) | Ryzen AI MAX+ PRO 395, Zen5: OpenVINO **INT8** conditioning **0.187 / 0.377 / 0.687 s** vs ComfyUI **INT8 ConvRot PyTorch CPU product path** **12.98 / 13.73 / 15.43 s**, i.e. **22–69× encoder-only** | Reuse [conditioning bridge](https://github.com/AIwork4me/qwen3vl-cpu-benchmark/blob/main/scripts/ov_encoder.py), [custom node](https://github.com/AIwork4me/qwen3vl-cpu-benchmark/blob/main/custom_nodes/qwen3vl_openvino_cpu/__init__.py) as reference; do **not** apply the 22–69× ratio to this course's **BF16** M5.0 |
| [qwen-image21-openvino-alignment](https://github.com/AIwork4me/qwen-image21-openvino-alignment) | EPYC 9334 Zen4 + W7900D gfx1100: OV CPU BF16 **0.39–1.33 s**, INT8 **0.22–1.08 s** on its 25–171-token workloads; clean OV FP32 vs PyTorch FP32 conditioning rel L2 **7.7e−6** | Reuse [numeric alignment method](https://github.com/AIwork4me/qwen-image21-openvino-alignment/blob/main/CONCLUSIONS.md), [reproduction guide](https://github.com/AIwork4me/qwen-image21-openvino-alignment/blob/main/reports/REPRODUCTION.md), [limitations](https://github.com/AIwork4me/qwen-image21-openvino-alignment/blob/main/reports/LIMITATIONS.md); its artifact/ROCm stack and prompt lengths differ from this course |

**Important:** the Zen5 result is an **INT8-vs-INT8** backend comparison; the Zen4 result includes FP32, BF16 and INT8; this course's M5.0 is **PyTorch BF16 CPU**. Never divide numbers from different repositories to invent a same-host speedup. The Zen5 full image pipeline improved warm latency only ~1.5–2%; its larger benefit was GPU-memory occupancy and cold start. The Zen4 study also measured a **~7% relative L2** OV-INT8 vs OV-FP32 conditioning delta; this is a tensor difference, **not** a 7% image-quality loss.

## 2. The controlled experiment

| Arm | Text Encoder | Hardware | Main comparison |
|---|---|---|---|
| **G — reference** | ComfyUI / PyTorch BF16 | GPU | GPU encoding reference, quality and E2E latency |
| **C0 — M5.0 baseline** | ComfyUI / PyTorch BF16 | **CPU** | Freeze the CPU PyTorch reference |
| **C1 — M5.1 main** | OpenVINO **FP32 clean export / BF16 CPU runtime** | **same CPU** | **Primary causal comparison C1 vs C0** |
| **C2 — optional** | OpenVINO **INT8 weight-compressed** | same CPU | New quantization A/B **C2 vs C1**, with independent quality gate |

Use the identical frozen prompts and token IDs, same checkpoint lineage, CPU/NUMA/thread budget, GPU DiT, scheduler, steps, VAE and seeds. Specify whether the BF16 variant is BF16 **storage**, runtime instruction precision or both. INT8 weight storage alone does **not** demonstrate true INT8 CPU GEMM. Audit oneDNN CPU kernels if making such a claim.

### Conditioning diagram

```text
Same ComfyUI Qwen-Image tokenizer / exact token IDs / attention masks
                                |
                  +-------------+-------------+
                  |                           |
            C0 PyTorch CPU              C1 OpenVINO CPU
            BF16 encoder                FP32/BF16 (then INT8)
                  |                           |
       final layer-36 hidden       SAME final layer-36 hidden
       PRE-final-RMSNorm           PRE-final-RMSNorm
                  +-------------+-------------+
                                |
            trim from second <|im_start|> (system prefix)
                  [1, seq_kept, 4096] conditioning
                                |
                  check shape / cos / relative L2 /
                  max abs / token IDs / negative cond
                                |
             Existing LinuxLabCondLoader boundary
                                |
           Original W7900 GPU DiT → original VAE → PNG
```

The [validated OpenVINO bridge](https://github.com/AIwork4me/qwen3vl-cpu-benchmark/blob/main/scripts/ov_encoder.py) uses ComfyUI's `QwenImage21Tokenizer`, `model.add_outputs` to expose the last residual **before final RMSNorm**, and trims the system prefix beginning at the second `<|im_start|>`. **Generic Qwen3-VL chat logits are not Qwen-Image conditioning.** The pinned ComfyUI 0.43.0 and its separate `qwen3vl_8b_bf16.safetensors` must be cross-checked against the other projects' model artifacts. Copy code only after checking author/license permission and source-commit compatibility.

## 3. Gates before any performance claim

| Gate | What to do | Required artifact |
|---|---|---|
| A. Runtime | Freeze W7900 gfx1100, CPU model/cgroup cores, memory, CPU ISA, OpenVINO version, prompt lengths, model SHA and exact source-commit; use an **isolated** OpenVINO environment, never replace ROCm PyTorch | `runs/m51-*/environment.json` |
| B. Conversion | Audit whether ComfyUI Qwen3-VL 8B checkpoint weights/tokenizer map to the chosen clean OV export; handle variable-length text and both positive/negative conditioning | `conversion_audit.json`; PASS or BLOCKED |
| C. Numerical | Compare G, C0, C1 (then C2) for identical tokens/hidden-layer extraction; shape, cosine, relative L2, per-token/max/mean abs, reproducibility and accuracy thresholds fixed in advance | `conditioning_alignment.json` |
| D. Encoding | On the SAME CPU measure C0 vs C1: model read/compile, cold encode, fresh prompt, warm medians/spread, threads sweep, RSS and GPU allocation | `encoder_ab.json`, no cross-machine ratios |
| E. Quantization | Compare C2 **against C1**: confirm actual CPU kernels, precision, latency/RAM improvement AND extra conditioning error | `int8_analysis.json`, never automatic quality PASS |
| F. End-to-end | Inject validated conditioning into the **unchanged** GPU DiT/VAE; compare same latent/seed/schedule PNGs, Chinese typography crops, LPIPS/SSIM/PSNR, human verdict | `e2e_ab.json`; human review remains PENDING |
| G. Pipelining | Four distinct fresh prompts: sequential GPU-TE vs PyTorch-CPU pipeline vs OpenVINO-BF16 pipeline (optional INT8); measure GPU/CPU concurrency, launch-thread contention, images/min, p50/p95 | `pipeline_ab.json` with exact timestamps |
| H. Report | Compute break-even, link all evidence, classify PASS / FAIL / BLOCKED / NOT_RUN; keep H1/H2 frozen | `M51_RESULTS.md` |

### Zen4 INT8 compatibility trap

The [alignment project's limitations](https://github.com/AIwork4me/qwen-image21-openvino-alignment/blob/main/reports/LIMITATIONS.md) document that an existing INT8 OpenVINO IR failed for some text inputs, particularly **more than ~40 tokens** on EPYC 9334. It also validates a [runtime-only, text-only graph patch](https://github.com/AIwork4me/qwen-image21-openvino-alignment/blob/main/scripts/patch_ov_artifact_textonly.py) for that specific artifact. **Never apply this patch to multimodal inputs or silently modify serialized customer models**. Prefer a clean FP32/BF16 export for C1, verify the actual failing sequence and artifact identity, then consider an independently verified fix for C2.

### What will count as a system-level improvement?

```text
Sequential G cycle    ≈ GPU_TE + GPU_denoise
Pipelined C1 cycle    ≈ max(OV_CPU_TE, GPU_denoise_under_CPU_load) + handoff
```

The pipeline wins throughput **only if** the second expression is smaller. A sub-second OpenVINO encoder does not imply a large image speedup when GPU denoising still takes ~9–12 s. Distinguish **TE-only speed**, **fresh-prompt E2E**, **cold start**, **GPU VRAM residency vs peak**, and **images/min**. CPU work may slow GPU kernel launches.

## 4. Student exercise and implementation status

The M5.0 CPU PyTorch baseline is runnable but slow and optional; students can use its already committed evidence:

```bash
# Existing M5.0 baseline; OPT-IN, budget the CPU/GPU time.
.venv-lab/bin/python -m lab run --experiment H1
.venv-lab/bin/python -m lab run --experiment H2
```

**Desired next implementation, NOT YET AVAILABLE in this course**:

```bash
# PROPOSED for a follow-up physical-W7900 implementation PR:
# .venv-lab/bin/python -m lab run --experiment H3_OV_BF16 --smoke
# .venv-lab/bin/python -m lab run --experiment H3_OV_INT8 --smoke
```

Do not download extra OpenVINO model files during the standard `scripts/cloud/start.sh` course bootstrap. M5.1 must plan additional disk use and CPU RAM on Radeon Cloud, avoid the known 4 GiB `/tmp` scratch exhaustion, and preserve `evidence/` byte-for-byte.

**Think:**
1. Why does the Zen5 report's 22–69× not answer this course's C0→C1 question? *(Different CPU baseline, weights and machine.)*
2. If GPU denoising takes 10 s, GPU TE 0.5 s, OV CPU TE 0.4 s and handoff 0.2 s with zero contention, what is the approximate cycle speedup? *(10.5 / 10.2 ≈ 1.03×, not 20×.)*
3. Why is `cosine ≥ 0.997` not automatic typography PASS? *(Diffusion may amplify small conditioning differences; examine native-resolution images and tail cases.)*
4. What is the difference between a BF16-vs-BF16 backend experiment and a BF16-vs-INT8 experiment? *(The second also changes quantization, a separate causal variable.)*

**Acceptance:** the student can explain the precise conditioning bridge, compute fair ratios and the pipeline break-even, cite both reference studies, and correctly mark the not-yet-run course experiment `NOT_RUN`. An implementation PR must clear Gates A–H before new numbers appear in the README.
