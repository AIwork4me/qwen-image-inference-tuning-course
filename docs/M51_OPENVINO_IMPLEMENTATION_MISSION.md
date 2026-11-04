# M5.1 Implementation Mission — Codex / Linux Radeon Cloud W7900

**Current state:** the M5.1 curriculum has been added, but no new OpenVINO run is claimed. Implement the experiment on a **real** Radeon Cloud W7900-class GPU, not with mocked GPU measurements.

Read, verify exact commits, and inspect (read-only until required):
- https://github.com/AIwork4me/qwen3vl-cpu-benchmark (scripts/ov_encoder.py, scripts/bench_openvino_bridge.py, custom_nodes/qwen3vl_openvino_cpu, report/QWEN_IMAGE_E2E_HYBRID.md)
- https://github.com/AIwork4me/qwen-image21-openvino-alignment (CONCLUSIONS.md, reports/REPRODUCTION.md, reports/LIMITATIONS.md, scripts/patch_ov_artifact_textonly.py)
- Local M5.0 `lab/pipeline.py`, `course/M05_parallel_heterogeneous/LESSON.md`, `evidence/results/p5-H1-r2-H1-cpu-te/` and `evidence/results/p5-H2-H2-pipeline/`.

**Goal:** paired C0 PyTorch BF16 CPU versus C1 OpenVINO high-fidelity CPU on SAME EPYC/CPU host; optional C2 OpenVINO INT8 is a separate quantization axis. GPU G remains reference. Bring the exact Qwen-Image text-conditioning tensor (layer-36 **pre-final-RMSNorm**, ComfyUI tokenizer, second `<|im_start|>` trim) across the `LinuxLabCondLoader` boundary into the unchanged W7900 GPU DiT/VAE. Do not compare generic chat logits.

**Stages:** (1) capacity/ISA/cgroup + isolated OpenVINO runtime; (2) checkpoint/export/tokenizer provenance gate; (3) positive/negative conditioning shape/token identity/cosine/relativeL2/max_abs and fixed thresholds; (4) CPU paired cold/warm/fresh latency+RSS/thread sweep; (5) separate INT8 and known >40-token Zen4 artifact hazard; (6) identical-seed/latent E2E quality + native-resolution Chinese poster, human review PENDING; (7) four-prompt pipelines G/C0/C1/(C2), timestamp overlap + contention, end-to-end images/min + GPU VRAM; (8) final PASS/FAIL/BLOCKED/NOT_RUN report with links.

**Implementation request:** add `H3_OV_BF16` and optionally `H3_OV_INT8` CLI arms, safe incremental provisioning and isolated dependencies; CPU tests for planning/token alignment/record schema and actual GPU/CPU validation. New experimental data must land **only in `runs/m51-*/`**. Leave existing `evidence/`, H1/H2, B0 and original image results unchanged. Do not alter global ROCm, default `start.sh`, or silently copy unlicensed code. Freeze upstream source commits and model digests.

**Evidence constraints:** show measured source-specific numbers only. Never claim the 22–69× Zen5 result is a same-host gain over the course's PyTorch BF16 CPU, never equate LPIPS/embedding cosine with image quality, never claim 16.8GB reduction in E2E peak just from TE allocation. If export/runtime fails, publish a reproducible BLOCKED result and stop rather than inventing a speedup.

**Delivery:** independent test summary, exact host/versions, comprehensive M5.1 student lesson updated with physical results, source citation/attribution, new tests green, Draft PR (no merge), actual HEAD SHA, and a recommendation on whether OV BF16 and OV INT8 are useful at both the encoder and system levels.
