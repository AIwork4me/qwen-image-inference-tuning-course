# M5.1 Reproduction

**Reference run:** `runs/m51-20261105-cleanroom-01/` · **Host:** EPYC 9334 + gfx1100 48 GiB · **Date:** 2026-11-05.

All commands are run from the course repository root. The default M0–M4 path is unchanged; OpenVINO is **optional**.

## 0. Host audit (read-only)

```bash
date -Is; uname -a; cat /etc/os-release; lscpu; free -h; df -h; df -i
rocminfo | grep -E 'Name:|gfx|Device Type'
rocm-smi --showproductname --showmeminfo vram
python3 - <<'PY'
import torch; print(torch.__version__, torch.version.hip, torch.cuda.is_available())
print(torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).gcnArchName)
PY
cat /sys/fs/cgroup/cpu.max /sys/fs/cgroup/memory.max
```

Expected: one gfx1100 AMD GPU, working ROCm PyTorch, >=55 GB disk free for the minimal path, >=45 GB cgroup RAM for the FP32 export. If ROCm PyTorch is broken, stop and fix the image; do not replace the working stack.

## 1. Clone and verify the M5.1 branch

```bash
git clone https://github.com/AIwork4me/qwen-image-inference-tuning-course.git
cd qwen-image-inference-tuning-course
git fetch origin --prune
git checkout feat/m5-1-openvino-cpu-te-course
git rev-parse HEAD   # 20a7f7e5afcb078b3899470365464f4922db10cb before implementation commits
```

If git-over-proxy returns HTTP 503, use the codeload tarball or retry; the reference run staged the pinned ComfyUI tarball and verified `comfyui_version.py == 0.43.0`.

## 2. Default bootstrap (unchanged)

```bash
bash scripts/cloud/start.sh --skip-smoke
.venv-lab/bin/python -m lab doctor --cloud
```

This stages SHA256-pinned weights, creates `.venv-lab` with a read-only bridge to the ROCm PyTorch, verifies the ComfyUI pinned tree, and runs the cloud doctor.

**Observed network workaround (2026-11-05 only):** `comfyui-workflow-templates==0.11.100` requires the optional `comfyui-workflow-templates-media-assets-02==0.1.26`, which was absent from the configured PyPI mirror. The runtime packages were installed with:

```bash
TMPDIR=/workspace/m51/tmp PIP_PROGRESS_BAR=off .venv-lab/bin/pip install -q --index-url https://pypi.org/simple/ \
  torchsde alembic 'SQLAlchemy>=2.0.0' 'av>=17.0.0' simpleeval 'kornia>=0.7.1' spandrel \
  comfy-aimdo==0.6.0 comfy-kitchen==0.2.40 comfyui-frontend-package==1.57.6 comfyui-embedded-docs==0.5.15
.venv-lab/bin/pip install -q --index-url https://pypi.org/simple/ --no-deps comfyui-workflow-templates==0.11.100
```

This does not affect inference; the first image smoke passed.

For the E2E quality run, download the turbo BF16 DiT as well:

```bash
MIRROR=https://hf-mirror.com
for spec in \
  "diffusion_models/qwen_image_2.1_turbo_bf16.safetensors 8b74237a7f505773e2c66726fa696444b4c704585c93a347cff9044937f7914a" \
  "text_encoders/qwen3vl_8b_bf16.safetensors 68bdc82bc1b66851162ae656225e7e2068166b603db19bd5d5a3b90eb12669a9" \
  "vae/qwen_image_2.1_vae_bf16.safetensors bb21f7473051e1ac368515dd3f2e15cd44d7a11748ee8823e1ddca3e4876b7c9"; do
  set -- $spec
  .venv-lab/bin/python -I lab/download_weights.py "$MIRROR/Comfy-Org/Qwen-Image-2.1/resolve/main/$1" "weights/comfyui-split/$1" 16
done
```

Verify every file:

```bash
.venv-lab/bin/python - <<'PY'
import sys; sys.path.insert(0,'.')
from lab import weights
for rel in weights.WEIGHTS: print(weights.verify_weight(rel))
PY
```

## 3. Optional OpenVINO provisioning (explicit, not in start.sh)

```bash
bash scripts/m51/provision_ov.sh
M51_OV_ARTIFACTS=/opt/m51-artifacts bash scripts/m51/provision_ov.sh --check-only
```

Artifacts: FP32 (25.876 GiB), FP16-stored/BF16-execution (12.938 GiB), INT8 (6.474 GiB). If `/workspace` is small, set `M51_OV_ARTIFACTS=/opt/m51-artifacts` and symlink `runs/m51-ov-artifacts`.

## 4. M5.0 baseline

```bash
.venv-lab/bin/python -I lab/m51_baseline.py --device cuda \
  --out runs/m51-20261105-cleanroom-01/baseline --repeats 3
.venv-lab/bin/python -I lab/m51_baseline.py --device cpu \
  --out runs/m51-20261105-cleanroom-01/baseline --repeats 2 --threads 16
```

CPU time is ~12–15 min for these four prompts. Do not mix cold and warm samples.

## 5. OpenVINO CLI

```bash
.venv-lab/bin/python -m lab run --experiment H3_OV_FP32 --smoke
.venv-lab/bin/python -m lab run --experiment H3_OV_BF16 --smoke
.venv-lab/bin/python -m lab run --experiment H3_OV_INT8 --smoke
.venv-lab/bin/python -m lab run --experiment H3_OV_BF16       # formal
```

The CLI byte-verifies the OpenVINO artifacts (streaming SHA256 vs the conversion manifest, sizes, precision identity), checks `.venv-ov`, and applies a conservative RAM gate (host MemAvailable vs cgroup-limit-minus-anon, safety headroom) before writing `H3_RESULTS.json`. The record carries per-gate statuses (`environment`, `artifact_integrity`, `encoder_runtime`, `token_contract`, `conditioning_alignment`, `e2e_gpu`, `performance`, `human_quality`) plus a derived `release_status` — a successful encoder execution alone never yields a release PASS. If artifacts are absent it prints the provisioning command and exits with BLOCKED; it never fakes a run.

## 6. Raw encoder probes and analysis

```bash
# FP32 alignment reference (compile ~85 s, RSS ~50 GB)
TMPDIR=/workspace/m51/tmp .venv-ov/bin/python -u lab/ov_encoder.py \
  --arm C1a_OV_FP32 --model-dir /opt/m51-artifacts/fp32 --precision f32 \
  --out runs/m51-20261105-cleanroom-01/ov/run_fp32 --threads 16 --repeats 3

# BF16 deployment (FP16-stored IR)
TMPDIR=/workspace/m51/tmp .venv-ov/bin/python -u lab/ov_encoder.py \
  --arm C1b_OV_BF16 --model-dir /opt/m51-artifacts/bf16 --xml language_model_bf16.xml \
  --precision bf16 --out runs/m51-20261105-cleanroom-01/ov/run_bf16_compressed --threads 16 --repeats 3

# INT8
TMPDIR=/workspace/m51/tmp .venv-ov/bin/python -u lab/ov_encoder.py \
  --arm C2_OV_INT8 --model-dir /opt/m51-artifacts/int8 --xml language_model_int8.xml \
  --precision f32 --dynamic-quant-group 32 \
  --out runs/m51-20261105-cleanroom-01/ov/run_int8 --threads 16 --repeats 3

# Alignment, speedup table, thread sweep, long-sequence stress
TMPDIR=/workspace/m51/tmp .venv-ov/bin/python lab/m51_alignment.py --run runs/m51-20261105-cleanroom-01
TMPDIR=/workspace/m51/tmp .venv-ov/bin/python lab/m51_perf.py --run runs/m51-20261105-cleanroom-01
TMPDIR=/workspace/m51/tmp .venv-ov/bin/python lab/m51_threads.py --model-dir /opt/m51-artifacts/bf16 \
  --out runs/m51-20261105-cleanroom-01/performance/thread_sweep_bf16.json --threads 4,8,16,32
TMPDIR=/workspace/m51/tmp .venv-ov/bin/python lab/m51_longseq.py --arm C2_OV_INT8 \
  --model-dir /opt/m51-artifacts/int8 --xml language_model_int8.xml --precision f32 \
  --out runs/m51-20261105-cleanroom-01/ov/longseq
```

## 7. E2E and four-request pipeline

```bash
# Native G, native C0 (one prompt), precomputed arms, grids and LPIPS
TMPDIR=/workspace/m51/tmp .venv-lab/bin/python -u lab/m51_e2e.py \
  --run runs/m51-20261105-cleanroom-01 --port 8199
TMPDIR=/workspace/m51/tmp .venv-lab/bin/python lab/m51_e2e_summary.py \
  --run runs/m51-20261105-cleanroom-01

# Steady GPU-native latency (fresh session after one warm-up)
TMPDIR=/workspace/m51/tmp .venv-lab/bin/python -m lab.m51_e2e_warm \
  --run runs/m51-20261105-cleanroom-01 --port 8199

# Pipeline A/C/D (fast), then B (~11 min)
TMPDIR=/workspace/m51/tmp .venv-lab/bin/python -u lab/m51_pipeline.py \
  --run runs/m51-20261105-cleanroom-01 --port 8199 --arms A,C,D
TMPDIR=/workspace/m51/tmp .venv-lab/bin/python -u lab/m51_pipeline.py \
  --run runs/m51-20261105-cleanroom-01 --port 8199 --arms B
TMPDIR=/workspace/m51/tmp .venv-lab/bin/python lab/m51_pipeline_summary.py \
  --run runs/m51-20261105-cleanroom-01
```

`lab/m51_e2e_warm.py` records a fresh warm-up, then the three frozen prompts. The committed `e2e/warm_gpu_native.json` contains the measured reference values.

## 8. Tests and evidence integrity

```bash
.venv-lab/bin/python -m pytest tests/ -q          # 53 passed
git status --short
git diff --check
git diff --stat
```

The original `evidence/` tree is not modified by M5.1. New measurements live only under `runs/m51-*`.

## 9. Recovery

| Symptom | Action |
|---|---|
| Disk full during OpenVINO compile | Delete `runs/**/cache/` (compile caches) or point `M51_OV_ARTIFACTS` to `/opt/m51-artifacts`; keep the IRs. |
| FP32 export OOM | Use the BF16 path or run on a host with a higher cgroup memory cap; the reference peak was 40.0 GiB for export, 50.5 GiB runtime. |
| Worker does not start | Check `stderr` in the worker; reinstall `.venv-ov` with `scripts/m51/provision_ov.sh --check-only` first. |
| GPU request hangs | Retry with a fresh server; one transient hang was observed on arm A and did not reproduce. Record it as a failure, do not hide it. |
| Missing model | Re-run `scripts/cloud/start.sh` (resumable ranged downloads, SHA-verified). |
| Network 503 | Retry; the reference used hf-mirror ranged transfers and codeload tarballs. |
| INT8 negative gate fails | Expected on this artifact; use BF16 deployment or accept INT8 only for cfg=1/exploratory work. |
