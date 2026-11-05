#!/usr/bin/env bash
# M5.1 OPTIONAL OpenVINO provisioning. Never part of the default M0-M4 bootstrap.
#
#   bash scripts/m51/provision_ov.sh              # export FP32 + BF16-compressed + INT8
#   bash scripts/m51/provision_ov.sh --check-only # verify existing artifacts only
#
# Requires: the pinned text encoder from scripts/cloud/start.sh, ~60 GB free,
# ~45 GB RAM peak for the FP32 export, and the isolated .venv-ov.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ART="${M51_OV_ARTIFACTS:-$REPO/runs/m51-ov-artifacts}"
PY="${PY:-python3}"
CHECK_ONLY=0
[[ "${1:-}" == "--check-only" ]] && CHECK_ONLY=1
cd "$REPO"
C(){ printf '\033[1;36m== %s ==\033[0m\n' "$*"; }
die(){ printf '\033[1;31mBLOCKED: %s\033[0m\n' "$*"; exit 1; }

C "1/6 Verify pinned text encoder"
if ! "$PY" - <<'PY'
import sys; sys.path.insert(0,".")
from lab import weights
r=weights.verify_weight("text_encoders/qwen3vl_8b_bf16.safetensors")
print(r)
raise SystemExit(0 if r["ok"] else 1)
PY
then die "text encoder absent/corrupt — run bash scripts/cloud/start.sh first"; fi

C "2/6 Disk and artifact status"
mkdir -p "$ART"
FREE_GIB="$(df -BG "$ART" | awk 'NR==2{gsub("G","",$4);print $4}')"
echo "  free on artifact volume: ${FREE_GIB} GiB; artifacts: $ART"
if [[ $CHECK_ONLY == 1 ]]; then
  ART="$ART" "$PY" - <<'PY'
import os,sys; sys.path.insert(0,".")
from lab.m51_contract import artifact_status
for name,files in [("fp32",["language_model_fp32.xml","language_model_fp32.bin"]),
                   ("bf16",["language_model_bf16.xml","language_model_bf16.bin"]),
                   ("int8",["language_model_int8.xml","language_model_int8.bin"])]:
    print(name, artifact_status(os.path.join(os.environ["ART"],name), files))
PY
  exit 0
fi
[[ ${FREE_GIB:-0} -ge 55 ]] || die "need ~55 GiB free for FP32+BF16+INT8 IRs (and more for caches)"

C "3/6 Isolated OpenVINO venv (ROCm torch is borrowed read-only, never replaced)"
if [[ ! -x .venv-ov/bin/python ]]; then
  "$PY" -m venv --system-site-packages .venv-ov
  SP="$(.venv-ov/bin/python -c 'import site;print(site.getsitepackages()[0])')"
  if ! .venv-ov/bin/python -c "import torch" 2>/dev/null; then
    TORCH_SP="$($PY -c 'import site;print(site.getsitepackages()[0])')"
    echo "$TORCH_SP" > "$SP/system_torch.pth"
  fi
fi
TMPDIR="${TMPDIR:-/workspace/tmp}" PIP_PROGRESS_BAR=off .venv-ov/bin/pip install -q --index-url https://pypi.org/simple/ \
  openvino==2026.4.1 nncf==3.4.0 comfy-aimdo==0.6.0 comfy-kitchen==0.2.40
.venv-ov/bin/python -c "import openvino,nncf,torch;print('openvino',openvino.__version__,'nncf',nncf.__version__,'torch',torch.__version__)"

C "4/6 Model config (text_config from the official Qwen-Image-2.1 text encoder)"
CFG="$ART/text_config.json"
if [[ ! -s "$CFG" ]]; then
  curl -fsSL --retry 5 --retry-all-errors -o "$CFG" \
    "https://hf-mirror.com/Qwen/Qwen-Image-2.1/raw/main/text_encoder/config.json" \
    || die "cannot fetch config.json (network); place it at $CFG manually"
fi

C "5/6 Export FP32 (peak ~40 GiB RAM, ~5 min)"
mkdir -p "$ART/fp32"
.venv-ov/bin/python -u lab/ov_export.py --config "$CFG" --out "$ART/fp32"

C "6/6 BF16-compressed deployment IR + INT8 weight compression"
mkdir -p "$ART/bf16" "$ART/int8"
.venv-ov/bin/python -u lab/ov_compress.py --src "$ART/fp32" --out "$ART/bf16"
.venv-ov/bin/python -u lab/ov_quantize.py --src "$ART/fp32" --out "$ART/int8"

C "Artifact status"
ART="$ART" "$PY" - <<'PY'
import os,sys; sys.path.insert(0,".")
from lab.m51_contract import artifact_status
for name,files in [("fp32",["language_model_fp32.xml","language_model_fp32.bin"]),
                   ("bf16",["language_model_bf16.xml","language_model_bf16.bin"]),
                   ("int8",["language_model_int8.xml","language_model_int8.bin"])]:
    print(name, artifact_status(os.path.join(os.environ["ART"],name), files)["status"])
PY
echo "DONE. Run: .venv-lab/bin/python -m lab run --experiment H3_OV_BF16 --smoke"
