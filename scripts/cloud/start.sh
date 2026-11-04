#!/usr/bin/env bash
# =============================================================================
# Radeon Cloud one-command onboarding for the Qwen-Image inference course
#
#   cd /workspace
#   git clone https://github.com/AIwork4me/qwen-image-inference-tuning-course.git
#   cd qwen-image-inference-tuning-course
#   bash scripts/cloud/start.sh
#
# What it does (all steps idempotent — safe to re-run after any interruption
# or instance restart; nothing already downloaded is ever re-downloaded):
#   1. Diagnose the instance (GPU/ROCm/PyTorch/disk/proxy) — no secrets used.
#   2. Fetch ComfyUI at the pinned commit (git clone, tarball fallback).
#   3. Create .venv-lab that BORROWS the preinstalled ROCm PyTorch via a
#      read-only .pth bridge (torch is never reinstalled or upgraded).
#   4. Stage the minimum pinned model set for a first image (~25.5 GB:
#      turbo-int8 DiT + 8B text encoder + VAE), verifying SHA256 digests.
#   5. Run the cloud-staged doctor (must not FAIL).
#   6. Generate ONE bounded smoke image and print its path + next lesson.
#
# Options:
#   --all-models   also fetch the remaining variants (bf16/int8 40-step DiTs
#                  + turbo bf16, ~61 GB total — check your quota first)
#   --skip-smoke   stop after doctor (no image generated)
#   --smoke-only   skip setup steps that are already done, just re-smoke
#   --hf-base URL  model source override (default: hf-mirror.com, the mirror
#                  the validated cloud network allows; use
#                  https://huggingface.co where it is reachable)
#
# This script NEVER writes into evidence/ — all new measurements land in
# runs/ (student area). The frozen campaign stays immutable.
# =============================================================================
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"

ALL_MODELS=0
SKIP_SMOKE=0
SMOKE_ONLY=0
HF_BASE="${HF_BASE:-https://hf-mirror.com}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --all-models) ALL_MODELS=1 ;;
    --skip-smoke) SKIP_SMOKE=1 ;;
    --smoke-only) SMOKE_ONLY=1 ;;
    --hf-base) HF_BASE="$2"; shift ;;
    *) echo "unknown option: $1 (see header of this script)"; exit 2 ;;
  esac
  shift
done
export ALL_MODELS

C()    { printf '\033[1;36m== %s ==\033[0m\n' "$*"; }
OK()   { printf '\033[1;32m   [ok]\033[0m %s\n' "$*"; }
WARN() { printf '\033[1;33m   [warn]\033[0m %s\n' "$*"; }
FAIL() { printf '\033[1;31m   [FAIL]\033[0m %s\n' "$*"; }
die()  { FAIL "$*"; exit 1; }

# -----------------------------------------------------------------------------
C "1/6 Instance diagnosis (no secrets, nothing installed yet)"
# -----------------------------------------------------------------------------
SYS_PY="${SYS_PY:-$(command -v python3 || true)}"
[[ -n "$SYS_PY" ]] || die "python3 not found — use a Radeon Cloud Linux image with PyTorch ROCm preinstalled"
"$SYS_PY" - <<'PYEOF' || die "no working ROCm PyTorch GPU stack in this image"
import os, platform, re, shutil, subprocess, sys

def run(cmd, t=20):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=t)
        return (r.stdout + r.stderr).strip()
    except Exception as e:
        return f"<{e}>"

print(f"  OS        : {platform.system()} {platform.release()}")
print(f"  Python    : {sys.version.split()[0]}  ({sys.executable})")
print(f"  CPUs      : {os.cpu_count()} visible")
try:
    with open("/proc/meminfo") as f:
        total = next(int(l.split()[1]) for l in f if l.startswith("MemTotal"))
    print(f"  RAM       : {total/2**20:.0f} MiB (host view; the container may see less)")
except OSError:
    pass
try:
    import torch
except Exception as e:
    print(f"  PyTorch   : IMPORT FAILED: {e}")
    raise SystemExit(1)
print(f"  PyTorch   : {torch.__version__}  hip {torch.version.hip}")
if not torch.version.hip:
    print("  GPU       : torch is NOT a ROCm/HIP build — pick a PyTorch-ROCm image")
    raise SystemExit(1)
if not torch.cuda.is_available():
    print("  GPU       : torch.cuda.is_available() = False — no usable GPU")
    raise SystemExit(1)
p = torch.cuda.get_device_properties(0)
print(f"  GPU       : {p.name} / {getattr(p, 'gcnArchName', '?')} / "
      f"{p.total_memory/2**30:.1f} GiB / {torch.cuda.device_count()} device(s)")
m = re.search(r"^\s+Name:\s+(gfx\d+)", run(["rocminfo"]), re.M)
print(f"  ROCm gfx  : {m.group(1) if m else 'unknown (rocminfo unavailable)'}")
print(f"  rocprofv3 : {'yes' if shutil.which('rocprofv3') else 'NO (M2 kernel profiling will be blocked)'}")
repo = os.path.dirname(os.path.abspath("extra_model_paths.yaml"))
st = os.statvfs(repo)
print(f"  Disk {repo:<11}: {st.f_bavail*st.f_frsize/2**30:.0f} GiB free")
# /tmp carries ROCm/LLVM codegen scratch — a full tmpfs kills the ComfyUI
# server with a confusing "LLVM ERROR: IO failure on output stream"
try:
    stt = os.statvfs("/tmp")
    tmp_free = stt.f_bavail * stt.f_frsize / 2**30
    print(f"  /tmp        : {tmp_free:.1f} GiB free "
          + ("(LOW — clean it: LLVM codegen needs a few GB, a full /tmp "
             "crashes the server)" if tmp_free < 2 else ""))
except OSError:
    pass
proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
print(f"  Egress    : {'proxy ' + proxy if proxy else 'direct'}")
PYEOF
OK "stack diagnosed"

# disk budget guard — INCREMENTAL: only files that still need downloading
# count; --smoke-only downloads nothing and is never disk-blocked. The pure
# logic (with regression tests) lives in scripts/cloud/plan_stage.py.
PLAN_FLAGS=""
[[ $ALL_MODELS == 1 ]] && PLAN_FLAGS="$PLAN_FLAGS --all-models"
[[ $SMOKE_ONLY == 1 ]] && PLAN_FLAGS="$PLAN_FLAGS --smoke-only"
PLAN_OUT="$("$SYS_PY" scripts/cloud/plan_stage.py plan --repo "$REPO" $PLAN_FLAGS)" \
  || { grep '^MESSAGE' <<<"$PLAN_OUT" | sed 's/^MESSAGE /BLOCKED: /'; \
       die "disk budget blocked (see above)"; }
sed 's/^MESSAGE /  /' <<<"$PLAN_OUT" | head -1
if [[ $SMOKE_ONLY == 1 ]]; then
  C "smoke-only: skipping setup steps 2-4"
else

# -----------------------------------------------------------------------------
C "2/6 ComfyUI @ pinned commit b5a4321 (v0.43.0) — safe staging"
# -----------------------------------------------------------------------------
COMFY_DIR="$REPO/ComfyUI"
COMFY_PIN="b5a432157de6b8884f81104cd02926adf4d597e4"
COMFY_MARKER="$COMFY_DIR/.lab_pinned_commit"
COMFY_ACTION="$("$SYS_PY" scripts/cloud/plan_stage.py comfy-action --repo "$REPO" --action-only)"
case "$COMFY_ACTION" in
  keep)
    OK "ComfyUI tree already at pinned commit (marker + version + main.py verified)"
    ;;
  refuse)
    die "$COMFY_DIR exists but is not a directory — resolve it manually; nothing was touched"
    ;;
  quarantine_then_stage)
    Q="$REPO/ComfyUI.unknown.$(date +%Y%m%d-%H%M%S)"
    mv "$COMFY_DIR" "$Q" || die "could not move the unrecognized ComfyUI tree aside"
    WARN "existing ComfyUI tree was NOT the lab's pinned checkout — moved to"
    WARN "  $Q   (nothing was deleted; remove it yourself once you've checked it)"
    ;;
esac
if [[ "$COMFY_ACTION" != "keep" ]]; then
  # Stage into a private directory THIS run owns; verify; then move into
  # place. A failure cleans up only the staging dir — an existing user tree
  # was already quarantined (never deleted) above.
  # NB: do NOT pre-create "$STAGING/tree" — the tarball path renames the
  # extracted dir TO that path, and mv-into-an-existing-dir would nest it.
  STAGING="$REPO/.comfy-staging.$$"
  rm -rf "$STAGING"   # safe: uniquely named, created by this very run
  mkdir -p "$STAGING"
  if timeout 300 git -c http.version=HTTP/1.1 clone -q https://github.com/comfyanonymous/ComfyUI.git "$STAGING/tree" 2>/dev/null \
     && git -C "$STAGING/tree" checkout -q "$COMFY_PIN" 2>/dev/null; then
    OK "ComfyUI cloned via git at $COMFY_PIN"
  else
    WARN "git clone unavailable (cloud egress proxy) — tarball fallback"
    TARBALL="$(mktemp /tmp/comfy-XXXXXX.tar.gz)"
    curl -fsSL --retry 3 -o "$TARBALL" \
      "https://codeload.github.com/comfyanonymous/ComfyUI/tar.gz/$COMFY_PIN" \
      || "$SYS_PY" - "https://codeload.github.com/comfyanonymous/ComfyUI/tar.gz/$COMFY_PIN" "$TARBALL" <<'PYEOF'
import requests, sys
open(sys.argv[2], "wb").write(requests.get(sys.argv[1], timeout=600).content)
PYEOF
    [[ -s "$TARBALL" ]] || { rm -rf "$STAGING"; die "cannot fetch ComfyUI $COMFY_PIN (git and tarball both blocked by the network)"; }
    tar xzf "$TARBALL" -C "$STAGING" || { rm -rf "$STAGING"; die "ComfyUI tarball extraction failed"; }
    mv "$STAGING/ComfyUI-$COMFY_PIN" "$STAGING/tree" || { rm -rf "$STAGING"; die "unexpected tarball layout"; }
    rm -f "$TARBALL"
    OK "ComfyUI extracted from tarball at $COMFY_PIN"
  fi
  # integrity gate BEFORE anything is moved into place
  "$SYS_PY" scripts/cloud/plan_stage.py verify-staged "$STAGING/tree" >/dev/null \
    || { rm -rf "$STAGING"; die "staged ComfyUI tree failed verification"; }
  if [[ -e "$COMFY_DIR" ]]; then
    rm -rf "$STAGING"
    die "$COMFY_DIR appeared during staging — aborting without touching it"
  fi
  mv "$STAGING/tree" "$COMFY_DIR" && rmdir "$STAGING" 2>/dev/null || rm -rf "$STAGING"
  printf '%s\n' "$COMFY_PIN" > "$COMFY_MARKER"
  OK "staged ComfyUI verified (v0.43.0) and moved into place"
fi

# -----------------------------------------------------------------------------
C "3/6 Lab venv (.venv-lab) — borrows the preinstalled ROCm torch"
# -----------------------------------------------------------------------------
VENV_PY="$REPO/.venv-lab/bin/python"
if [[ -x "$VENV_PY" ]] && "$VENV_PY" -c "import torch, websocket, jsonschema, matplotlib, requests" 2>/dev/null; then
  OK ".venv-lab already complete (torch $($VENV_PY -c 'import torch; print(torch.__version__)'))"
else
  [[ -x "$VENV_PY" ]] || "$SYS_PY" -m venv .venv-lab
  # read-only bridge to the system ROCm site-packages (torch stays pristine;
  # pip sees it as installed and never reinstalls over a working ROCm build)
  "$SYS_PY" - "$REPO" <<'PYEOF'
import site, sys
from pathlib import Path
repo = Path(sys.argv[1])
sys_site = Path(site.getsitepackages()[0]).resolve()
venv_site = repo / ".venv-lab" / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
if sys_site == venv_site.resolve():
    raise SystemExit("active python IS the lab venv (unexpected)")
(venv_site / "system_rocm_torch.pth").write_text(str(sys_site) + "\n")
print(f"  bridged system site-packages: {sys_site}")
PYEOF
  if ! PIP_PROGRESS_BAR=off .venv-lab/bin/pip install -q -r ComfyUI/requirements.txt -r requirements-dev.txt -r requirements-quality.txt 2>/tmp/pip_first.log; then
    WARN "pip failed on the configured index — retrying with the official PyPI"
    PIP_PROGRESS_BAR=off .venv-lab/bin/pip install -q --index-url https://pypi.org/simple/ \
      -r ComfyUI/requirements.txt -r requirements-dev.txt -r requirements-quality.txt 2>/tmp/pip_retry.log \
      || { tail -5 /tmp/pip_retry.log; die "pip install failed on both indexes"; }
  fi
  .venv-lab/bin/python -c "import torch; assert torch.version.hip and torch.cuda.is_available()" \
    || die "the torch bridge broke — delete .venv-lab and re-run (and please report this bug)"
  OK "venv ready (torch $(.venv-lab/bin/python -c 'import torch; print(torch.__version__)'))"
fi

# -----------------------------------------------------------------------------
C "4/6 Pinned model staging (SHA256-anchored, resumable, staged)"
# -----------------------------------------------------------------------------
# The download list comes from the INCREMENTAL plan computed above (only
# absent/size-mismatched files are fetched; present files cost nothing).
need_dl=()
while IFS= read -r line; do
  rel="${line#DOWNLOAD }"
  need_dl+=("$rel")
  OK "needs download: ${rel##*/}"
done < <(grep '^DOWNLOAD ' <<<"$PLAN_OUT")
n_present=$(grep -c '^PRESENT ' <<<"$PLAN_OUT" || true)
[[ $n_present -gt 0 ]] && OK "$n_present pinned file(s) already present (not re-downloaded)"
n_broken=$(grep -c '^BROKEN ' <<<"$PLAN_OUT" || true)
[[ $n_broken -gt 0 ]] && WARN "$n_broken file(s) present with wrong size — will be re-downloaded"

if [[ ${#need_dl[@]} -gt 0 ]]; then
  echo "  downloading ${#need_dl[@]} file(s) from $HF_BASE (16-way ranged, resumable)"
  for rel in "${need_dl[@]}"; do
    echo "  -> $rel"
    .venv-lab/bin/python -I lab/download_weights.py \
      "$HF_BASE/Comfy-Org/Qwen-Image-2.1/resolve/main/$rel" "weights/comfyui-split/$rel" 16 \
      || die "download failed for $rel — re-run this script to resume (progress is kept)"
  done
fi
echo "  verifying pinned SHA256 digests (full hash on first check)..."
.venv-lab/bin/python - <<'PYEOF' || die "digest verification FAILED — see the files above; delete the corrupt one and re-run"
import sys
sys.path.insert(0, ".")
from lab import weights
stage1 = {"diffusion_models/qwen_image_2.1_turbo_int8_convrot.safetensors",
          "text_encoders/qwen3vl_8b_bf16.safetensors",
          "vae/qwen_image_2.1_vae_bf16.safetensors"}
results = [weights.verify_weight(rel) for rel in weights.WEIGHTS]
for r in results:
    if r["ok"]:
        print(f"  ok     : {r['rel'].split('/')[-1]} ({r['reason']})")
    elif r["rel"] in stage1:
        print(f"  BROKEN : {r['rel'].split('/')[-1]} ({r['reason']})")
    else:
        print(f"  absent : {r['rel'].split('/')[-1]} (optional for the smoke path)")
fatal = [r for r in results if not r["ok"] and r["rel"] in stage1]
if fatal:
    sys.exit(1)
print("  stage-1 digests verified against lab/weights.py pins")
PYEOF

fi  # SMOKE_ONLY

# -----------------------------------------------------------------------------
C "5/6 Doctor (cloud-staged preflight)"
# -----------------------------------------------------------------------------
PYLAB=".venv-lab/bin/python"
[[ -x "$PYLAB" ]] || PYLAB="$SYS_PY"
"$PYLAB" -m lab doctor --cloud || die "doctor FAILED — fix the reported problem before continuing"

# -----------------------------------------------------------------------------
C "6/6 First image (bounded smoke, BEST config = turbo-int8 8-step)"
# -----------------------------------------------------------------------------
if [[ $SKIP_SMOKE == 1 ]]; then
  OK "smoke skipped (--skip-smoke); run it later:"
  echo "       python -m lab run --experiment BEST --smoke --only-prompt p1-zh-scene"
else
  SMOKE_TAG="cloud-first-image-$(date +%Y%m%d-%H%M%S)"
  "$PYLAB" -m lab run --experiment BEST --smoke --only-prompt p1-zh-scene --date "$SMOKE_TAG" \
    || die "smoke generation failed — inspect runs/${SMOKE_TAG}-BEST-smoke/server.log"
  IMG=$(ls -t runs/raw/${SMOKE_TAG}-BEST-smoke/*.png 2>/dev/null | head -1 || true)
  REC=$(ls runs/${SMOKE_TAG}-BEST-smoke/*.json 2>/dev/null | head -1 || true)
  echo
  C "YOUR FIRST IMAGE"
  [[ -n "$IMG" ]] && { echo "  image  : $REPO/$IMG"; echo "  record : $REPO/$REC"; }
fi

echo
C "Next steps"
cat <<EOF
  1. Open course/M01_baseline/LESSON.md — measure the frozen B0 baseline
     (needs the bf16 DiT: re-run this script with --all-models, +14.2 GB).
  2. Budget rule of thumb (W7900-class, full protocol session):
       B0 40-step ~16 min   L1/L1O turbo ~5 min   F1 ~13 min
       F2 INT8 ~15 min      LF* ~4 min            BEST ~4 min
     Classroom smoke variants cost 1-3 minutes (--smoke).
  3. Your measurements live in runs/ and never touch the frozen evidence
     under evidence/ — aggregate yours with:
       python -m lab report --scope student --write
  4. GPU time is a metered resource: stop or destroy the instance when you
     are not using it. runs/ persists together with /workspace.
EOF
echo
OK "start.sh finished"
