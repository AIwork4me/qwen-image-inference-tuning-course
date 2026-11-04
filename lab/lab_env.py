"""Lab environment: snapshots, GPU health gate, stack introspection."""

from __future__ import annotations

import json
import platform
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LAB_DIR = Path(__file__).resolve().parent
RESULTS_ROOT = REPO / "evidence" / "results"
QUALITY_ROOT = REPO / "evidence" / "quality"

# Student/new runs are ISOLATED from the frozen official campaign under
# evidence/ (immutable, never written by the lab). Everything a student or a
# fresh measurement session produces lands under runs/ instead:
#   runs/<run_id>/          schema-valid records
#   runs/raw/<run_id>/      generated PNGs (stay on the student host)
#   runs/env/               environment snapshots
#   runs/doctor.json        latest doctor output
STUDENT_RUNS_ROOT = REPO / "runs"
RUNS_ENV_DIR = STUDENT_RUNS_ROOT / "env"
ENV_DIR = RUNS_ENV_DIR  # snapshots for NEW measurements (the 14 snapshots of
# the official campaign remain frozen under evidence/results/env/)

COMFY_DIR = REPO / "ComfyUI"
COMFY_PINNED_COMMIT = "b5a432157de6b8884f81104cd02926adf4d597e4"

# Marker verifying the ComfyUI checkout matches the pinned commit (the
# course network may not allow git clone; file-tree verification is the
# recorded evidence — see docs/REPRODUCTION_GUIDE.md).
COMFY_TREE_MARKER = COMFY_DIR / ".lab_pinned_commit"


def _run(cmd: list[str], timeout: int = 30) -> str:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return (r.stdout + r.stderr).strip()
    except Exception as exc:  # noqa: BLE001
        return f"<unavailable: {exc}>"


def torch_stack() -> dict:
    import torch

    props = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
    return {
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
        "device_name": torch.cuda.get_device_name(0) if props else None,
        "gfx_arch": getattr(props, "gcnArchName", None),
        "total_vram_mb": round(props.total_memory / 2**20) if props else None,
    }


def _gfx_from_rocminfo() -> str | None:
    out = _run(["rocminfo"])
    m = re.search(r"^\s+Name:\s+(gfx\d+)", out, re.M)
    return m.group(1) if m else None


def rocm_versions() -> dict:
    drv = _run(["rocm-smi", "--showdriverversion"])
    driver = None
    for line in drv.splitlines():
        if "Driver version" in line:
            driver = line.split(":")[-1].strip()
    return {
        "rocminfo_gfx": _gfx_from_rocminfo(),
        "driver": driver,
        "rocprofv3": _run(["rocprofv3", "--version"]),
    }


def comfyui_version() -> dict:
    v = {}
    try:
        ns = {}
        exec((COMFY_DIR / "comfyui_version.py").read_text(), ns)
        v["version"] = ns.get("__version__")
    except Exception as exc:  # noqa: BLE001
        v["version_error"] = str(exc)
    v["pinned_commit"] = COMFY_PINNED_COMMIT
    v["tree_marker_ok"] = (
        COMFY_TREE_MARKER.exists()
        and COMFY_TREE_MARKER.read_text().strip() == COMFY_PINNED_COMMIT
    )
    return v


def build_snapshot() -> dict:
    snap = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "os_build": f"{platform.system()} {platform.release()} ({platform.version()})",
        "python_version": platform.python_version(),
        "cpu": _run(["bash", "-lc", "grep -m1 'model name' /proc/cpuinfo"]).split(":")[-1].strip(),
        "nproc": _run(["nproc"]),
        "ram_total_gb": None,
        "gpu_name": None,
        "gfx_arch": None,
        "torch": None,
        "hip": None,
        "comfyui": comfyui_version(),
        "repo_head": _run(["git", "-C", str(REPO), "rev-parse", "HEAD"]),
        "repo_branch": _run(["git", "-C", str(REPO), "rev-parse", "--abbrev-ref", "HEAD"]),
        "gpu_note": "detected at snapshot time (see gpu_name/gfx_arch; the "
                    "reference campaign evidence was measured on a single "
                    "Radeon PRO W7900 48GB gfx1100)",
    }
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    snap["ram_total_gb"] = round(int(line.split()[1]) / 2**20, 1)
    except OSError:
        pass
    try:
        stack = torch_stack()
        snap.update(stack)
        snap["gpu_name"] = stack.get("device_name") or snap["gpu_name"]
        snap["gfx_arch"] = stack.get("gfx_arch") or snap["gfx_arch"]
        snap["torch"] = stack.get("torch")
        snap["hip"] = stack.get("hip")
    except Exception as exc:  # noqa: BLE001
        snap["torch_error"] = str(exc)
    snap["rocm"] = rocm_versions()
    return snap


def write_snapshot() -> str:
    ENV_DIR.mkdir(parents=True, exist_ok=True)
    snap = build_snapshot()
    name = "snapshot-" + datetime.now().strftime("%Y%m%d-%H%M%S") + ".json"
    (ENV_DIR / name).write_text(json.dumps(snap, indent=2), encoding="utf-8")
    return name


def latest_snapshot() -> dict:
    snaps = sorted(ENV_DIR.glob("snapshot-*.json")) if ENV_DIR.is_dir() else []
    return json.loads(snaps[-1].read_text(encoding="utf-8")) if snaps else {}


def classify_gpu_expectations(snap: dict | None = None) -> tuple[list[str], list[str]]:
    """Split GPU expectations into (blocking problems, advisory warnings).

    Blocking: no GPU at all, wrong GPU count, non-AMD device — measuring
    here would produce invalid records (CPU fallback, multi-GPU ambiguity).
    Advisory (non-blocking): a different-but-working AMD Radeon (arch or
    VRAM differs from the W7900 reference) — the methodology still applies,
    the numbers are simply not comparable to the committed reference.
    """
    snap = snap or latest_snapshot()
    problems: list[str] = []
    warnings: list[str] = []
    if snap.get("cuda_available") is False:
        problems.append("torch.cuda.is_available() is False — no usable GPU")
    if snap.get("device_count") not in (1, None):
        problems.append(f"expected exactly 1 GPU, torch sees {snap['device_count']}")
    arch = snap.get("gfx_arch") or snap.get("rocm", {}).get("rocminfo_gfx")
    name = snap.get("gpu_name") or ""
    vram = snap.get("total_vram_mb")
    if arch and not str(arch).startswith("gfx1100"):
        warnings.append(
            f"gfx arch is {arch}, not gfx1100 (W7900-class) — this is NOT "
            "the validated reference platform; your numbers are a valid "
            "independent measurement but not comparable to the committed "
            "evidence")
    if name and "AMD" not in name:
        problems.append(f"device name is not an AMD device: {name!r}")
    if isinstance(vram, (int, float)) and not (46000 <= vram <= 50000):
        warnings.append(
            f"VRAM is {vram} MB, not ~48 GB (W7900-class) — check which "
            "experiments fit (bf16 40-step peaks at ~37.7 GB; int8 turbo "
            "at ~31.0 GB)")
    return problems, warnings


def check_gpu_expectations(snap: dict | None = None) -> list[str]:
    """Fail-closed checks: exactly one AMD GPU of the gfx1100 family.

    This is the STRICT gate used by `python -m lab doctor` (formal
    reproduction of the reference campaign). Radeon-Cloud staged mode uses
    `classify_gpu_expectations` so a working non-W7900 Radeon produces an
    explicit warning instead of a silent substitution.
    """
    problems, _ = classify_gpu_expectations(snap)
    return problems


def check_server_linux(base_url: str) -> dict:
    """GPU gate for a running ComfyUI server (anti CPU-fallback)."""
    import requests

    stats = requests.get(f"{base_url}/system_stats", timeout=10).json()
    devices = stats.get("devices", [])
    amd = [d for d in devices if "AMD" in (d.get("name") or "")]
    if not devices or not amd:
        raise RuntimeError(f"no AMD device in /system_stats: {devices} — refusing to benchmark")
    return stats
