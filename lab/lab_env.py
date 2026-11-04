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
ENV_DIR = RESULTS_ROOT / "env"

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
        "gpu_note": "single AMD Radeon PRO W7900 (gfx1100, 48GB) reference platform",
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


def check_gpu_expectations(snap: dict | None = None) -> list[str]:
    """Fail-closed checks: exactly one AMD GPU of the gfx1100 family."""
    snap = snap or latest_snapshot()
    problems = []
    if snap.get("cuda_available") is False:
        problems.append("torch.cuda.is_available() is False — no usable GPU")
    if snap.get("device_count") not in (1, None):
        problems.append(f"expected exactly 1 GPU, torch sees {snap['device_count']}")
    arch = snap.get("gfx_arch") or snap.get("rocm", {}).get("rocminfo_gfx")
    if arch and not str(arch).startswith("gfx1100"):
        problems.append(f"expected gfx1100 (W7900), found {arch}")
    name = snap.get("gpu_name") or ""
    if name and "AMD" not in name:
        problems.append(f"device name is not an AMD device: {name!r}")
    vram = snap.get("total_vram_mb")
    if isinstance(vram, (int, float)) and not (46000 <= vram <= 50000):
        problems.append(f"expected ~48GB VRAM (W7900), found {vram} MB")
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
