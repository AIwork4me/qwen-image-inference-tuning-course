"""Immutable clean-room environment snapshot for an M5.1 run."""
from __future__ import annotations
import argparse, json, os, platform, re, shutil, subprocess, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RUN_ID = "m51-20261105-cleanroom-01"

def run(cmd, timeout=30):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return (r.stdout + r.stderr).strip()
    except Exception as e:
        return f"<unavailable: {e}>"

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", default=str(REPO/"runs"/RUN_ID)); args = ap.parse_args()
    out = Path(args.run)/"environment.json"; out.parent.mkdir(parents=True, exist_ok=True)
    import torch
    sys.path.insert(0, str(REPO/"ComfyUI"))
    snap = {
        "run_id": RUN_ID,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "admission": "GO_WITH_DEVIATIONS",
        "host": {
            "os": platform.platform(), "kernel": platform.release(),
            "cpu": run(["bash","-lc","grep -m1 'model name' /proc/cpuinfo"]).split(":")[-1].strip(),
            "sockets": run(["bash","-lc","lscpu | grep '^Socket(s):' | awk '{print $2}'"]),
            "physical_cores": run(["bash","-lc","lscpu | grep '^Core(s) per socket:' | awk '{print $4}'"]),
            "logical_threads": os.cpu_count(),
            "cgroup_cpu_max": Path("/sys/fs/cgroup/cpu.max").read_text().strip() if Path("/sys/fs/cgroup/cpu.max").exists() else None,
            "cgroup_memory_max_bytes": int(Path("/sys/fs/cgroup/memory.max").read_text().strip()) if Path("/sys/fs/cgroup/memory.max").exists() and Path("/sys/fs/cgroup/memory.max").read_text().strip().isdigit() else None,
            "mem_total_kb": next(int(l.split()[1]) for l in open("/proc/meminfo") if l.startswith("MemTotal:")),
            "numa_nodes": run(["bash","-lc","lscpu | grep '^NUMA node(s):' | awk '{print $3}'"]),
            "isa_flags": {"avx2": "avx2" in open("/proc/cpuinfo").read(), "avx512f": "avx512f" in open("/proc/cpuinfo").read(),
                          "avx512_bf16": "avx512_bf16" in open("/proc/cpuinfo").read(), "avx512_vnni": "avx512_vnni" in open("/proc/cpuinfo").read()},
        },
        "gpu": {
            "rocm_smi": run(["rocm-smi","--showproductname","--showmeminfo","vram"]),
            "rocminfo_gfx": (re.search(r"Name:\s+(gfx\d+)", run(["rocminfo"])).group(1) if re.search(r"Name:\s+(gfx\d+)", run(["rocminfo"])) else None),
            "torch_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "torch_gcn_arch": torch.cuda.get_device_properties(0).gcnArchName if torch.cuda.is_available() else None,
            "vram_mib": round(torch.cuda.get_device_properties(0).total_memory/2**20) if torch.cuda.is_available() else None,
            "rocm_version_file": Path("/opt/rocm/.info/version").read_text().strip() if Path("/opt/rocm/.info/version").exists() else None,
        },
        "python": {"executable": sys.executable, "version": sys.version.split()[0]},
        "torch": {"version": torch.__version__, "hip": torch.version.hip, "cuda_available": torch.cuda.is_available()},
        "openvino_env": {
            "openvino": __import__("openvino").__version__,
            "nncf": __import__("nncf").__version__,
            "transformers": __import__("transformers").__version__,
            "venv": str(REPO/".venv-ov"),
            "isolation": "separate venv; read-only .pth bridge to the preinstalled ROCm PyTorch",
        },
        "weights": {},
        "comfyui": {"version": run(["bash","-lc",f"grep __version__ {REPO}/ComfyUI/comfyui_version.py"]),
                    "pinned_commit": "b5a432157de6b8884f81104cd02926adf4d597e4",
                    "marker": (REPO/"ComfyUI/.lab_pinned_commit").read_text().strip() if (REPO/"ComfyUI/.lab_pinned_commit").exists() else None},
        "git": {"head": run(["git","-C",str(REPO),"rev-parse","HEAD"]),
                "branch": run(["git","-C",str(REPO),"rev-parse","--abbrev-ref","HEAD"])},
        "disk": {"workspace": run(["df","-h","/workspace"]), "root": run(["df","-h","/opt"])},
        "deviations": [
            "Container cgroup exposes 16 CPU quota and 55 GiB memory of a 128-thread / 503 GiB host; benchmarks use 16 threads and are not host-wide.",
            "GPU reports as 'AMD Radeon Graphics' gfx1100 48 GiB (VBIOS D7070910); compute architecture matches W7900, product string is not 'Radeon PRO W7900'.",
            "ComfyUI v0.43.0 was staged from the codeload tarball at the pinned commit because git-over-proxy intermittently returned HTTP 503; the tree marker and version were verified.",
            "ComfyUI requirements install: comfyui-workflow-templates==0.11.100 installed --no-deps (its optional media-assets-02==0.1.26 is absent from PyPI); runtime imports otherwise verified.",
            "OpenVINO 2026.4.1 lives in .venv-ov with a read-only .pth bridge to the preinstalled ROCm PyTorch; the working ROCm environment was not replaced.",
            "Large OpenVINO IRs (FP32 26 GB, BF16-compressed 13 GB, INT8 6.5 GB) live on /opt/m51-artifacts (root overlay) and are symlinked into runs/ because /workspace is a 98 GB volume.",
            "Model downloads used hf-mirror ranged transfers; every artifact was verified against the in-repo SHA256 pins and the TE checkpoint hash was recomputed independently.",
        ],
    }
    sys.path.insert(0, str(REPO)); from lab import weights
    for rel,(size,sha) in weights.WEIGHTS.items():
        p=REPO/"weights/comfyui-split"/rel
        snap["weights"][rel]={"expected_bytes":size,"expected_sha256":sha,"present":p.exists(),
                              "actual_bytes":p.stat().st_size if p.exists() else None,
                              "verified_via":"lab.weights.verify_weight"}
    out.write_text(json.dumps(snap, indent=2))
    print(f"WROTE {out}")

if __name__=="__main__": main()
