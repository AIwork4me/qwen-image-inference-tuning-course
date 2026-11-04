"""Doctor: environment preflight (fail-closed; run before any measurement)."""

from __future__ import annotations

import json
import subprocess
import sys

from . import lab_env


def run(quick: bool = False) -> int:
    checks: list[dict] = []

    def add(name, ok, detail=""):
        checks.append({"check": name, "status": "PASS" if ok else "FAIL",
                       "detail": str(detail)[:400]})

    snap = lab_env.build_snapshot()
    problems = lab_env.check_gpu_expectations(snap)
    add("gpu_gfx1100_single", not problems, problems or
        f"{snap.get('gpu_name')} / {snap.get('gfx_arch')} / {snap.get('total_vram_mb')}MB")
    add("torch_rocm", bool(snap.get("cuda_available")),
        f"torch {snap.get('torch')} hip {snap.get('hip')}")
    add("comfyui_pinned", snap.get("comfyui", {}).get("tree_marker_ok"),
        f"v{snap.get('comfyui', {}).get('version')} @ "
        f"{snap.get('comfyui', {}).get('pinned_commit', '')[:12]}")
    add("weights_integrity", _weights_ok(),
        "sha256 vs pinned digests (lab/weights.py)")
    add("prompts_frozen", _prompts_ok(), "3 prompts x 3 seeds, ids intact")

    if not quick:
        try:
            import torch

            q = torch.randn((1, 8, 128, 64), dtype=torch.bfloat16, device="cuda")
            with torch.nn.attention.sdpa_kernel(
                    torch.nn.attention.SDPBackend.FLASH_ATTENTION):
                torch.nn.functional.scaled_dot_product_attention(q, q, q)
            torch.cuda.synchronize()
            add("flash_sdpa_probe", True, "flash attention launches + syncs on gfx1100")
        except Exception as exc:  # noqa: BLE001
            add("flash_sdpa_probe", False, exc)
        try:
            r = subprocess.run(["rocprofv3", "--version"], capture_output=True,
                               text=True, timeout=20)
            add("rocprofv3", r.returncode == 0, (r.stdout + r.stderr).strip()[:200])
        except Exception as exc:  # noqa: BLE001
            add("rocprofv3", False, exc)

    env_ref = lab_env.write_snapshot()
    out = {"verdict": "PASS" if all(c["status"] == "PASS" for c in checks) else "FAIL",
           "checks": checks, "snapshot": env_ref}
    p = lab_env.RESULTS_ROOT / "doctor.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    for c in checks:
        print(f"{c['status']:4s} {c['check']}: {c['detail']}")
    print(f"\nverdict: {out['verdict']}  evidence: {p}")
    return 0 if out["verdict"] == "PASS" else 1


def _weights_ok() -> bool:
    from . import weights as weights_mod

    ok = True
    for rel in weights_mod.WEIGHTS:
        r = weights_mod.verify_weight(rel)
        ok = ok and r["ok"]
    return ok


def _prompts_ok() -> bool:
    import json

    from .core import PROMPTS_PATH

    d = json.loads(PROMPTS_PATH.read_text(encoding="utf-8"))
    ids = [p["id"] for p in d["prompts"]]
    return (d["seeds"] == [42, 1337, 20261030]
            and ids == ["p1-zh-scene", "p2-en-scene", "p3-poster-text"])
