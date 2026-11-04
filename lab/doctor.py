"""Doctor: environment preflight (fail-closed; run before any measurement).

Two modes:
- default (strict): the gate for formally reproducing the reference
  campaign — requires the full six-file pinned weight set, a single
  gfx1100/W7900-class GPU, and (unless --quick) live SDPA + rocprofv3
  probes.
- ``--cloud`` (Radeon Cloud staged mode): the gate for starting the course
  on a freshly provisioned cloud instance. Weights are checked against the
  STAGE-1 set needed for the first smoke image (turbo-int8 DiT + text
  encoder + VAE); other AMD Radeon GPUs produce explicit WARN lines
  (non-blocking) instead of silent substitution; the remaining variants are
  reported as PENDING with the exact download command.

The doctor output goes to ``runs/doctor.json`` — the shipped
``evidence/results/doctor.json`` is part of the frozen official campaign and
is never overwritten.
"""

from __future__ import annotations

import json
import subprocess
import sys

from . import lab_env

# Minimum pinned set for the first bounded smoke image (BEST config):
# turbo-int8 DiT + 8B text encoder + VAE ≈ 25.5 GB.
CLOUD_STAGE1_WEIGHTS = (
    "diffusion_models/qwen_image_2.1_turbo_int8_convrot.safetensors",
    "text_encoders/qwen3vl_8b_bf16.safetensors",
    "vae/qwen_image_2.1_vae_bf16.safetensors",
)


def _weights_report(required: tuple[str, ...] | None) -> tuple[bool, list[dict]]:
    """Per-file pinned-digest check. Returns (all_required_ok, per_file).

    Present-but-wrong files are always a hard FAIL. Missing files are a hard
    FAIL only when they are in the required set (all six by default, the
    stage-1 subset with --cloud).
    """
    from . import weights as weights_mod

    required = set(required) if required is not None else set(weights_mod.WEIGHTS)
    ok = True
    report = []
    for rel in weights_mod.WEIGHTS:
        r = weights_mod.verify_weight(rel)
        entry = {"rel": rel, "ok": r["ok"], "reason": r["reason"],
                 "required": rel in required}
        if not r["ok"]:
            if rel in required:
                ok = False
                entry["verdict"] = "FAIL (required, " + r["reason"] + ")"
            else:
                entry["verdict"] = "PENDING (optional for this stage, " + \
                    r["reason"] + ")"
        else:
            entry["verdict"] = "PASS"
        report.append(entry)
    return ok, report


def run(quick: bool = False, cloud: bool = False) -> int:
    checks: list[dict] = []
    warnings: list[str] = []

    def add(name, ok, detail="", warn=False):
        checks.append({"check": name,
                       "status": "WARN" if warn else ("PASS" if ok else "FAIL"),
                       "detail": str(detail)[:400]})
        if warn and detail:
            warnings.append(f"{name}: {detail}")

    snap = lab_env.build_snapshot()
    if cloud:
        problems, gpu_warnings = lab_env.classify_gpu_expectations(snap)
        add("gpu_amd_single_working", not problems,
            problems or f"{snap.get('gpu_name')} / {snap.get('gfx_arch')} / "
            f"{snap.get('total_vram_mb')}MB")
        for w in gpu_warnings:
            add("gpu_reference_class", True, w, warn=True)
    else:
        problems = lab_env.check_gpu_expectations(snap)
        add("gpu_gfx1100_single", not problems, problems or
            f"{snap.get('gpu_name')} / {snap.get('gfx_arch')} / "
            f"{snap.get('total_vram_mb')}MB")
    add("torch_rocm", bool(snap.get("cuda_available")),
        f"torch {snap.get('torch')} hip {snap.get('hip')}")
    add("comfyui_pinned", snap.get("comfyui", {}).get("tree_marker_ok"),
        f"v{snap.get('comfyui', {}).get('version')} @ "
        f"{snap.get('comfyui', {}).get('pinned_commit', '')[:12]}")

    required = CLOUD_STAGE1_WEIGHTS if cloud else None
    weights_ok, weights_detail = _weights_report(required)
    if cloud:
        add("weights_stage1_integrity", weights_ok,
            {"per_file": weights_detail,
             "note": "stage-1 set (first smoke image); run "
                     "`bash scripts/cloud/start.sh --all-models` for the "
                     "full six-file set (~61 GB)"})
    else:
        add("weights_integrity", weights_ok,
            {"per_file": weights_detail,
             "note": "sha256 vs pinned digests (lab/weights.py); full hash "
                     "on first check, size+mtime-keyed stamp cache after"})
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

    hard_fail = any(c["status"] == "FAIL" for c in checks)
    verdict = "FAIL" if hard_fail else ("PASS_WITH_WARNINGS" if warnings else "PASS")
    env_ref = lab_env.write_snapshot()
    out = {"verdict": verdict, "mode": "cloud-staged" if cloud else "strict",
           "checks": checks, "snapshot": env_ref}
    p = lab_env.STUDENT_RUNS_ROOT / "doctor.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    for c in checks:
        print(f"{c['status']:4s} {c['check']}: {c['detail'] if isinstance(c['detail'], str) else json.dumps(c['detail'])[:400]}")
    print(f"\nverdict: {out['verdict']}  evidence: {p}")
    if cloud and verdict != "FAIL":
        print("\ncloud-staged doctor: enough to generate the FIRST image "
              "(turbo-int8 smoke).")
        print("formal benchmarking still requires the strict doctor: "
              "`python -m lab doctor`")
    return 0 if verdict != "FAIL" else 1


def _prompts_ok() -> bool:
    import json

    from .core import PROMPTS_PATH

    d = json.loads(PROMPTS_PATH.read_text(encoding="utf-8"))
    ids = [p["id"] for p in d["prompts"]]
    return (d["seeds"] == [42, 1337, 20261030]
            and ids == ["p1-zh-scene", "p2-en-scene", "p3-poster-text"])
