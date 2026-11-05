"""H3_OV_* experiment runner: explicit, optional, and honest about blockers."""
from __future__ import annotations
import json, os, shutil, subprocess, time
from pathlib import Path

from . import lab_env
from .m51_contract import artifact_status, budget_check, classify_admission, validate_result_schema

REPO = lab_env.REPO
OV_PY = REPO / ".venv-ov" / "bin" / "python"
ARTIFACT_ROOT = Path(os.environ.get("M51_OV_ARTIFACTS", REPO / "runs" / "m51-ov-artifacts"))
SPECS = {
    "H3_OV_FP32": {"dir": "fp32", "xml": "language_model_fp32.xml", "precision": "f32", "group": None},
    "H3_OV_BF16": {"dir": "bf16", "xml": "language_model_bf16.xml", "precision": "bf16", "group": None},
    "H3_OV_INT8": {"dir": "int8", "xml": "language_model_int8.xml", "precision": "f32", "group": 32},
}

def _blocked(out: Path, arm: str, reason: str, detail: dict) -> int:
    rec = {"experiment": arm, "status": "BLOCKED", "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "source_head": subprocess.run(["git","-C",str(REPO),"rev-parse","HEAD"],capture_output=True,text=True).stdout.strip(),
           "artifact_digest": None, "environment": {"python": str(OV_PY)},
           "measurements": {}, "evidence_paths": [str(out)],
           "blocker": {"reason": reason, "detail": detail,
                       "next_action": "run: bash scripts/m51/provision_ov.sh"}}
    out.mkdir(parents=True, exist_ok=True); (out/"H3_RESULTS.json").write_text(json.dumps(rec,indent=2))
    print(f"BLOCKED {arm}: {reason}\n{json.dumps(detail,indent=1)}\nnext: bash scripts/m51/provision_ov.sh")
    return 2

def run_h3(experiment: str, smoke: bool = False, date: str | None = None, port: int = 8188, threads: int = 16) -> int:
    spec = SPECS[experiment]
    date = date or time.strftime("%Y%m%d-%H%M%S")
    out = REPO / "runs" / f"{date}-{experiment}{'-smoke' if smoke else ''}"
    out.mkdir(parents=True, exist_ok=True)
    model_dir = ARTIFACT_ROOT / spec["dir"]
    status = artifact_status(model_dir, [spec["xml"], spec["xml"].replace(".xml", ".bin")])
    if status["status"] != "PASS":
        return _blocked(out, experiment, status.get("reason","artifact_invalid"), status)
    if not OV_PY.exists():
        return _blocked(out, experiment, "missing_openvino_venv", {"path": str(OV_PY)})
    probe = subprocess.run([str(OV_PY), "-c", "import openvino; print(openvino.__version__)"],
                           capture_output=True, text=True, timeout=120)
    if probe.returncode != 0:
        return _blocked(out, experiment, "openvino_import_failed", {"stderr": probe.stderr[-800:]})
    s0 = lab_env.build_snapshot()
    snap = {"gpu": {"gfx_arch": s0.get("gfx_arch"), "vram_mib": s0.get("total_vram_mb")},
            "torch": {"cuda_available": s0.get("cuda_available"), "device_count": s0.get("device_count")}}
    admission, why = classify_admission(snap)
    mem = Path("/sys/fs/cgroup/memory.max"); cur = Path("/sys/fs/cgroup/memory.current")
    # memory.current includes reclaimable page cache; use the cgroup ceiling
    # (the export/encoder working set is anonymous memory, bounded by this).
    free_ram = int(mem.read_text()) if mem.exists() and mem.read_text().strip().isdigit() else 0
    need_ram = {"fp32": 45*2**30, "bf16": 34*2**30, "int8": 20*2**30}[spec["dir"]]
    free_disk = shutil.disk_usage(out).free
    budget = budget_check(free_disk, 2*2**30, free_ram, need_ram)
    if budget["status"] != "PASS":
        return _blocked(out, experiment, "resource_budget", budget)
    cmd = [str(OV_PY), "-u", str(REPO/"lab/ov_encoder.py"), "--arm", experiment,
           "--model-dir", str(model_dir), "--xml", spec["xml"], "--precision", spec["precision"],
           "--out", str(out), "--threads", str(threads), "--repeats", "1" if smoke else "3"]
    if smoke:
        cmd += ["--prompt-ids", "p1-zh-scene"]
    if spec["group"] is not None:
        cmd += ["--dynamic-quant-group", str(spec["group"])]
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
    (out/"stdout.log").write_text(r.stdout[-50000:]); (out/"stderr.log").write_text(r.stderr[-50000:])
    pf = out / f"probe_{experiment}.json"
    if r.returncode != 0 or not pf.exists():
        return _blocked(out, experiment, "encoder_run_failed", {"rc": r.returncode, "stderr": r.stderr[-1500:]})
    d = json.loads(pf.read_text())
    finite = all(row["finite"] for rows in d["prompts"].values() for row in rows)
    measured = [row["encode_s"] for rows in d["prompts"].values() for row in rows]
    rec = {"experiment": experiment, "status": "PASS" if finite else "FAIL",
           "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "source_head": subprocess.run(["git","-C",str(REPO),"rev-parse","HEAD"],capture_output=True,text=True).stdout.strip(),
           "artifact_digest": status["manifest"].get("output_bin_sha256") or status["manifest"].get("source_sha256"),
           "artifact_manifest": status["manifest"], "environment": {"openvino": probe.stdout.strip(), "admission": admission, "admission_reasons": why},
           "measurements": {"encode_s": measured, "load_s": d.get("load_s"), "rss_peak_mb": d.get("rss_peak_mb"),
                            "shape": [row["shape"] for rows in d["prompts"].values() for row in rows],
                            "n_tokens_full": [row["n_tokens_full"] for rows in d["prompts"].values() for row in rows]},
           "evidence_paths": [str(pf), str(out/"stdout.log"), str(out/"stderr.log")],
           "mode": "smoke" if smoke else "formal", "runtime_s": round(time.time()-t0,2)}
    errs = validate_result_schema(rec)
    if errs: rec["schema_errors"] = errs
    (out/"H3_RESULTS.json").write_text(json.dumps(rec,indent=2))
    print(json.dumps({k: rec[k] for k in ("experiment","status","mode","runtime_s","measurements")}, indent=1)[:3000])
    print(f"H3 results: {out/'H3_RESULTS.json'}")
    return 0 if rec["status"] == "PASS" else 1
