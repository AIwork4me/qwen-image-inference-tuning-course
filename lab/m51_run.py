"""H3_OV_* experiment runner: explicit, optional, and honest about blockers.

P0-A release-gate semantics: a successful encoder execution reports
encoder_runtime=PASS, never an automatic overall accuracy/release PASS. The
result carries a gates dict (environment, artifact_integrity, encoder_runtime,
token_contract, conditioning_alignment, e2e_gpu, performance, human_quality)
plus a derived release_status. Missing evidence stays NOT_RUN; a human-quality
verdict can only come from an explicit human record (settle_human_quality) —
for H3_OV_BF16 this may be the committed owner verdict
(docs/release-closure/PR2_OWNER_QUALITY_VERDICT.json), applied only when the
record's declared scope covers the experiment's arm.

P0-B: admission runs byte-level verify_artifact() (streaming SHA256 vs the
conversion manifest, size checks, identity checks) once per run — never per
inference request.

P1-A: RAM admission uses conservative_ram_gate() (host MemAvailable and
cgroup limit minus non-reclaimable anon, minus safety headroom), not the
cgroup ceiling.
"""
from __future__ import annotations
import json, os, shutil, subprocess, time
from pathlib import Path

from . import lab_env
from .m51_contract import (artifact_status, budget_check, classify_admission,
                           conservative_ram_gate, host_mem_available_bytes,
                           read_cgroup_mem, release_status, settle_human_quality,
                           validate_result_schema, verify_artifact)

REPO = lab_env.REPO
OV_PY = REPO / ".venv-ov" / "bin" / "python"


def artifact_root() -> Path:
    """Single documented artifact-root resolution policy (P1-B):
    $M51_OV_ARTIFACTS, else <repo>/runs/m51-ov-artifacts."""
    return Path(os.environ.get("M51_OV_ARTIFACTS", REPO / "runs" / "m51-ov-artifacts"))


ARTIFACT_ROOT = artifact_root()
TOKEN_CONTRACT = REPO / "lab" / "m51_token_contract.json"
# Reference conditioning hashes from the frozen cleanroom campaign (same
# artifacts). Used for the conditioning REPRODUCIBILITY check; the numerical
# alignment vs C0 remains gate-scoped to the cleanroom run.
REFERENCE_PROBES = {
    "H3_OV_FP32": REPO / "runs/m51-20261105-cleanroom-01/ov/run_fp32/probe_C1a_OV_FP32.json",
    "H3_OV_BF16": REPO / "runs/m51-20261105-cleanroom-01/ov/run_bf16_compressed/probe_C1b_OV_BF16_COMPRESSED.json",
    "H3_OV_INT8": REPO / "runs/m51-20261105-cleanroom-01/ov/run_int8/probe_C2_OV_INT8.json",
}
# Committed numerical verdict of the frozen cleanroom alignment gate
# (C1a FP32 PASS; C1b deployment BF16 PASS; C2 INT8 FAIL on the empty
# negative at relL2 28.25%). A reproducible encoder never overrides a
# declared accuracy failure: hash-repro AND committed verdict must both PASS.
REFERENCE_VERDICT_KEYS = {
    "H3_OV_FP32": "C1a",
    "H3_OV_BF16": "C1b_compressed",
    "H3_OV_INT8": "C2",
}
REFERENCE_ALIGNMENT = REPO / "runs/m51-20261105-cleanroom-01/alignment/conditioning_alignment.json"
# Committed owner human-quality verdict (docs/release-closure/PR2_OWNER_QUALITY_
# VERDICT.json). Applied ONLY to experiments whose arm falls inside the
# record's declared scope; every other experiment keeps human_quality=PENDING.
OWNER_VERDICT = REPO / "docs/release-closure/PR2_OWNER_QUALITY_VERDICT.json"
OWNER_VERDICT_SCOPE = {"H3_OV_BF16": "M5.1 C1b deployment vs G_native"}


def load_owner_verdict(experiment: str) -> dict | None:
    """Return the committed owner verdict record iff it exists and its
    declared scope covers this experiment's arm. Anything missing, malformed
    or out-of-scope keeps the gate PENDING (never guesses a verdict)."""
    if not OWNER_VERDICT.is_file() or experiment not in OWNER_VERDICT_SCOPE:
        return None
    try:
        rec = json.loads(OWNER_VERDICT.read_text())
    except ValueError:
        return None
    if not isinstance(rec, dict):
        return None
    if rec.get("scope") != OWNER_VERDICT_SCOPE[experiment]:
        return None
    if not rec.get("reviewer") or str(rec.get("verdict", "")).upper() not in ("PASS", "FAIL"):
        return None
    return {"reviewer": rec["reviewer"], "verdict": str(rec["verdict"]).upper(),
            "date": rec.get("date"), "scope": rec.get("scope"),
            "record_path": str(OWNER_VERDICT.relative_to(REPO))}
SPECS = {
    "H3_OV_FP32": {"dir": "fp32", "xml": "language_model_fp32.xml", "precision": "f32", "group": None,
                   "identity": {"dtype": "fp32"}, "need_ram": 45 * 2**30},
    "H3_OV_BF16": {"dir": "bf16", "xml": "language_model_bf16.xml", "precision": "bf16", "group": None,
                   "identity": {"storage": "FP16", "runtime_precision": "bf16"}, "need_ram": 34 * 2**30},
    "H3_OV_INT8": {"dir": "int8", "xml": "language_model_int8.xml", "precision": "f32", "group": 32,
                   "identity": {"mode": "INT8_ASYM"}, "need_ram": 20 * 2**30},
}


def _blocked(out: Path, arm: str, reason: str, detail: dict, gates: dict | None = None) -> int:
    gates = gates or {}
    gates.setdefault("environment", "NOT_RUN")
    for k in ("artifact_integrity", "encoder_runtime", "token_contract",
              "conditioning_alignment", "e2e_gpu", "performance"):
        gates.setdefault(k, "NOT_RUN")
    gates.setdefault("human_quality", "PENDING")
    rec = {"experiment": arm, "status": "BLOCKED", "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "source_head": subprocess.run(["git","-C",str(REPO),"rev-parse","HEAD"],capture_output=True,text=True).stdout.strip(),
           "artifact_digest": None, "environment": {"python": str(OV_PY)},
           "measurements": {}, "evidence_paths": [str(out)],
           "gates": gates, "release_status": release_status(gates),
           "blocker": {"reason": reason, "detail": detail,
                       "next_action": "run: bash scripts/m51/provision_ov.sh"}}
    out.mkdir(parents=True, exist_ok=True); (out/"H3_RESULTS.json").write_text(json.dumps(rec,indent=2))
    print(f"BLOCKED {arm}: {reason}\n{json.dumps(detail,indent=1)}\nnext: bash scripts/m51/provision_ov.sh")
    return 2


def run_h3(experiment: str, smoke: bool = False, date: str | None = None, port: int = 8188, threads: int = 16,
           apply_owner_verdict: bool = False) -> int:
    """Run one H3 OpenVINO TE experiment.

    apply_owner_verdict=False (default): a NEW physical run never inherits the
    committed C1b owner human-quality acceptance — that verdict covers the
    historically evaluated deployment only. New runs keep human_quality
    PENDING and cite the historical verdict as reference. The frozen campaign
    records already carry their own scoped verdicts and are never rewritten.
    """
    spec = SPECS[experiment]
    date = date or time.strftime("%Y%m%d-%H%M%S")
    out = REPO / "runs" / f"{date}-{experiment}{'-smoke' if smoke else ''}"
    out.mkdir(parents=True, exist_ok=True)
    gates: dict[str, str] = {}

    # Gate: environment admission (GPU class; GO_WITH_DEVIATIONS is admissible)
    s0 = lab_env.build_snapshot()
    snap = {"gpu": {"gfx_arch": s0.get("gfx_arch"), "vram_mib": s0.get("total_vram_mb")},
            "torch": {"cuda_available": s0.get("cuda_available"), "device_count": s0.get("device_count")}}
    admission, why = classify_admission(snap)
    gates["environment"] = "BLOCKED" if admission == "BLOCKED_ENVIRONMENT" else "PASS"
    if gates["environment"] == "BLOCKED":
        return _blocked(out, experiment, "environment_admission", {"admission": admission, "reasons": why}, gates)

    # Gate: byte-level artifact integrity (P0-B) — admission time, once.
    model_dir = ARTIFACT_ROOT / spec["dir"]
    files = [spec["xml"], spec["xml"].replace(".xml", ".bin")]
    status = verify_artifact(model_dir, files, expected_identity=spec["identity"])
    if status["status"] == "BLOCKED":
        gates["artifact_integrity"] = "BLOCKED"
        return _blocked(out, experiment, status.get("reason", "artifact_invalid"), status, gates)
    gates["artifact_integrity"] = "PASS"
    if status["status"] == "PASS_WITH_WARNINGS":
        gates["artifact_integrity"] = "PASS"  # warnings recorded, not silent
    if not OV_PY.exists():
        gates["artifact_integrity"] = "BLOCKED"
        return _blocked(out, experiment, "missing_openvino_venv", {"path": str(OV_PY)}, gates)
    probe = subprocess.run([str(OV_PY), "-c", "import openvino; print(openvino.__version__)"],
                           capture_output=True, text=True, timeout=120)
    if probe.returncode != 0:
        return _blocked(out, experiment, "openvino_import_failed", {"stderr": probe.stderr[-800:]}, gates)

    # P1-A: conservative RAM admission (host MemAvailable + cgroup anon accounting)
    ram = conservative_ram_gate(read_cgroup_mem(), host_mem_available_bytes(), spec["need_ram"])
    free_disk = shutil.disk_usage(out).free
    budget = budget_check(free_disk, 2*2**30, ram["usable_bytes"], spec["need_ram"])
    if budget["status"] != "PASS" or ram["status"] != "PASS":
        return _blocked(out, experiment, "resource_budget",
                        {"ram_gate": ram, "disk_free_bytes": free_disk, "disk_need_bytes": 2*2**30}, gates)

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
        gates["encoder_runtime"] = "FAIL"
        return _blocked(out, experiment, "encoder_run_failed", {"rc": r.returncode, "stderr": r.stderr[-1500:]}, gates)
    d = json.loads(pf.read_text())
    finite = all(row["finite"] for rows in d["prompts"].values() for row in rows)
    on_cpu = d.get("meta", {}).get("execution_devices") == ["CPU"]
    gates["encoder_runtime"] = "PASS" if (finite and on_cpu) else "FAIL"

    # Gate: token contract — live probe IDs must equal the pinned production
    # contract (drift anywhere in tokenizer/prompt handling is a FAIL).
    contract = json.loads(TOKEN_CONTRACT.read_text()) if TOKEN_CONTRACT.is_file() else None
    if contract is None:
        gates["token_contract"] = "NOT_RUN"
        token_detail = {"reason": "lab/m51_token_contract.json missing"}
    else:
        drift = {}
        for pid, rows in d["prompts"].items():
            want = contract["prompts"].get(pid, {}).get("ids")
            got = rows[0].get("ids")
            if want is not None and list(got or []) != list(want):
                drift[pid] = {"expected": want, "got": got}
        gates["token_contract"] = "PASS" if not drift else "FAIL"
        token_detail = drift or {"checked": sorted(d["prompts"])}

    # Gate: conditioning reproducibility vs the frozen cleanroom reference,
    # combined with the committed cleanroom numerical verdict. Hash-repro
    # alone can never promote a declared accuracy failure (INT8 negative).
    ref_path = REFERENCE_PROBES.get(experiment)
    verdict_key = REFERENCE_VERDICT_KEYS.get(experiment)
    committed = None
    if REFERENCE_ALIGNMENT.is_file() and verdict_key:
        try:
            committed = json.loads(REFERENCE_ALIGNMENT.read_text())["verdict"][verdict_key]["status"]
        except (KeyError, ValueError):
            committed = None
    if ref_path and ref_path.is_file():
        ref = json.loads(ref_path.read_text())
        mismatch = {}
        for pid, rows in d["prompts"].items():
            want = next((r["cond_sha256"] for r in ref.get("prompts", {}).get(pid, [])), None)
            got = rows[0].get("cond_sha256")
            if want and got and want != got:
                mismatch[pid] = {"reference": want, "got": got}
        if mismatch:
            gates["conditioning_alignment"] = "FAIL"
        elif committed == "FAIL":
            gates["conditioning_alignment"] = "FAIL"
        elif committed == "PASS":
            gates["conditioning_alignment"] = "PASS"
        else:
            gates["conditioning_alignment"] = "NOT_RUN"
        cond_detail = {"hash_repro": "reproduced byte-identical vs runs/m51-20261105-cleanroom-01"
                       if not mismatch else mismatch,
                       "committed_cleanroom_verdict": committed,
                       "note": "hash-repro proves determinism; the numerical verdict comes from "
                               "the cleanroom alignment vs C0 and is never overridden here"}
    else:
        gates["conditioning_alignment"] = "NOT_RUN"
        cond_detail = {"reason": "no committed reference probe for this arm"}

    gates["e2e_gpu"] = "NOT_RUN"  # scope: separate E2E experiment, never implied here
    measured = [row["encode_s"] for rows in d["prompts"].values() for row in rows]
    gates["performance"] = "PASS" if measured else "NOT_RUN"
    owner_verdict = load_owner_verdict(experiment) if apply_owner_verdict else None
    settle_human_quality(gates, owner_verdict)  # PASS only from an explicit, scope-matched owner record
    historical = None if apply_owner_verdict else load_owner_verdict(experiment)
    if historical is not None:
        gates["human_quality"] = "PENDING"  # a new run needs its own human review
        historical = dict(historical)
        historical["note"] = ("historical C1b owner acceptance is REFERENCE ONLY "
                              "for new physical runs; it does not approve this deployment")
    rel = release_status(gates)

    exec_status = "PASS" if finite else "FAIL"
    rec = {"experiment": experiment, "status": exec_status,
           "gates": gates, "release_status": rel,
           "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "source_head": subprocess.run(["git","-C",str(REPO),"rev-parse","HEAD"],capture_output=True,text=True).stdout.strip(),
           "artifact_digest": next((f["sha256"] for f in status.get("files", {}).values() if not f.get("unpinned")), None),
           "artifact_verification": status, "artifact_manifest": status.get("manifest"),
           "environment": {"openvino": probe.stdout.strip(), "admission": admission, "admission_reasons": why,
                           "ram_gate": ram},
           "measurements": {"encode_s": measured, "load_s": d.get("load_s"), "rss_peak_mb": d.get("rss_peak_mb"),
                            "shape": [row["shape"] for rows in d["prompts"].values() for row in rows],
                            "n_tokens_full": [row["n_tokens_full"] for rows in d["prompts"].values() for row in rows]},
            "token_contract_detail": token_detail, "conditioning_alignment_detail": cond_detail,
            "human_quality_reference": historical,
           "evidence_paths": [str(pf), str(out/"stdout.log"), str(out/"stderr.log")],
           "mode": "smoke" if smoke else "formal", "runtime_s": round(time.time()-t0,2)}
    errs = validate_result_schema(rec)
    if errs: rec["schema_errors"] = errs
    (out/"H3_RESULTS.json").write_text(json.dumps(rec,indent=2))
    print(json.dumps({k: rec[k] for k in ("experiment","status","gates","release_status","mode","runtime_s")}, indent=1)[:3000])
    print(f"H3 results: {out/'H3_RESULTS.json'}")
    return 0 if exec_status == "PASS" else 1
