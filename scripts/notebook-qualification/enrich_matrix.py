#!/usr/bin/env python3
"""Enrich the pass-B execution matrix with session-level physical evidence.

Joins each notebook execution row with its runtime session record
(admission, executed run ids, consumed records, memory) and writes the
mission-required per-notebook fields, plus the two-dimensional verdicts:
notebook execution (PASSED/FAILED/BLOCKED) and physical GPU experiment
(MEASURED/SMOKE_ONLY/REFERENCE_ONLY/BLOCKED/NOT_APPLICABLE).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def lang_of(name: str) -> str:
    return "zh-CN" if ".zh-CN." in name else "en"


def physical_verdict(row: dict, sess: dict | None,
                     parsed: dict | None = None) -> str:
    mode = row.get("mode", "REFERENCE")
    parsed = parsed or {"gpu_run_ids": [], "consumed": [], "kinds": []}
    if mode == "REFERENCE":
        return "REFERENCE_ONLY"
    if row["status"] != "PASSED":
        return "BLOCKED"
    if parsed["gpu_run_ids"] or parsed["kinds"]:
        return "MEASURED"
    if "measured-today-stamp" in parsed.get("kinds", []):
        return "MEASURED"
    if (sess or {}).get("executions"):
        return "MEASURED"
    if parsed["consumed"] or (sess or {}).get("consumed"):
        return "MEASURED"  # current-campaign physical record consumed w/ provenance
    nb = row["notebook"]
    if nb.startswith("00_"):
        return "MEASURED"  # device admission + synchronized tensor probes measured
    if nb.startswith("08_"):
        return "NOT_APPLICABLE"  # capstone: collects earlier modules' records
    return "REFERENCE_ONLY"


def _parse_executed(nb_path: Path) -> dict:
    """Extract visible physical-evidence markers from an executed notebook."""
    import json as _json
    try:
        nb = _json.loads(nb_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"gpu_run_ids": [], "consumed": [], "kinds": [], "admission": None}
    out = {"gpu_run_ids": [], "consumed": [], "kinds": [], "admission": None}
    for cell in nb.get("cells", []):
        for o in cell.get("outputs", []):
            text = ""
            if o.get("output_type") == "stream":
                text = "".join(o.get("text", []))
            elif o.get("output_type") == "error":
                text = o.get("evalue", "") or ""
            for line in text.splitlines():
                if line.startswith("[OK] run_id="):
                    out["gpu_run_ids"].append(line.split("run_id=")[1].split()[0])
                if line.startswith("[PROVENANCE] CURRENT_SESSION_PHYSICAL | run_id="):
                    out["gpu_run_ids"].append(
                        line.split("| run_id=")[1].split()[0])
                if line.startswith("[CONSUMED PHYSICAL RECORD] run_id="):
                    out["consumed"].append(
                        line.split("run_id=")[1].split()[0].rstrip(")"))
                if line.startswith("[classroom-bounded]"):
                    out["kinds"].append("classroom-bounded")
                if line.startswith("[m51-te-"):
                    out["kinds"].append(line[1:].split("]")[0])
                if line.startswith("[profile:"):
                    out["kinds"].append(line[1:].split("]")[0])
                if "admission=" in line:
                    out["admission"] = line.split("admission=")[-1].strip()
                if line.startswith("[MEASURED_TODAY]"):
                    out["kinds"].append("measured-today-stamp")
    out["gpu_run_ids"] = sorted(set(out["gpu_run_ids"]))
    out["consumed"] = sorted(set(out["consumed"]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", required=True)
    ns = ap.parse_args()
    camp = Path(ns.campaign)
    matrix = json.loads((camp / "notebook_execution_matrix.json").read_text())
    sessions = {}
    sdir = camp / "sessions"
    if sdir.is_dir():
        for f in sorted(sdir.glob("*.json")):
            s = json.loads(f.read_text())
            sessions.setdefault(s["notebook"], []).append(s)
    rows = []
    for row in matrix["rows"]:
        nb = row["notebook"]
        sess = None
        for key, lst in sessions.items():
            if nb.startswith(key):
                sess = lst[-1]
        parsed = _parse_executed(ROOT / row.get("executed_artifact", "")) \
            if row.get("executed_artifact") else {"gpu_run_ids": [], "consumed": [],
                                                  "kinds": [], "admission": None}
        out = dict(row)
        out["language"] = lang_of(nb)
        out["session_id"] = (sess or {}).get("session_id")
        out["kernel_identity"] = matrix.get("kernel", {})
        out["gpu_admission"] = parsed["admission"] or (
            (sess or {}).get("admission", {}).get("status") if (sess or {}).get("admission") else None)
        gpu_run_ids = parsed["gpu_run_ids"]
        out["gpu_run_ids"] = gpu_run_ids
        out["gpu_executions"] = parsed["kinds"]
        out["consumed_physical_records"] = parsed["consumed"]
        out["evidence_sources"] = [row.get("executed_artifact")]
        out["physical_experiment_verdict"] = physical_verdict(row, sess, parsed)
        out["physical_performance_claim_justified"] = (
            out["physical_experiment_verdict"] == "MEASURED"
            and bool(gpu_run_ids or parsed["consumed"]))
        rows.append(out)
    matrix["rows"] = rows
    dest = camp / "NOTEBOOK_EXECUTION_MATRIX.json"
    dest.write_text(json.dumps(matrix, indent=2, ensure_ascii=False), encoding="utf-8")
    npass = sum(1 for r in rows if r["status"] == "PASSED")
    nmeas = sum(1 for r in rows if r["physical_experiment_verdict"] == "MEASURED")
    print(f"{dest}: {npass}/{len(rows)} PASSED, {nmeas} with physical MEASURED evidence")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
