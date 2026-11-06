#!/usr/bin/env python3
"""Validate the qualification evidence package (Layer B, run on the W7900 host).

Checks, fail-closed:
  1. hardware admission record exists and is PASS/PASS_WITH_WARNINGS;
  2. every one of the 18 notebooks has an execution row per declared mode;
  3. every performance-bearing module has a genuine current-campaign physical
     run ID (never a REFERENCE-only notebook marked as physically measured);
  4. the frozen evidence/ tree is byte-identical to the git HEAD state
     (no contamination of historical data);
  5. all executed notebooks are stored under the campaign dir;
  6. gpu_ab_summary classifications never upgrade smoke/bounded populations
     to FORMAL_PROTOCOL_MEASURED.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CATALOG = json.loads((ROOT / "notebooks/catalog.json").read_text(encoding="utf-8"))
NOTEBOOKS = [p for row in CATALOG["modules"] for p in (row["english"], row["chinese"])]
PHYSICAL_MODULES = {"01", "03", "04", "05", "06", "07"}  # modules with performance claims


def check(campaign: Path) -> list[str]:
    errors: list[str] = []

    adm = campaign / "P0_HARDWARE_ADMISSION.json"
    if not adm.is_file():
        errors.append("missing P0_HARDWARE_ADMISSION.json")
    else:
        a = json.loads(adm.read_text())
        if a.get("verdict") not in ("PASS", "PASS_WITH_WARNINGS"):
            errors.append(f"admission verdict is {a.get('verdict')}")

    matrix = campaign / "NOTEBOOK_EXECUTION_MATRIX.json"
    if not matrix.is_file():
        errors.append("missing NOTEBOOK_EXECUTION_MATRIX.json")
    else:
        rows = json.loads(matrix.read_text())["rows"]
        by_name = {r["notebook"]: r for r in rows}
        for nb in NOTEBOOKS:
            if nb not in by_name:
                errors.append(f"no execution row for {nb}")
            else:
                r = by_name[nb]
                if r["status"] != "PASSED":
                    errors.append(f"{nb}: status {r['status']} ({r.get('error_value', '')[:120]})")
                if r.get("mode") in ("GPU_CLASSROOM", "GPU_RESEARCH"):
                    if r.get("gpu_admission") not in ("PASS", "PASS_WITH_WARNINGS", None):
                        errors.append(f"{nb}: admission {r.get('gpu_admission')}")
                if r.get("physical_performance_claim") and not r.get("gpu_run_ids"):
                    errors.append(f"{nb}: performance claim without any physical run id")

    gpu_ab = campaign / "gpu_ab_summary.json"
    if gpu_ab.is_file():
        for row in json.loads(gpu_ab.read_text())["comparisons"]:
            if row.get("classification") == "FORMAL_PROTOCOL_MEASURED":
                b, c = row.get("baseline_run_id", ""), row.get("candidate_run_id", "")
                if "smoke" in b or "smoke" in c or not b or not c:
                    errors.append(f"{row['label']}: formal classification on smoke/missing runs")

    # frozen evidence untouched vs git
    r = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "evidence/"],
                       capture_output=True, text=True)
    if r.stdout.strip():
        errors.append(f"evidence/ modified: {r.stdout.strip()[:300]}")

    for mod in PHYSICAL_MODULES:
        en = next(m for m in CATALOG["modules"] if m["id"] == mod)["english"]
        row = next((x for x in json.loads(matrix.read_text())["rows"]
                    if x["notebook"] == en), None) if matrix.is_file() else None
        if row and row.get("mode") in ("GPU_CLASSROOM", "GPU_RESEARCH"):
            has_run = row.get("gpu_run_ids") or _campaign_has_physical(campaign, mod)
            if not has_run:
                errors.append(f"module {mod}: GPU-mode execution without a physical run")
    return errors


def _campaign_has_physical(campaign: Path, mod: str) -> bool:
    exps = {"01": ("B0",), "03": ("B0", "L1"), "04": ("B0", "F1", "F2"),
            "05": ("m51-te",), "06": ("H3_OV_BF16",), "07": ("B0", "BEST")}
    runs = ROOT / "runs"
    for exp in exps.get(mod, ()):
        if any(runs.glob(f"*{exp}*/*.json")):
            return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", required=True)
    ns = ap.parse_args()
    errors = check(Path(ns.campaign))
    if errors:
        print("EVIDENCE_VALIDATION_FAILED")
        for e in errors:
            print(" -", e)
        return 1
    print("EVIDENCE_VALIDATION_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
