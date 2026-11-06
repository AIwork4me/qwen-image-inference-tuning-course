#!/usr/bin/env python3
"""Execute the course notebooks in real Jupyter kernels (nbclient/ipykernel).

Pass A (CPU CI):       --mode REFERENCE           (no GPU, no network)
Pass B (classroom):    --mode GPU_CLASSROOM       (physical W7900 exercises)
Pass C (research):     --mode GPU_RESEARCH        (full protocol)

Writes one row per notebook into an execution matrix JSON and keeps the
executed notebooks (with outputs) under <campaign>/executed/.

    .venv-lab/bin/python scripts/notebook-qualification/run_all.py \
        --mode GPU_CLASSROOM --campaign runs/notebook-gpu-qualification/<id> \
        [--only 01,03] [--timeout-s 3600]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CATALOG = json.loads((ROOT / "notebooks/catalog.json").read_text(encoding="utf-8"))

DEFAULT_TIMEOUTS = {"REFERENCE": 600, "GPU_CLASSROOM": 5400, "GPU_RESEARCH": 14400}


def kernel_identity() -> dict:
    r = subprocess.run([sys.executable, "-c",
                        "import sys,ipykernel,jupyter_client;"
                        "print(sys.version.split()[0]);print(ipykernel.__version__);"
                        "print(jupyter_client.__version__)"],
                       capture_output=True, text=True, timeout=60)
    py, ik, jc = (r.stdout.split() + ["?", "?", "?"])[:3]
    return {"python": py, "ipykernel": ik, "jupyter_client": jc,
            "executable": sys.executable}


def run_one(nb_path: Path, mode: str, timeout_s: int, workdir: Path,
            resources_dir: Path) -> dict:
    import nbformat
    from nbclient import NotebookClient

    nb = nbformat.read(str(nb_path), as_version=4)
    client = NotebookClient(
        nb, timeout=timeout_s, kernel_name="python3",
        resources={"metadata": {"path": str(workdir)}},
        allow_errors=False,
    )
    t0 = time.time()
    row = {"notebook": nb_path.name, "mode": mode,
           "started": datetime.now().astimezone().isoformat(timespec="seconds")}
    try:
        client.execute()
        row["status"] = "PASSED"
    except Exception as exc:  # noqa: BLE001 - record any client failure
        row["status"] = "FAILED"
        row["error"] = repr(exc)[:2000]
        # keep the partially executed notebook for diagnosis
    row["wall_s"] = round(time.time() - t0, 1)
    row["finished"] = datetime.now().astimezone().isoformat(timespec="seconds")
    out = resources_dir / nb_path.name
    out.write_text(nbformat.writes(nb), encoding="utf-8")
    try:
        row["executed_artifact"] = str(out.relative_to(ROOT))
    except ValueError:
        row["executed_artifact"] = str(out)
    # extract trailing stderr of the failing cell if any
    for cell in nb.cells:
        for o in cell.get("outputs", []):
            if o.get("output_type") == "error":
                row["error_cell"] = cell.get("execution_count")
                row["error_name"] = o.get("ename")
                row["error_value"] = (o.get("evalue") or "")[:800]
                break
        if "error_name" in row:
            break
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True,
                    choices=["REFERENCE", "GPU_CLASSROOM", "GPU_RESEARCH"])
    ap.add_argument("--campaign", default=None,
                    help="campaign dir (runs/notebook-gpu-qualification/<id>)")
    ap.add_argument("--only", default=None,
                    help="comma-separated module indices or filenames")
    ap.add_argument("--timeout-s", type=int, default=None)
    ap.add_argument("--workdir", default=str(ROOT))
    ap.add_argument("--out", default=None)
    ns = ap.parse_args()

    timeout = ns.timeout_s or DEFAULT_TIMEOUTS[ns.mode]
    campaign = Path(ns.campaign) if ns.campaign else \
        ROOT / "runs" / f"notebook-exec-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    exec_dir = campaign / "executed" / ns.mode
    exec_dir.mkdir(parents=True, exist_ok=True)

    only = set(ns.only.split(",")) if ns.only else None
    rows = []
    for mod in CATALOG["modules"]:
        for fname in (mod["english"], mod["chinese"]):
            key = mod["id"][-2:] if mod["id"][-2:].isdigit() else mod["id"]
            if only and not (fname in only or key in only
                             or fname.replace(".ipynb", "") in only):
                continue
            path = ROOT / "notebooks" / fname
            print(f"--- executing {fname} [{ns.mode}] timeout={timeout}s",
                  flush=True)
            row = run_one(path, ns.mode, timeout, Path(ns.workdir), exec_dir)
            rows.append(row)
            print(f"    -> {row['status']} ({row['wall_s']}s) "
                  f"{row.get('error_name', '')} {row.get('error_value', '')[:120]}",
                  flush=True)
            matrix_path = Path(ns.out) if ns.out else campaign / "notebook_execution_matrix.json"
            matrix_path.parent.mkdir(parents=True, exist_ok=True)
            matrix_path.write_text(json.dumps({
                "mode": ns.mode, "campaign": str(campaign),
                "kernel": kernel_identity(),
                "finished_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "rows": rows}, indent=2, ensure_ascii=False), encoding="utf-8")
    failed = [r for r in rows if r["status"] != "PASSED"]
    print(f"\n{len(rows) - len(failed)}/{len(rows)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
