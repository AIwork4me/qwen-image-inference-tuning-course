"""Read-only, standard-library helpers for bilingual Jupyter course labs.

The published evidence/ directory is immutable. Jupyter execution never edits
it. Physical ROCm smoke runs are separately opt-in, bounded, and land in runs/.
"""
from __future__ import annotations

import json
import os
import re
import statistics
import subprocess
from pathlib import Path


def repo_root(start: str | Path | None = None) -> Path:
    start = Path.cwd() if start is None else Path(start)
    start = start.resolve()
    for candidate in (start, *start.parents):
        if (candidate / "lab" / "reference_runs.json").is_file() and (
            candidate / "evidence" / "results" / "SUMMARY.json"
        ).is_file():
            return candidate
    raise FileNotFoundError(
        "Course root not found. Launch Jupyter from the cloned "
        "qwen-image-inference-tuning-course directory."
    )


def read_json(relative_path: str, root: Path | None = None) -> dict:
    root = repo_root() if root is None else Path(root).resolve()
    path = (root / relative_path).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Refusing to read outside the repository")
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def summary_rows(root: Path | None = None) -> dict[str, dict]:
    data = read_json("evidence/results/SUMMARY.json", root)
    assert data["scope"] == "reference", "Do not mix student/smoke runs into reference"
    return {row["experiment"]: row for row in data["per_experiment"]}


def reference_warm(experiment: str, root: Path | None = None) -> list[float]:
    root = repo_root() if root is None else Path(root).resolve()
    names = read_json("lab/reference_runs.json", root)["reference_run_dirs"]
    if experiment not in names:
        raise ValueError(f"Unknown official reference experiment: {experiment}")
    values: list[float] = []
    for directory in names[experiment]:
        for f in sorted((root / "evidence/results" / directory).glob("*.json")):
            rec = read_json(str(f.relative_to(root)), root)
            if rec.get("experiment") != experiment or rec.get("run_kind") != "measure":
                continue
            for run in rec.get("runs", []):
                if run.get("kind") == "warm":
                    values.append(float(run["total_s"]))
    if not values:
        raise ValueError(f"No official warm measurement for {experiment}")
    return values


def fair_factorial(root: Path | None = None, *, clean_head: bool = False) -> dict:
    path = (
        "runs/p4-fairab-cleanhead-003750/pipeline_profile/pipeline_profile.json"
        if clean_head else
        "runs/p4-fairab-225326/pipeline_profile/pipeline_profile.json"
    )
    raw = read_json(path, root)
    values: dict[str, list[float]] = {
        "A_original": [], "A_fast": [], "C_original": [], "C_fast": []
    }
    for arm in ("A", "C"):
        for session in raw["arms"][arm]["sessions"]:
            for batch in session["batches"]:
                mode = "original" if batch["mode"] == "poll" else "fast"
                values[f"{arm}_{mode}"].append(float(batch["batch_wall_s"]))
    if any(not row for row in values.values()):
        raise ValueError("Incomplete physical A/C factorial population")
    return {
        "source": path,
        "cells": {
            key: {
                "n": len(walls),
                "mean_s": statistics.mean(walls),
                "median_s": statistics.median(walls),
                "images_per_min": 240 / statistics.mean(walls),
                "walls_s": walls,
            }
            for key, walls in values.items()
        },
        "source_provenance": raw.get("environment", {}).get("source_provenance"),
    }


SAFE_SMOKE_EXPERIMENTS = frozenset({"B0", "L1", "F1", "F2", "BEST", "H3_OV_BF16"})


def gpu_smoke(experiment: str, tag: str, *, confirmed: bool = False,
              timeout_s: int = 900, root: Path | None = None) -> None:
    """Explicit GPU smoke. NEVER called automatically by a notebook or CPU CI.

    Only --smoke is used. A second environment acknowledgement is required.
    """
    if experiment not in SAFE_SMOKE_EXPERIMENTS:
        raise ValueError("Not a bounded, allowlisted GPU smoke experiment")
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,31}", tag):
        raise ValueError("Use a short lower-case run tag with digits/hyphens")
    if not confirmed or os.environ.get("QIR_NOTEBOOK_GPU_ACK") != "YES":
        raise PermissionError(
            "GPU is opt-in: set RUN_GPU=True in this cell AND explicitly set "
            "QIR_NOTEBOOK_GPU_ACK=YES after checking GPU time/model availability."
        )
    root = repo_root() if root is None else Path(root).resolve()
    python = root / ".venv-lab/bin/python"
    if not python.is_file():
        raise FileNotFoundError(
            "ROCm lab environment missing; first follow docs/RADEON_CLOUD_QUICKSTART.md"
        )
    cmd = [str(python), "-m", "lab", "run", "--experiment", experiment,
           "--smoke", "--date", tag]
    print("Opt-in physical smoke:", " ".join(cmd))
    subprocess.run(cmd, cwd=root, timeout=timeout_s, check=True)


def student_record(payload: dict, tag: str, *, confirmed: bool = False,
                   root: Path | None = None) -> Path:
    """Opt-in isolated student analysis; never modifies evidence/."""
    if not confirmed:
        raise PermissionError("Writing an exercise record must be explicitly enabled")
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,31}", tag):
        raise ValueError("Invalid run tag")
    root = repo_root() if root is None else Path(root).resolve()
    path = root / "runs" / "notebook-student" / (tag + ".json")
    if path.exists():
        raise FileExistsError(f"Append-only student output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")
    return path
