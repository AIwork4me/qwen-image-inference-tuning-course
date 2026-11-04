"""Experiment manifests: ``lab/experiments/<ID>.json`` — one file per
configuration in the methodology matrix. Data-driven; the runner derives
every record field from its manifest.

IDs are the course matrix (never silently substituted):
B0 baseline · L1/L1O less-computation · F1/F2 faster-computation ·
LF1-LF3 combined · H1/H2 heterogeneous · BEST final.
"""

from __future__ import annotations

import json
from pathlib import Path

from . import lab_env

EXPERIMENTS_DIR = lab_env.LAB_DIR / "experiments"

REQUIRED_FIELDS = (
    "id", "category", "description", "framework", "precision",
    "resolution", "steps", "cfg", "sampler", "scheduler",
    "unet", "clip", "vae", "weight_dtype", "tiled_vae",
    "attention_backend", "measured",
)
OPTIONAL_DEFAULTS = {
    "server_flags": [],
    "sdp_mode": "none",  # normal functional Linux attention
    "env_vars": {},
    "attention_log_patterns": None,  # server.py defaults
    "fresh_prompt_arm": False,
    "scheduler_note": "",
    "enabled": True,
}

KNOWN_IDS = ("B0", "L1", "L1O", "F1", "F2", "LF1", "LF2", "LF3", "H1", "H2", "BEST")


def validate_manifest(m: dict) -> None:
    missing = [f for f in REQUIRED_FIELDS if f not in m]
    if missing:
        raise ValueError(f"manifest {m.get('id', '?')!r} missing fields: {missing}")
    if m["id"] not in KNOWN_IDS:
        raise ValueError(
            f"experiment id {m['id']!r} not in the methodology matrix {KNOWN_IDS} "
            "— never silently substitute a different optimization")
    if m["framework"] != "comfyui":
        raise ValueError("lab manifests are comfyui-only")
    if m.get("sdp_mode", "none") == "math" and m["id"] == "B0":
        raise ValueError("B0 must use the normal Linux attention (sdp_mode=none); "
                         "math forcing is a separate diagnostic variant")


def load_manifest(name: str) -> dict:
    path = EXPERIMENTS_DIR / f"{name}.json"
    if not path.is_file():
        available = sorted(p.stem for p in EXPERIMENTS_DIR.glob("*.json"))
        raise FileNotFoundError(
            f"no manifest {name!r} under lab/experiments/ (available: {available})")
    m = json.loads(path.read_text(encoding="utf-8"))
    validate_manifest(m)
    merged = dict(m)
    for key, default in OPTIONAL_DEFAULTS.items():
        merged.setdefault(key, default)
    return merged


def list_manifests() -> list[str]:
    return sorted(p.stem for p in EXPERIMENTS_DIR.glob("*.json"))
