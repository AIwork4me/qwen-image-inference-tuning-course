"""Pure, dependency-light contracts for M5.1 (CPU-testable, no GPU/OV/torch).

These functions are the single source of truth for admission classification,
artifact provenance, conditioning shape/trim rules, numerical gates, budgets,
worker status, and performance statistics. Tests exercise them on CPU-only CI.
"""
from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

IM_START = 151644
COND_DIM = 4096


def classify_admission(snapshot: dict) -> tuple[str, list[str]]:
    """Return (GO | GO_WITH_DEVIATIONS | BLOCKED_ENVIRONMENT, reasons)."""
    reasons = []
    gpu = snapshot.get("gpu", {})
    torch_info = snapshot.get("torch", {})
    if not torch_info.get("cuda_available"):
        return "BLOCKED_ENVIRONMENT", ["torch ROCm GPU is not available"]
    if torch_info.get("device_count") not in (1, None):
        return "BLOCKED_ENVIRONMENT", [f"expected 1 GPU, got {torch_info.get('device_count')}"]
    arch = gpu.get("gfx_arch") or gpu.get("rocminfo_gfx") or ""
    if not str(arch).startswith("gfx1100"):
        reasons.append(f"gfx arch {arch!r} is not the validated W7900 gfx1100 class")
    vram = gpu.get("vram_mib")
    if isinstance(vram, (int, float)) and not (46000 <= vram <= 50000):
        reasons.append(f"VRAM {vram} MiB is outside the 48 GB W7900 class")
    if reasons:
        return "GO_WITH_DEVIATIONS", reasons
    return "GO", []


def artifact_status(root: str | Path, required: list[str], manifest_name: str = "conversion_manifest.json") -> dict:
    """Check artifact presence + manifest shape. Missing/corrupt => BLOCKED."""
    root = Path(root)
    missing = [r for r in required if not (root / r).is_file()]
    manifest = root / manifest_name
    if missing:
        return {"status": "BLOCKED", "reason": "missing_artifacts", "missing": missing}
    if not manifest.is_file():
        return {"status": "BLOCKED", "reason": "missing_manifest", "path": str(manifest)}
    try:
        m = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        return {"status": "BLOCKED", "reason": "corrupt_manifest", "error": repr(e)}
    if not m.get("source_sha256") and not m.get("source_xml_sha256"):
        return {"status": "BLOCKED", "reason": "manifest_has_no_source_digest"}
    return {"status": "PASS", "manifest": m, "files": sorted(required)}


def trim_at_second_im_start(ids: list[int], im_start: int = IM_START) -> int:
    """Index of the second <|im_start|>; ComfyUI drops everything before it."""
    starts = [i for i, t in enumerate(ids) if t == im_start]
    return starts[1] if len(starts) > 1 else 0


def validate_cond_descriptor(shape, dtype: str, finite: bool) -> dict:
    ok = (len(shape) == 3 and shape[0] == 1 and shape[2] == COND_DIM
          and str(dtype).lower() in ("float32", "torch.float32") and finite)
    return {"ok": bool(ok), "shape": list(shape), "dtype": str(dtype),
            "reason": None if ok else "expected [1,seq,4096] float32 finite"}


def cosine(a, b) -> float:
    import numpy as np
    a = np.asarray(a, dtype=np.float64).ravel(); b = np.asarray(b, dtype=np.float64).ravel()
    return float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))


def relative_l2(a, b) -> float:
    import numpy as np
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-12))


def numerical_gate(metrics: dict, threshold: dict) -> dict:
    ok = (metrics.get("rel_l2", math.inf) <= threshold["rel_l2"]
          and metrics.get("cosine", -1) >= threshold["cosine"]
          and bool(metrics.get("finite", False)))
    return {"status": "PASS" if ok else "FAIL", "threshold": threshold, "metrics": metrics}


def budget_check(free_bytes: int, required_bytes: int, ram_free_bytes: int, ram_required_bytes: int) -> dict:
    reasons = []
    if free_bytes < required_bytes:
        reasons.append(f"disk short: {free_bytes} < {required_bytes}")
    if ram_free_bytes < ram_required_bytes:
        reasons.append(f"RAM short: {ram_free_bytes} < {ram_required_bytes}")
    return {"status": "PASS" if not reasons else "BLOCKED", "reasons": reasons}


def safe_cond_path(path: str | Path, allowed_root: str | Path) -> Path:
    p = Path(path).resolve(); root = Path(allowed_root).resolve()
    if root not in p.parents and p != root:
        raise ValueError(f"conditioning path escapes run root: {p}")
    if p.suffix != ".safetensors":
        raise ValueError("only .safetensors conditioning is accepted")
    return p


def performance_stats(samples: list[float]) -> dict:
    if not samples:
        return {"n": 0}
    s = sorted(float(x) for x in samples)
    def pct(q):
        if len(s) == 1: return s[0]
        idx = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
        return s[idx]
    return {"n": len(s), "min": s[0], "median": statistics.median(s),
            "p90": pct(0.90), "p95": pct(0.95), "max": s[-1]}


def validate_result_schema(record: dict) -> list[str]:
    required = ("experiment", "status", "timestamp", "source_head", "artifact_digest",
                "environment", "measurements", "evidence_paths")
    errors = [f"missing {k}" for k in required if k not in record]
    if record.get("status") not in ("PASS", "FAIL", "BLOCKED", "NOT_RUN", "PENDING"):
        errors.append("invalid status")
    if record.get("status") == "PASS" and not record.get("measurements"):
        errors.append("PASS requires measurements")
    return errors


def honest_status(requested: str, evidence: dict) -> str:
    """Never let a blocker be reported as PASS."""
    if evidence.get("blocked"):
        return "BLOCKED"
    if not evidence.get("measured"):
        return "NOT_RUN"
    if requested not in ("PASS", "FAIL", "PENDING"):
        return "NOT_RUN"
    return requested
