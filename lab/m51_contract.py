"""Pure, dependency-light contracts for M5.1 (CPU-testable, no GPU/OV/torch).

These functions are the single source of truth for admission classification,
artifact provenance (incl. real byte-level digest verification), conditioning
shape/trim rules, numerical gates, budgets, worker status, release-gate
propagation, conservative RAM admission, empty-negative tokenizer policy, and
performance statistics. Tests exercise them on CPU-only CI.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
from pathlib import Path

IM_START = 151644
COND_DIM = 4096

# ---------------------------------------------------------------------------
# Release gates (P0-A): execution success is never scientific success.
# ---------------------------------------------------------------------------

GATE_ORDER = [
    "environment", "artifact_integrity", "encoder_runtime", "token_contract",
    "conditioning_alignment", "e2e_gpu", "performance", "human_quality",
]
GATE_STATES = ("PASS", "FAIL", "BLOCKED", "NOT_RUN", "PENDING")

# The empty-string negative token of the fallback tokenizer policy (the
# literal word "empty"). The pinned ComfyUI production tokenizer does NOT
# insert it; see empty_negative_explain().
EMPTY_FALLBACK_TOKEN = 220


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


def release_status(gates: dict) -> str:
    """Derive the honest release status from gate states.

    FAIL beats BLOCKED beats PENDING; human PENDING can only be cleared by an
    explicit human verdict (settle_human_quality), never by numeric PASSes.
    """
    unknown = [k for k, v in gates.items() if v not in GATE_STATES]
    if unknown:
        raise ValueError(f"invalid gate states: {unknown}")
    vals = [gates.get(k, "NOT_RUN") for k in GATE_ORDER]
    if "FAIL" in vals:
        return "RELEASE_FAIL"
    if "BLOCKED" in vals:
        return "BLOCKED"
    if gates.get("human_quality") == "PENDING":
        return "PENDING_HUMAN_REVIEW"
    if "PENDING" in vals:
        return "PENDING_GATES"
    if "NOT_RUN" in vals:
        return "INCOMPLETE_SCOPE"
    if all(v == "PASS" for v in vals):
        return "RELEASE_PASS"
    return "INCOMPLETE_SCOPE"


def settle_human_quality(gates: dict, verdict: dict | None) -> dict:
    """Apply a human image-quality verdict. Only an explicit record with a
    reviewer and a PASS/FAIL verdict may change the gate; nothing numeric."""
    if not isinstance(verdict, dict) or not verdict.get("reviewer"):
        gates["human_quality"] = "PENDING"
        return gates
    v = str(verdict.get("verdict", "")).upper()
    if v in ("PASS", "FAIL"):
        gates["human_quality"] = v
    else:
        gates["human_quality"] = "PENDING"
    return gates


def empty_negative_explain(ids_production: list[int], ids_fallback: list[int]) -> dict:
    """Explain the empty-negative contract between the ComfyUI production
    tokenizer (prevent_empty_text=False, the pinned default in
    comfy/text_encoders/qwen_image21.py) and the diagnostic fallback policy
    (prevent_empty_text=True, which inserts the literal token 220).
    """
    if ids_production == ids_fallback:
        return {"policy": "identical", "inserted_fallback_token": None,
                "production_ids": ids_production, "fallback_ids": ids_fallback}
    extra = [t for t in ids_fallback if t == EMPTY_FALLBACK_TOKEN]
    idx = None
    for i, t in enumerate(ids_fallback):
        if t == EMPTY_FALLBACK_TOKEN and (i >= len(ids_production) or ids_production[i] != t):
            idx = i
            break
    return {"policy": "fallback_inserts_empty_token",
            "inserted_fallback_token": EMPTY_FALLBACK_TOKEN if extra else None,
            "inserted_at_index": idx,
            "production_ids": ids_production, "fallback_ids": ids_fallback,
            "production_is_canonical": EMPTY_FALLBACK_TOKEN not in ids_production or
            ids_production.count(EMPTY_FALLBACK_TOKEN) == ids_fallback.count(EMPTY_FALLBACK_TOKEN)}


# ---------------------------------------------------------------------------
# Artifact integrity (P0-B): verify bytes, not manifest claims.
# ---------------------------------------------------------------------------

def streaming_sha256(path: str | Path, chunk_bytes: int = 1 << 23) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk_bytes), b""):
            h.update(block)
    return h.hexdigest()


_DIGEST_KEYS = ("sha256", "output_sha256", "output_xml_sha256", "output_bin_sha256")


def _manifest_digest_for(manifest: dict, filename: str) -> tuple[str | None, str]:
    """Return (digest, key) the manifest pins for this artifact file.

    Generic keys (output_bin_sha256 / output_xml_sha256) only apply to the
    matching file type; stem-named keys (output_<stem>_sha256) apply to their
    own file. A manifest pinning only the bin digest leaves the xml unpinned
    (reported as a warning, not silently trusted).
    """
    stem = filename.rsplit(".", 1)[0]
    cands = []
    for key in (f"output_{stem}_sha256", f"{stem}_sha256"):
        v = manifest.get(key)
        if isinstance(v, str) and len(v) == 64:
            cands.append((v, key))
    if not cands:
        generic = "output_xml_sha256" if filename.endswith(".xml") else \
                  ("output_bin_sha256" if filename.endswith(".bin") else None)
        if generic:
            v = manifest.get(generic)
            if isinstance(v, str) and len(v) == 64:
                cands.append((v, generic))
    return cands[0] if cands else (None, "")


def _cache_load(root: Path) -> dict:
    p = root / ".m51_verification_cache.json"
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _cache_save(root: Path, cache: dict) -> None:
    p = root / ".m51_verification_cache.json"
    try:
        p.write_text(json.dumps(cache, indent=1), encoding="utf-8")
    except OSError:
        pass


def verify_artifact(root: str | Path, required: list[str],
                    manifest_name: str = "conversion_manifest.json",
                    expected_identity: dict | None = None,
                    use_cache: bool = False) -> dict:
    """Byte-level artifact verification against the conversion manifest.

    - streaming SHA256 of every required file (never loads the model in RAM)
    - manifest must pin the SOURCE digest (chain anchor to the git-pinned TE
      checkpoint) and a digest for each binary weights file
    - size checks catch truncation even before hashing
    - path safety: resolved artifact files must live under the resolved root
      (directory symlinks into the root are allowed; file symlink escapes are
      rejected)
    - identity: optional expected manifest fields (storage/runtime_precision/
      dtype/mode) must match, or the wrong artifact is BLOCKED
    - optional persistent cache (use_cache=True) keyed by
      (realpath, ino, size, mtime_ns, ctime_ns) remembers digests verified
      against a pin. NOTE: filesystems with coarse timestamp granularity can
      keep a stale entry alive across a same-size same-tick tamper, so the
      default is NO persistent cache: admission re-verifies bytes each run
      (~20 s for a 26 GB bin at NVMe speeds), and the cache exists only for
      callers that accept the documented tradeoff (e.g. provisioning loops).

    Statuses: PASS | PASS_WITH_WARNINGS | BLOCKED.
    Runtime policy: call this at admission/provisioning time, NOT per
    inference request.
    """
    root_p = Path(root)
    if not root_p.is_dir():
        return {"status": "BLOCKED", "reason": "missing_artifact_root", "root": str(root_p)}
    root_real = root_p.resolve()
    missing = [r for r in required if not (root_p / r).is_file()]
    if missing:
        return {"status": "BLOCKED", "reason": "missing_artifacts", "missing": missing}
    manifest = root_p / manifest_name
    if not manifest.is_file():
        return {"status": "BLOCKED", "reason": "missing_manifest", "path": str(manifest)}
    try:
        m = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        return {"status": "BLOCKED", "reason": "corrupt_manifest", "error": repr(e)}
    if not (m.get("source_sha256") or m.get("source_xml_sha256")):
        return {"status": "BLOCKED", "reason": "manifest_has_no_source_digest"}

    warnings: list[str] = []
    cache = _cache_load(root_p) if use_cache else {}
    files = {}
    for name in sorted(required):
        p = root_p / name
        real = p.resolve()
        if root_real != real and root_real not in real.parents:
            return {"status": "BLOCKED", "reason": "unsafe_path", "file": name,
                    "resolved": str(real), "root": str(root_real)}
        st = p.stat()
        pinned, pin_key = _manifest_digest_for(m, name)
        stem = name.rsplit(".", 1)[0]
        size_key = None
        for key in (f"output_{stem}_bytes", f"{stem}_bytes"):
            if isinstance(m.get(key), int):
                size_key = key
                break
        if size_key is None and name.endswith(".bin"):
            for key in ("output_bin_bytes", "bin_bytes"):
                if isinstance(m.get(key), int):
                    size_key = key
                    break
        if size_key and st.st_size != m[size_key]:
            return {"status": "BLOCKED", "reason": "size_mismatch", "file": name,
                    "expected": m[size_key], "actual": st.st_size}
        ck = str(real)
        entry = cache.get(ck) if use_cache else None
        if (use_cache and pinned and entry and entry.get("verified")
                and entry.get("size") == st.st_size
                and entry.get("mtime_ns") == st.st_mtime_ns
                and entry.get("ctime_ns") == st.st_ctime_ns
                and entry.get("ino") == st.st_ino
                and entry.get("sha256") == pinned):
            files[name] = {"sha256": pinned, "cached": True, "pinned_via": pin_key}
            continue
        try:
            digest = streaming_sha256(real)
        except OSError as e:
            return {"status": "BLOCKED", "reason": "unreadable_file", "file": name, "error": repr(e)}
        if pinned:
            if digest != pinned:
                return {"status": "BLOCKED", "reason": "digest_mismatch", "file": name,
                        "pinned": pinned, "actual": digest, "pinned_via": pin_key}
            if use_cache:
                cache[ck] = {"verified": True, "sha256": digest, "size": st.st_size,
                             "mtime_ns": st.st_mtime_ns, "ctime_ns": st.st_ctime_ns,
                             "ino": st.st_ino}
            files[name] = {"sha256": digest, "pinned_via": pin_key}
        else:
            warnings.append(f"{name}: manifest pins no digest (computed {digest}); "
                            f"re-provision to pin it")
            files[name] = {"sha256": digest, "unpinned": True}
    if use_cache and cache:
        _cache_save(root_p, cache)

    if expected_identity:
        mism = {k: {"expected": v, "manifest": m.get(k)} for k, v in expected_identity.items()
                if str(m.get(k)).lower() != str(v).lower()}
        if mism:
            return {"status": "BLOCKED", "reason": "identity_mismatch", "mismatch": mism,
                    "manifest": m, "files": files}
    status = "PASS_WITH_WARNINGS" if warnings else "PASS"
    return {"status": status, "warnings": warnings, "manifest": m, "files": files}


# ---------------------------------------------------------------------------
# Conservative RAM admission (P1-A).
# ---------------------------------------------------------------------------

def read_cgroup_mem(path: str = "/sys/fs/cgroup") -> dict:
    """Read effective cgroup memory facts without assuming availability."""
    base = Path(path)
    out: dict = {}
    try:
        mx = (base / "memory.max").read_text().strip()
        out["limit_bytes"] = int(mx) if mx.isdigit() else None  # None == "max" (unlimited)
    except OSError:
        out["limit_bytes"] = None
    try:
        out["current_bytes"] = int((base / "memory.current").read_text().strip())
    except OSError:
        out["current_bytes"] = None
    anon = 0
    try:
        for line in (base / "memory.stat").read_text().splitlines():
            if line.startswith("anon "):
                anon = int(line.split()[1])
                break
    except OSError:
        pass
    out["anon_bytes"] = anon
    return out


def host_mem_available_bytes(meminfo: str = "/proc/meminfo") -> int | None:
    try:
        for line in Path(meminfo).read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


def conservative_ram_gate(cgroup: dict, host_available: int | None,
                          required_bytes: int, safety_bytes: int = 2 * 2**30,
                          concurrent_reserve_bytes: int = 0) -> dict:
    """Conservative RAM admission.

    Policy (explicit): available = min(host MemAvailable, cgroup_limit - cgroup_anon)
    minus safety headroom minus concurrent process reserve (e.g. a resident
    ComfyUI server). cgroup memory.current is NOT treated as available because
    it mostly counts reclaimable page cache; the non-reclaimable part (anon) is
    what constrains a new anonymous working set. Unlimited cgroup (limit None)
    falls back to host MemAvailable only.
    """
    cands = []
    if host_available is not None:
        cands.append(("host_mem_available", host_available))
    if cgroup.get("limit_bytes") is not None:
        headroom = cgroup["limit_bytes"] - (cgroup.get("anon_bytes") or 0)
        cands.append(("cgroup_limit_minus_anon", headroom))
    if not cands:
        return {"status": "BLOCKED", "reason": "no_memory_information",
                "available_bytes": 0, "required_bytes": required_bytes}
    basis, avail = min(cands, key=lambda kv: kv[1])
    usable = avail - safety_bytes - concurrent_reserve_bytes
    ok = usable >= required_bytes
    return {"status": "PASS" if ok else "BLOCKED",
            "reason": None if ok else "insufficient_ram_headroom",
            "basis": basis, "available_bytes": avail, "usable_bytes": usable,
            "required_bytes": required_bytes, "safety_bytes": safety_bytes,
            "concurrent_reserve_bytes": concurrent_reserve_bytes,
            "cgroup": {k: cgroup.get(k) for k in ("limit_bytes", "current_bytes", "anon_bytes")}}


def artifact_status(root: str | Path, required: list[str], manifest_name: str = "conversion_manifest.json") -> dict:
    """Check artifact presence + manifest shape. Missing/corrupt => BLOCKED.

    Byte-level verification is verify_artifact(); admission paths must use it.
    """
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


# ---------------------------------------------------------------------------
# Student-facing BF16 admission (M5.1 notebook Part A; student-independence
# closure P2). One call, one contract:
#   PASS                every identity/integrity/resource check satisfied
#   PASS_WITH_WARNINGS  artifact operational + verified to the documented
#                       extent, but >=1 historical integrity field unpinned
#                       (never silently rounded up to PASS)
#   FAIL                a mandatory check failed on an artifact that IS
#                       present (digest/size/identity/manifest corruption) —
#                       dependent inference MUST NOT proceed
#   BLOCKED             a required artifact or resource is unavailable —
#                       remediation command is returned
# A subprocess return code is never a scientific status; this contract is.
# ---------------------------------------------------------------------------

OV_BF16_REQUIRED_FILES = ("language_model_bf16.xml", "language_model_bf16.bin")
OV_BF16_IDENTITY = {"storage": "FP16", "runtime_precision": "bf16"}
OV_BF16_NEED_RAM_BYTES = 34 * 2**30  # same budget the H3_OV_BF16 spec pins

# verify_artifact() BLOCKED reasons reclassified as FAIL (invalid artifact
# present) rather than BLOCKED (artifact/resource unavailable).
_FAIL_REASONS = frozenset({
    "corrupt_manifest", "manifest_has_no_source_digest", "unsafe_path",
    "size_mismatch", "unreadable_file", "digest_mismatch", "identity_mismatch",
})


def _read_cpu_quota(path: str = "/sys/fs/cgroup/cpu.max") -> dict:
    try:
        parts = (Path(path).read_text().strip().split())
        if len(parts) == 2 and parts[1].isdigit():
            quota = int(parts[0]) / int(parts[1])
            return {"cpus": round(quota, 2), "raw": " ".join(parts)}
    except (OSError, ValueError, ZeroDivisionError):
        pass
    return {"cpus": None, "raw": None}


def admit_ov_bf16(artifact_root: str | Path | None = None, *,
                  repo: Path | None = None,
                  need_ram_bytes: int = OV_BF16_NEED_RAM_BYTES,
                  concurrent_reserve_bytes: int = 4 * 2**30,
                  runtime_check: bool = True) -> dict:
    """Admission contract for the M5.1 OpenVINO BF16 deployment.

    Reuses the single byte-level verifier (verify_artifact) — there is no
    second, weaker validator. Environment overrides come only from
    $M51_OV_ARTIFACTS (documented provisioning variable), never from
    notebook-editable metadata."""
    from . import lab_env
    repo = repo or lab_env.REPO
    resolved_from = "argument"
    if artifact_root is None:
        env = os.environ.get("M51_OV_ARTIFACTS")
        if env:
            artifact_root, resolved_from = env, "env:M51_OV_ARTIFACTS"
        else:
            artifact_root, resolved_from = repo / "runs" / "m51-ov-artifacts", "default"
    root = Path(artifact_root) / "bf16"
    checks: list[dict] = []

    def _remedy_missing() -> str:
        return ("stage verified OpenVINO artifacts: `bash scripts/m51/provision_ov.sh`, "
                "or point $M51_OV_ARTIFACTS at the shared provisioned root "
                "(e.g. /opt/m51-artifacts) and re-run this cell")

    verification = verify_artifact(root, list(OV_BF16_REQUIRED_FILES),
                                   expected_identity=dict(OV_BF16_IDENTITY))
    vstat = verification.get("status")
    if vstat in ("PASS", "PASS_WITH_WARNINGS"):
        checks.append({"name": "artifact_integrity", "status": vstat,
                       "detail": {"root": str(root),
                                  "warnings": verification.get("warnings", [])}})
    else:
        reason = verification.get("reason", "unknown")
        if reason in _FAIL_REASONS:
            checks.append({"name": "artifact_integrity", "status": "FAIL",
                           "detail": verification})
            return {"contract": "m51-ov-admission/v1", "status": "FAIL",
                    "artifact_root": str(root), "resolved_from": resolved_from,
                    "checks": checks, "verification": verification,
                    "identity_expected": dict(OV_BF16_IDENTITY),
                    "ram_gate": None, "cpu_quota": _read_cpu_quota(),
                    "remediation": ("artifact present but INVALID "
                                    f"({reason}): re-provision with "
                                    "`bash scripts/m51/provision_ov.sh`; do NOT "
                                    "continue dependent inference with it")}
        checks.append({"name": "artifact_availability", "status": "BLOCKED",
                       "detail": verification})
        return {"contract": "m51-ov-admission/v1", "status": "BLOCKED",
                "artifact_root": str(root), "resolved_from": resolved_from,
                "checks": checks, "verification": verification,
                "identity_expected": dict(OV_BF16_IDENTITY),
                "ram_gate": None, "cpu_quota": _read_cpu_quota(),
                "remediation": _remedy_missing()}

    warnings = list(verification.get("warnings", []))

    # OpenVINO runtime availability (the OV workers run in the repo venv)
    if runtime_check:
        ov_py = repo / ".venv-ov" / "bin" / "python"
        if not ov_py.is_file():
            return {"contract": "m51-ov-admission/v1", "status": "BLOCKED",
                    "artifact_root": str(root), "resolved_from": resolved_from,
                    "checks": checks + [{"name": "openvino_runtime",
                                         "status": "BLOCKED",
                                         "detail": {"missing": str(ov_py)}}],
                    "verification": verification,
                    "identity_expected": dict(OV_BF16_IDENTITY),
                    "ram_gate": None, "cpu_quota": _read_cpu_quota(),
                    "remediation": ("create the pinned OpenVINO venv: "
                                    "python3 -m venv .venv-ov && "
                                    ".venv-ov/bin/pip install openvino==2026.4.1 "
                                    "(see scripts/m51/provision_ov.sh)")}

    # Conservative RAM admission (same policy as the H3 runner; the pipeline
    # C arm additionally keeps a ComfyUI server resident -> reserve).
    ram = conservative_ram_gate(read_cgroup_mem(), host_mem_available_bytes(),
                                need_ram_bytes,
                                concurrent_reserve_bytes=concurrent_reserve_bytes)
    checks.append({"name": "ram_admission",
                   "status": ram.get("status", "BLOCKED"), "detail": ram})
    if ram.get("status") != "PASS":
        return {"contract": "m51-ov-admission/v1", "status": "BLOCKED",
                "artifact_root": str(root), "resolved_from": resolved_from,
                "checks": checks, "verification": verification,
                "identity_expected": dict(OV_BF16_IDENTITY),
                "ram_gate": ram, "cpu_quota": _read_cpu_quota(),
                "remediation": (f"insufficient RAM headroom for the BF16 CPU TE "
                                f"(needs {need_ram_bytes/2**30:.0f} GiB usable; "
                                f"basis {ram.get('basis')}, usable "
                                f"{ram.get('usable_bytes', 0)/2**30:.1f} GiB). "
                                "Free memory, lower concurrency, or use a larger "
                                "instance; do not lower the safety headroom")}

    quota = _read_cpu_quota()
    if quota.get("cpus") is not None and quota["cpus"] < 8:
        warnings.append(f"CPU quota {quota['cpus']} cores is below the 16-thread "
                        "reference; OV encoder numbers will differ from the campaign")

    status = "PASS_WITH_WARNINGS" if warnings else "PASS"
    return {"contract": "m51-ov-admission/v1", "status": status,
            "artifact_root": str(root), "resolved_from": resolved_from,
            "checks": checks, "verification": verification,
            "identity_expected": dict(OV_BF16_IDENTITY),
            "identity_manifest": {k: (verification.get("manifest") or {}).get(k)
                                  for k in OV_BF16_IDENTITY},
            "ram_gate": ram, "cpu_quota": quota,
            "warnings": warnings,
            "remediation": None}
