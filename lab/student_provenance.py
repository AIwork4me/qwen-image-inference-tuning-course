"""Student experiment provenance contract (student-independence closure).

Every physical measurement a course notebook touches carries an explicit,
machine-checkable ORIGIN. The categories are mutually exclusive:

    CURRENT_SESSION_PHYSICAL     this notebook session initiated and awaited
                                 the workload; bound by PROVENANCE.json
    SAME_STUDENT_PRIOR_PHYSICAL  an earlier session on THIS workspace,
                                 selected by exact run id, whose session
                                 record still exists
    SHARED_CAMPAIGN_PHYSICAL     a verified physical run produced by another
                                 notebook/campaign on this host, reused
                                 deliberately (real hardware evidence, not a
                                 new measurement by this student)
    COMMITTED_PHYSICAL_REFERENCE a physical result already tracked in Git;
                                 never presentable as newly measured here
    FROZEN_REFERENCE             official archived evidence under evidence/
    SIMULATED                    analytical estimate / scheduler simulation
    NOT_RUN                      no record exists yet
    BLOCKED                      admission or verification refused

Hard rules enforced here (fail closed):
  * run ids resolve strictly under ``runs/`` (no traversal, no escapes);
  * a Git-tracked run can never classify as CURRENT_SESSION_PHYSICAL;
  * an unbound, untracked, non-campaign run cannot be promoted to physical
    evidence at all (ProvenanceError);
  * CURRENT_SESSION_PHYSICAL requires a PROVENANCE.json binding written by
    the session AFTER the physical execution completed successfully;
  * performance claims are validated by validate_result_record().
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import subprocess
from datetime import datetime
from pathlib import Path

from . import lab_env

REPO = lab_env.REPO

PROVENANCE_CATEGORIES = (
    "CURRENT_SESSION_PHYSICAL",
    "SAME_STUDENT_PRIOR_PHYSICAL",
    "SHARED_CAMPAIGN_PHYSICAL",
    "COMMITTED_PHYSICAL_REFERENCE",
    "FROZEN_REFERENCE",
    "SIMULATED",
    "NOT_RUN",
    "BLOCKED",
)

EXECUTION_STATUSES = ("MEASURED", "REUSED", "SIMULATED", "NOT_RUN", "BLOCKED", "FAIL")

RESULT_CLASSIFICATIONS = (
    "FORMAL_PROTOCOL_MEASURED",
    "BOUNDED_PHYSICAL_COMPARISON",
    "SMOKE_PIPELINE_CHECK",
    "REFERENCE_ONLY",
    "SIMULATION_ONLY",
)

# verify_artifact() BLOCKED reasons split into FAIL (the artifact exists but
# is invalid) vs BLOCKED (required resource is unavailable).
ARTIFACT_FAIL_REASONS = frozenset({
    "corrupt_manifest", "manifest_has_no_source_digest", "unsafe_path",
    "size_mismatch", "unreadable_file", "digest_mismatch", "identity_mismatch",
})

_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_BINDING_NAME = "PROVENANCE.json"


class ProvenanceError(RuntimeError):
    """A provenance rule was violated; the claim must fail closed."""


# ---------------------------------------------------------------------------
# session identity
# ---------------------------------------------------------------------------

def new_session_id() -> str:
    """Unique per-kernel-session id, generated BEFORE any GPU work."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"stu-{stamp}-{secrets.token_hex(3)}"


def source_identity(repo: Path | None = None) -> dict:
    """Git identity of the source the session is executing from."""
    repo = repo or REPO
    head, dirty = None, None
    try:
        r = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                           capture_output=True, text=True, timeout=15)
        if r.returncode == 0:
            head = r.stdout.strip()
        s = subprocess.run(["git", "-C", str(repo), "status", "--porcelain"],
                           capture_output=True, text=True, timeout=15)
        if s.returncode == 0:
            dirty = bool(s.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        pass
    return {"head": head, "dirty": dirty}


def is_git_tracked(path: Path, repo: Path | None = None) -> bool:
    repo = repo or REPO
    try:
        rel = Path(path).resolve().relative_to(repo.resolve())
    except ValueError:
        return False
    r = subprocess.run(["git", "-C", str(repo), "ls-files", "--", str(rel)],
                       capture_output=True, text=True, timeout=15)
    return r.returncode == 0 and bool(r.stdout.strip())


# ---------------------------------------------------------------------------
# run-id safety + bindings
# ---------------------------------------------------------------------------

def run_dir(run_id: str, repo: Path | None = None) -> Path:
    if not isinstance(run_id, str) or not _RUN_ID_RE.match(run_id):
        raise ProvenanceError(
            f"invalid run id {run_id!r}: run ids are single path components "
            "under runs/ (letters, digits, '.', '_', '-')")
    d = (repo or REPO) / "runs" / run_id
    real = d.resolve()
    if (repo or REPO).resolve() not in real.parents:
        raise ProvenanceError(f"run id {run_id!r} escapes the repository runs/ root")
    return d


def binding_path(run_id: str, repo: Path | None = None) -> Path:
    return run_dir(run_id, repo) / _BINDING_NAME


def write_binding(run_id: str, session_id: str, *, kind: str,
                  repo: Path | None = None, extra: dict | None = None) -> dict:
    """Bind a completed physical run to the session that executed it.

    Written only AFTER the CLI finished successfully, so the binding itself
    is the execution-to-output evidence (hard requirement 4).
    """
    d = run_dir(run_id, repo)
    if not d.is_dir():
        raise ProvenanceError(f"cannot bind: run directory missing: {d}")
    rec = {
        "contract": "student-provenance/v1",
        "session_id": session_id,
        "run_origin": "CURRENT_SESSION_PHYSICAL",
        "kind": kind,
        "bound_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source": source_identity(repo),
    }
    if extra:
        rec.update(extra)
    p = d / _BINDING_NAME
    if p.exists():  # append-only: never silently rewrite a binding
        try:
            old = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            old = None
        if isinstance(old, dict) and old.get("session_id") != session_id:
            raise ProvenanceError(
                f"run {run_id} is already bound to session "
                f"{old.get('session_id')}; it cannot be re-bound here")
        if old == rec:
            return rec
    p.write_text(json.dumps(rec, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return rec


def read_binding(run_id: str, repo: Path | None = None) -> dict | None:
    p = binding_path(run_id, repo)
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else None
    except (OSError, ValueError):
        return None


# ---------------------------------------------------------------------------
# session-record index (what makes a prior session verifiable)
# ---------------------------------------------------------------------------

def _campaign_roots(repo: Path | None = None) -> list[Path]:
    repo = repo or REPO
    roots = []
    env = os.environ.get("QIR_NOTEBOOK_CAMPAIGN")
    if env:
        roots.append((repo / env).resolve())
    roots.append((repo / "runs" / "notebook-gpu-qualification").resolve())
    return roots


def sessions_roots(repo: Path | None = None) -> list[Path]:
    repo = repo or REPO
    out = [repo / "runs" / "notebook-sessions"]
    out += [r / "sessions" for r in _campaign_roots(repo)]
    return out


def session_record_exists(session_id: str, repo: Path | None = None) -> bool:
    """True iff a finished session record declares this session id."""
    for root in sessions_roots(repo):
        if not root.is_dir():
            continue
        for f in root.glob("*.json"):
            try:
                d = json.loads(f.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if isinstance(d, dict) and d.get("session_id") == session_id:
                return True
    return False


def _under_campaign(run_path: Path, repo: Path | None) -> bool:
    real = run_path.resolve()
    return any(root == real or root in real.parents for root in _campaign_roots(repo))


# ---------------------------------------------------------------------------
# classification
# ---------------------------------------------------------------------------

def classify_run(run_id: str, *, session_id: str | None = None,
                 repo: Path | None = None) -> dict:
    """Classify a physical run directory into one PROVENANCE_CATEGORY.

    Never guesses: anything that cannot be verified raises ProvenanceError
    unless it is plainly absent (NOT_RUN) or plainly tracked (committed).
    """
    d = run_dir(run_id, repo)
    if not d.is_dir():
        return {"category": "NOT_RUN", "run_id": run_id, "run_dir": str(d)}
    binding = read_binding(run_id, repo)
    tracked = is_git_tracked(d, repo)
    campaign = _under_campaign(d, repo)
    out = {"run_id": run_id, "run_dir": str(d), "binding": binding,
           "git_tracked": tracked, "in_campaign_root": campaign}
    if tracked:
        # Hard rule 3: a committed run is never current-session evidence,
        # even if this workspace happens to hold a binding for it.
        out["category"] = "COMMITTED_PHYSICAL_REFERENCE"
        if campaign:
            out["also_campaign_referenced"] = True
        return out
    if binding is not None:
        bound_session = binding.get("session_id")
        if session_id is not None and bound_session == session_id:
            out["category"] = "CURRENT_SESSION_PHYSICAL"
            return out
        if isinstance(bound_session, str) and session_record_exists(bound_session, repo):
            out["category"] = "SAME_STUDENT_PRIOR_PHYSICAL"
            out["bound_session_id"] = bound_session
            return out
        if campaign:
            out["category"] = "SHARED_CAMPAIGN_PHYSICAL"
            return out
        raise ProvenanceError(
            f"run {run_id} carries a binding to session {bound_session!r} but "
            "no session record for it exists on this workspace; it cannot be "
            "silently consumed. Re-run the experiment or select an explicitly "
            "verified reference run id.")
    if campaign:
        out["category"] = "SHARED_CAMPAIGN_PHYSICAL"
        return out
    raise ProvenanceError(
        f"run {run_id} has no PROVENANCE.json binding, is not tracked in Git, "
        "and is not part of a declared campaign; it cannot be promoted to "
        "physical evidence (fail closed). Re-run the experiment in this "
        "session or select a committed/campaign reference run id.")


def classify_path(rel: str, repo: Path | None = None) -> dict:
    """Classify a repository file (frozen/committed reference evidence)."""
    repo = repo or REPO
    p = (repo / rel).resolve()
    if not str(p).startswith(str(repo.resolve())):
        raise ProvenanceError(f"path escapes repository: {rel}")
    if p.parts[len(repo.resolve().parts):][:1] == ("evidence",):
        return {"category": "FROZEN_REFERENCE", "path": rel}
    if is_git_tracked(p, repo):
        return {"category": "COMMITTED_PHYSICAL_REFERENCE", "path": rel}
    return {"category": "UNVERIFIED", "path": rel}


def require_category(run_id: str, expect: str | tuple[str, ...], *,
                     session_id: str | None = None,
                     repo: Path | None = None) -> dict:
    """resolve + enforce: an explicit expectation that must hold exactly."""
    if isinstance(expect, str):
        expect = (expect,)
    for e in expect:
        if e not in PROVENANCE_CATEGORIES:
            raise ProvenanceError(f"unknown provenance category {e!r}")
    info = classify_run(run_id, session_id=session_id, repo=repo)
    if info["category"] not in expect:
        raise ProvenanceError(
            f"run {run_id} is {info['category']}, expected "
            f"{' or '.join(expect)}; refusing to proceed (fail closed)")
    return info


# ---------------------------------------------------------------------------
# labels (what the student sees)
# ---------------------------------------------------------------------------

_STAMP_TEXT = {
    "CURRENT_SESSION_PHYSICAL":
        "measured on THIS GPU in THIS session (physical)",
    "SAME_STUDENT_PRIOR_PHYSICAL":
        "YOUR earlier physical run on this workspace (prior session, explicit run id)",
    "SHARED_CAMPAIGN_PHYSICAL":
        "shared physical campaign run from another session (real hardware evidence, not a new measurement by you)",
    "COMMITTED_PHYSICAL_REFERENCE":
        "committed physical reference tracked in Git (instructor campaign; NOT measured by you)",
    "FROZEN_REFERENCE":
        "frozen reference evidence from the committed campaign",
    "SIMULATED": "analytical estimate or simulation - NOT a measurement",
    "NOT_RUN": "no physical record exists (experiment not run)",
    "BLOCKED": "blocked by admission or verification (no measurement)",
}


def stamp_text(category: str) -> str:
    if category not in _STAMP_TEXT:
        raise ProvenanceError(f"unknown provenance category {category!r}")
    return _STAMP_TEXT[category]


def label(run_id: str, category: str) -> str:
    """One-line provenance label pinned next to every consumed number."""
    return f"{category} | run_id={run_id} | {stamp_text(category)}"


# ---------------------------------------------------------------------------
# P5: machine-checkable student result records
# ---------------------------------------------------------------------------

def _sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True,
                                     ensure_ascii=False).encode()).hexdigest()[:16]


def workload_identity(pipeline_ab: dict) -> dict:
    """Workload identity of an A/C pipeline record (matched-arms check)."""
    prompts = pipeline_ab.get("prompts") or []
    return {
        "n_requests": len(prompts),
        "seed": prompts[0].get("seed") if prompts else None,
        "prompt_digest": _sha([[p.get("id"), p.get("text")] for p in prompts]),
        "definitions": pipeline_ab.get("definitions", {}),
    }


_FORMAL_MIN_MEASURED = 9
_registry: dict[str, dict] = {}


def register_run_metadata(run_id: str, meta: dict) -> None:
    """Duplicate run ids with conflicting metadata are rejected."""
    prev = _registry.get(run_id)
    if prev is not None and prev != meta:
        raise ProvenanceError(
            f"run {run_id} registered twice with conflicting metadata: "
            f"{prev!r} vs {meta!r}")
    _registry[run_id] = dict(meta)


def validate_result_record(rec: dict, *, session_id: str | None = None,
                           repo: Path | None = None) -> list[str]:
    """Validate a student result record; returns a list of violations."""
    errs: list[str] = []
    if not isinstance(rec, dict):
        return ["result record is not a dict"]
    cat = rec.get("provenance")
    if cat not in PROVENANCE_CATEGORIES:
        errs.append(f"invalid provenance category {cat!r}")
        return errs
    status = rec.get("execution_status")
    if status not in EXECUTION_STATUSES:
        errs.append(f"invalid execution status {status!r}")
    cls = rec.get("classification")
    if cls not in RESULT_CLASSIFICATIONS:
        errs.append(f"invalid classification {cls!r}")
    if not rec.get("session_id"):
        errs.append("missing session_id")
    if not rec.get("experiment_id"):
        errs.append("missing experiment_id")
    for k in ("source_head", "gpu", "workload", "measurements", "memory"):
        if not rec.get(k):
            errs.append(f"missing {k}")

    run_ids = rec.get("run_ids") or []
    if cat == "CURRENT_SESSION_PHYSICAL":
        if status != "MEASURED":
            errs.append("CURRENT_SESSION_PHYSICAL requires execution_status=MEASURED")
        if not run_ids:
            errs.append("CURRENT_SESSION_PHYSICAL requires run_ids")
        if not rec.get("source_head"):
            errs.append("a new formal claim requires the source Git head")
        for rid in run_ids:
            try:
                info = classify_run(rid, session_id=rec.get("session_id") or session_id,
                                    repo=repo)
                if info["category"] != "CURRENT_SESSION_PHYSICAL":
                    errs.append(f"run {rid} classifies as {info['category']}, "
                                "not CURRENT_SESSION_PHYSICAL")
            except ProvenanceError as exc:
                errs.append(f"run {rid}: {exc}")
    else:
        if status == "MEASURED":
            errs.append(f"{cat} is not a new measurement; execution_status "
                        "must be REUSED/SIMULATED/NOT_RUN/BLOCKED, never MEASURED")
        if cat in ("COMMITTED_PHYSICAL_REFERENCE", "FROZEN_REFERENCE",
                   "SHARED_CAMPAIGN_PHYSICAL", "SAME_STUDENT_PRIOR_PHYSICAL"):
            if not run_ids:
                errs.append(f"{cat} requires the explicit reference run_ids")

    if cls == "FORMAL_PROTOCOL_MEASURED":
        pop = rec.get("populations") or {}
        n = pop.get("measured_count") or 0
        if n < _FORMAL_MIN_MEASURED:
            errs.append(f"FORMAL classification needs >= {_FORMAL_MIN_MEASURED} "
                        f"measured samples (got {n}); use BOUNDED_PHYSICAL_COMPARISON")
    if (rec.get("speedup") is not None and cls not in
            ("FORMAL_PROTOCOL_MEASURED", "BOUNDED_PHYSICAL_COMPARISON",
             "SMOKE_PIPELINE_CHECK")):
        errs.append(f"{cls} results must not carry a speedup number")

    thr = rec.get("throughput")
    if thr is not None:
        for k in ("numerator", "denominator", "unit"):
            if k not in thr:
                errs.append(f"throughput missing {k}")

    ac = rec.get("ac_pair")
    if ac is not None:
        for side in ("a", "c"):
            if not ac.get(f"{side}_run_id"):
                errs.append(f"ac_pair missing {side}_run_id")
        if ac.get("a_workload") and ac.get("c_workload") \
                and ac["a_workload"] != ac["c_workload"]:
            errs.append("A/C workload mismatch: arms must share prompts, seeds "
                        "and request count")

    num = rec.get("numerical_status")
    human = rec.get("human_quality_status")
    if num == "FAIL" and human == "PASS":
        errs.append("numerical FAIL cannot be overridden by a human PASS")
    if human not in (None, "PASS", "FAIL", "PENDING_HUMAN_REVIEW", "NOT_ASSESSED"):
        errs.append(f"invalid human_quality_status {human!r}")
    if rec.get("experiment_id") == "M6" and rec.get("result_status") == "PASS" \
            and human == "PENDING_HUMAN_REVIEW":
        errs.append("M6 BEST result cannot be released PASS while human quality "
                    "review is PENDING_HUMAN_REVIEW")
    return errs


def validate_or_raise(rec: dict, **kw) -> dict:
    errs = validate_result_record(rec, **kw)
    if errs:
        raise ProvenanceError("invalid student result record: " + "; ".join(errs))
    return rec
