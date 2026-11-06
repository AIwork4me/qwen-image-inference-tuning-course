"""Source-identity capture for physical experiment runs (publication-closure P2).

Why this exists: the original P4 campaign (runs/p4-fairab-225326) executed
with a dirty worktree (one untracked + one modified module), and no surviving
artifact pinned the exact runtime bytes — Git HEAD alone could not identify
the executed source. This module makes every future run record carry a
verifiable source manifest:

  capture(modules, snapshot_dir=None) ->
    {
      "recorded_at_utc": ...,
      "git": {"head", "branch", "dirty", "dirty_files"},
      "tracked_diff_sha256": sha256 of `git diff` ("" digest when clean),
      "modules": [ {"path", "sha256", "bytes", "status"} ... ]   # sorted
      "provenance_class": ...
    }

Statuses: COMMITTED_CLEAN | TRACKED_MODIFIED | UNTRACKED | MISSING.
provenance_class:
  EXACT_SOURCE_VERIFIED             - clean worktree, every module committed
                                      (bytes recoverable from git by digest)
  EXACT_SOURCE_VERIFIED_WITH_SNAPSHOT - dirty/untracked modules had their
                                      exact bytes copied into snapshot_dir
  SOURCE_DIRTY_DIGESTS_ONLY         - dirty/untracked without a snapshot
                                      (digests recorded, bytes not preserved)

Rules honored: never reads os.environ (no secret capture); deterministic
ordering (sorted paths, stable dict key order); snapshots are capped
(<= 512 KiB per file) so run records never embed huge files.
"""
from __future__ import annotations

import hashlib
import subprocess
import time
from pathlib import Path

SNAPSHOT_MAX_BYTES = 512 * 1024


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args],
                          capture_output=True, text=True).stdout


def _digest_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _digest_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def capture(modules: list[str], repo: Path, snapshot_dir: Path | None = None) -> dict:
    """Record the source identity of `modules` (repo-relative paths) under `repo`."""
    head = _git(repo, "rev-parse", "HEAD").strip()
    branch = _git(repo, "rev-parse", "--abbrev-ref", "HEAD").strip()
    status = [l for l in _git(repo, "status", "--porcelain").splitlines() if l.strip()]
    tracked = set(_git(repo, "ls-files").splitlines())
    dirty = bool(status)
    diff = subprocess.run(["git", "-C", str(repo), "diff"], capture_output=True).stdout

    rows: list[dict] = []
    all_clean = not dirty
    any_snapshotted = False
    any_unpreserved_dirty = False
    for rel in sorted(modules):
        p = repo / rel
        if not p.is_file():
            rows.append({"path": rel, "sha256": None, "bytes": None, "status": "MISSING"})
            any_unpreserved_dirty = True  # a missing module is never "exact"
            continue
        d = _digest_file(p)
        if rel in tracked:
            blob = _git(repo, "rev-parse", f"HEAD:{rel}").strip()
            committed_digest = None
            if blob and not blob.startswith("HEAD:"):
                out = subprocess.run(["git", "-C", str(repo), "cat-file", "blob", blob],
                                     capture_output=True).stdout
                committed_digest = _digest_bytes(out)
            status_this = ("COMMITTED_CLEAN" if committed_digest == d
                           else "TRACKED_MODIFIED")
        else:
            status_this = "UNTRACKED"
        row = {"path": rel, "sha256": d, "bytes": p.stat().st_size, "status": status_this}
        if status_this == "COMMITTED_CLEAN":
            pass
        elif snapshot_dir is not None and p.stat().st_size <= SNAPSHOT_MAX_BYTES:
            dest = snapshot_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(p.read_bytes())
            row["snapshot"] = str(dest.relative_to(snapshot_dir))
            any_snapshotted = True
        else:
            any_unpreserved_dirty = True
        if status_this != "COMMITTED_CLEAN":
            all_clean = False
        rows.append(row)

    if all_clean and not any_unpreserved_dirty:
        cls = "EXACT_SOURCE_VERIFIED"
    elif any_snapshotted and not any_unpreserved_dirty:
        cls = "EXACT_SOURCE_VERIFIED_WITH_SNAPSHOT"
    else:
        cls = "SOURCE_DIRTY_DIGESTS_ONLY"

    return {
        "recorded_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git": {"head": head, "branch": branch, "dirty": dirty, "dirty_files": sorted(status)},
        "tracked_diff_sha256": _digest_bytes(diff),
        "modules": rows,
        "provenance_class": cls,
    }
