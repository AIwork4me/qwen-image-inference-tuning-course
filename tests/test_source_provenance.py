"""CPU-only tests for lab.source_provenance (publication-closure P2).

Each test builds a tiny throwaway git repo and verifies one provenance rule:
clean-worktree identity, dirty detection, tracked-modified digests, untracked
module snapshots, missing paths, deterministic manifest format, and the
no-secrets guarantee (the capture never reads the environment).
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from lab import source_provenance as SP  # noqa: E402


def _repo(tmp_path: Path) -> Path:
    r = tmp_path / "proj"
    r.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=r, check=True)
    subprocess.run(["git", "-C", str(r), "config", "user.email", "t@t"], check=True)
    subprocess.run(["git", "-C", str(r), "config", "user.name", "t"], check=True)
    (r / "lab").mkdir()
    (r / "lab" / "mod.py").write_text("X = 1\n")
    (r / "other.txt").write_text("hello\n")
    subprocess.run(["git", "-C", str(r), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(r), "commit", "-qm", "init"], check=True)
    return r


def test_clean_worktree_is_exact_source_verified(tmp_path):
    r = _repo(tmp_path)
    rec = SP.capture(["lab/mod.py"], r)
    assert rec["git"]["dirty"] is False
    assert rec["git"]["dirty_files"] == []
    assert rec["tracked_diff_sha256"] == hashlib.sha256(b"").hexdigest()
    assert [m["status"] for m in rec["modules"]] == ["COMMITTED_CLEAN"]
    assert rec["provenance_class"] == "EXACT_SOURCE_VERIFIED"


def test_dirty_tracked_file_detected_with_changed_digest(tmp_path):
    r = _repo(tmp_path)
    (r / "lab" / "mod.py").write_text("X = 2  # modified\n")
    rec = SP.capture(["lab/mod.py"], r)
    assert rec["git"]["dirty"] is True
    assert any("lab/mod.py" in l for l in rec["git"]["dirty_files"])
    assert rec["tracked_diff_sha256"] != hashlib.sha256(b"").hexdigest()
    m = rec["modules"][0]
    assert m["status"] == "TRACKED_MODIFIED"
    # recorded digest is the ACTUAL worktree digest, not the HEAD digest
    assert m["sha256"] == hashlib.sha256((r / "lab" / "mod.py").read_bytes()).hexdigest()
    assert rec["provenance_class"] != "EXACT_SOURCE_VERIFIED"


def test_untracked_module_snapshot_preserves_exact_bytes(tmp_path):
    r = _repo(tmp_path)
    (r / "lab" / "new_module.py").write_text("Y = 42\n")
    snap = tmp_path / "snap"
    rec = SP.capture(["lab/new_module.py", "lab/mod.py"], r, snapshot_dir=snap)
    byp = {m["path"]: m for m in rec["modules"]}
    assert byp["lab/new_module.py"]["status"] == "UNTRACKED"
    assert byp["lab/new_module.py"]["snapshot"] == "lab/new_module.py"
    saved = snap / "lab" / "new_module.py"
    assert saved.read_bytes() == (r / "lab" / "new_module.py").read_bytes()
    assert rec["provenance_class"] == "EXACT_SOURCE_VERIFIED_WITH_SNAPSHOT"


def test_untracked_without_snapshot_is_digests_only(tmp_path):
    r = _repo(tmp_path)
    (r / "lab" / "new_module.py").write_text("Y = 42\n")
    rec = SP.capture(["lab/new_module.py"], r)  # no snapshot_dir
    assert rec["provenance_class"] == "SOURCE_DIRTY_DIGESTS_ONLY"
    assert rec["modules"][0]["sha256"] == hashlib.sha256(b"Y = 42\n").hexdigest()


def test_missing_source_path_is_reported_not_invented(tmp_path):
    r = _repo(tmp_path)
    rec = SP.capture(["lab/ghost.py"], r)
    assert rec["modules"][0] == {"path": "lab/ghost.py", "sha256": None,
                                 "bytes": None, "status": "MISSING"}
    assert rec["provenance_class"] == "SOURCE_DIRTY_DIGESTS_ONLY"


def test_manifest_format_is_deterministic(tmp_path):
    r = _repo(tmp_path)
    (r / "lab" / "lab2.py").write_text("Z = 3\n")
    a = SP.capture(["lab/lab2.py", "lab/mod.py"], r, snapshot_dir=tmp_path / "s1")
    b = SP.capture(["lab/lab2.py", "lab/mod.py"], r, snapshot_dir=tmp_path / "s2")
    a2 = dict(a); b2 = dict(b)
    a2.pop("recorded_at_utc"); b2.pop("recorded_at_utc")
    assert json.dumps(a2, sort_keys=False) == json.dumps(b2, sort_keys=False)
    # sorted module order regardless of input order
    assert [m["path"] for m in a["modules"]] == sorted(m["path"] for m in a["modules"])


def test_no_environment_or_secrets_captured(tmp_path):
    r = _repo(tmp_path)
    os.environ["P6_FAKE_SECRET_TOKEN"] = "hunter2-supersecret"
    try:
        rec = SP.capture(["lab/mod.py"], r)
        blob = json.dumps(rec)
        assert "hunter2-supersecret" not in blob
        assert "P6_FAKE_SECRET_TOKEN" not in blob
        assert set(rec.keys()) == {"recorded_at_utc", "git", "tracked_diff_sha256",
                                   "modules", "provenance_class"}
    finally:
        del os.environ["P6_FAKE_SECRET_TOKEN"]


def test_experiment_module_list_resolves_in_real_repo():
    # the runner's EXPERIMENT_MODULES must all exist in this checkout
    from lab import m51_pipeline_fair_ab as FA
    for rel in FA.EXPERIMENT_MODULES:
        assert (REPO / rel).is_file(), rel
    # capture on the real repo must classify every module (statuses depend on
    # the developer's worktree; the contract is that none is MISSING)
    rec = SP.capture(FA.EXPERIMENT_MODULES, REPO)
    assert all(m["status"] != "MISSING" for m in rec["modules"])
    assert rec["provenance_class"] in ("EXACT_SOURCE_VERIFIED",
                                       "EXACT_SOURCE_VERIFIED_WITH_SNAPSHOT",
                                       "SOURCE_DIRTY_DIGESTS_ONLY")
