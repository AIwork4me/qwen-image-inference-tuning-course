"""CPU-only regression tests for the cloud staging planner
(scripts/cloud/plan_stage.py) — the decision logic behind start.sh.

Covers the release-blocker class: incremental disk budgeting (present files
must not count as new downloads; --smoke-only must never be disk-blocked),
safe ComfyUI staging (unknown trees are quarantined, never deleted), staged
integrity verification, and the rerun/restart paths.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "cloud"))

import plan_stage  # noqa: E402

GiB = 2**30


def _mkweights(root: Path, files: dict[str, int]):
    """Create a fake weights tree: rel path -> file size in bytes.

    Uses SPARSE files (truncate): the planner only inspects existence and
    st_size, and materializing 25 real GB per test would be absurd.
    """
    for rel, size in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "wb") as f:
            f.truncate(size)


def _pinned_sizes(rels) -> dict[str, int]:
    from lab import weights

    return {rel: weights.WEIGHTS[rel][0] for rel in rels}


STAGE1 = plan_stage.STAGE1
FULL = plan_stage.ALL
STAGE1_BYTES = sum(_pinned_sizes(STAGE1).values())
ALL_BYTES = sum(_pinned_sizes(FULL).values())


# ---------------------------------------------------------------- budgets
def test_first_install_budget_full_stage(tmp_path):
    """First install: nothing present -> the whole stage-1 set is needed."""
    plan = plan_stage.plan_model_staging(
        weights_root=tmp_path, free_bytes=STAGE1_BYTES + 4 * GiB)
    assert plan["verdict"] == "OK"
    assert sorted(plan["downloads"]) == sorted(STAGE1)
    assert plan["need_gib"] == pytest.approx(STAGE1_BYTES / GiB, abs=0.1)


def test_first_install_low_disk_blocks(tmp_path):
    """Low disk on a first install is BLOCKED with an actionable message."""
    plan = plan_stage.plan_model_staging(
        weights_root=tmp_path, free_bytes=STAGE1_BYTES // 2)
    assert plan["verdict"] == "BLOCKED"
    assert "free space" in plan["message"].lower()


def test_rerun_all_present_needs_nothing(tmp_path):
    """Re-run with the staged set complete: budget requirement is zero —
    already-verified files must never count as new downloads."""
    _mkweights(tmp_path, _pinned_sizes(STAGE1))
    plan = plan_stage.plan_model_staging(
        weights_root=tmp_path, free_bytes=1 * GiB)  # almost-full disk
    assert plan["verdict"] == "OK"
    assert plan["downloads"] == []
    assert plan["need_gib"] == 0.0
    assert sorted(plan["present"]) == sorted(STAGE1)


def test_partial_download_counts_only_missing(tmp_path):
    """Partial download: only the missing file counts toward the budget."""
    sizes = _pinned_sizes(STAGE1)
    present = [r for r in STAGE1 if "vae" in r]
    _mkweights(tmp_path, {r: sizes[r] for r in present})
    missing = [r for r in STAGE1 if r not in present]
    plan = plan_stage.plan_model_staging(
        weights_root=tmp_path,
        free_bytes=sum(sizes[r] for r in missing) + 3 * GiB)
    assert plan["verdict"] == "OK"
    assert plan["downloads"] == missing
    assert plan["need_gib"] == pytest.approx(
        sum(sizes[r] for r in missing) / GiB, abs=0.05)


def test_all_models_is_incremental_over_stage1(tmp_path):
    """--all-models with stage-1 already present: needs only the remaining
    variants, not the whole 61 GB."""
    sizes = _pinned_sizes(STAGE1)
    _mkweights(tmp_path, sizes)
    rest = [r for r in FULL if r not in STAGE1]
    rest_bytes = sum(_pinned_sizes(rest).values())
    plan = plan_stage.plan_model_staging(
        weights_root=tmp_path, all_models=True,
        free_bytes=rest_bytes + 4 * GiB)
    assert plan["verdict"] == "OK"
    assert sorted(plan["downloads"]) == sorted(rest)
    assert sorted(plan["present"]) == sorted(STAGE1)
    # far below the full set: exactly the remaining variants
    assert plan["need_gib"] == pytest.approx(rest_bytes / GiB, abs=0.1)


def test_all_models_low_disk_blocks_incrementally(tmp_path):
    """--all-models blocked only when the REMAINING bytes don't fit."""
    _mkweights(tmp_path, _pinned_sizes(STAGE1))
    plan = plan_stage.plan_model_staging(
        weights_root=tmp_path, all_models=True, free_bytes=10 * GiB)
    assert plan["verdict"] == "BLOCKED"
    # Assert incremental staging semantics, not a substring in a temporary
    # directory name (which may happen to contain the digits "61").
    rest = [rel for rel in FULL if rel not in STAGE1]
    remaining_gib = sum(_pinned_sizes(rest).values()) / GiB
    assert sorted(plan["downloads"]) == sorted(rest)
    assert sorted(plan["present"]) == sorted(STAGE1)
    assert plan["need_gib"] == pytest.approx(remaining_gib, abs=0.05)
    assert "Already-present pinned files were NOT counted" in plan["message"]


def test_smoke_only_never_disk_blocked(tmp_path):
    """--smoke-only downloads nothing: a nearly-full disk must not block it
    (the release blocker: it used to require 25.5 GiB unconditionally)."""
    plan = plan_stage.plan_model_staging(
        weights_root=tmp_path, smoke_only=True, free_bytes=512 * 2**20)
    assert plan["verdict"] == "OK"
    assert plan["need_gib"] == 0.0


def test_corrupted_size_counts_as_download(tmp_path):
    """A present-but-wrong-size file (torn download) is flagged BROKEN and
    its full size counts toward the budget again."""
    sizes = _pinned_sizes(STAGE1)
    torn = STAGE1[0]
    files = dict(sizes)
    files[torn] = sizes[torn] // 3  # truncated
    _mkweights(tmp_path, files)
    plan = plan_stage.plan_model_staging(
        weights_root=tmp_path, free_bytes=ALL_BYTES + 4 * GiB)
    assert torn in plan["broken"]
    assert torn in plan["downloads"]
    assert plan["need_gib"] == pytest.approx(sizes[torn] / GiB, abs=0.05)


def test_restarted_instance_plan_is_idempotent(tmp_path):
    """Restarted instance: full state -> same plan on every invocation
    (nothing to download, nothing to quarantine)."""
    _mkweights(tmp_path, _pinned_sizes(FULL))
    a = plan_stage.plan_model_staging(weights_root=tmp_path, all_models=True,
                                      free_bytes=5 * GiB)
    b = plan_stage.plan_model_staging(weights_root=tmp_path, all_models=True,
                                      free_bytes=5 * GiB)
    assert a == b
    assert a["verdict"] == "OK" and a["downloads"] == []


# ------------------------------------------------------------- comfyui staging
def _mkcomfy(tree: Path, marker=None, version="0.43.0", main=True):
    tree.mkdir(parents=True, exist_ok=True)
    if main:
        (tree / "main.py").write_text("# comfyui\n")
    if version is not None:
        (tree / "comfyui_version.py").write_text(
            f'__version__ = "{version}"\n')
    if marker is not None:
        (tree / ".lab_pinned_commit").write_text(marker + "\n")


def test_comfyui_absent_stages_fresh(tmp_path):
    r = plan_stage.classify_comfyui(tmp_path / "ComfyUI")
    assert r["action"] == "stage_fresh"


def test_comfyui_valid_tree_kept(tmp_path):
    d = tmp_path / "ComfyUI"
    _mkcomfy(d, marker=plan_stage.COMFY_PIN)
    assert plan_stage.classify_comfyui(d)["action"] == "keep"


def test_comfyui_unknown_tree_quarantined_not_deleted(tmp_path):
    """A user's own/unknown ComfyUI checkout is NEVER deleted — the plan
    must say quarantine (move aside), and classification must not touch it."""
    d = tmp_path / "ComfyUI"
    _mkcomfy(d, marker=None, version=None, main=False)
    (d / "user_custom_node.py").write_text("print('my work')\n")
    r = plan_stage.classify_comfyui(d)
    assert r["action"] == "quarantine_then_stage"
    assert (d / "user_custom_node.py").exists()  # nothing was touched


def test_comfyui_wrong_pin_quarantined(tmp_path):
    d = tmp_path / "ComfyUI"
    _mkcomfy(d, marker="deadbeef" * 5)
    assert plan_stage.classify_comfyui(d)["action"] == "quarantine_then_stage"


def test_comfyui_partial_extraction_quarantined(tmp_path):
    """Marker matches but main.py missing (torn extraction) -> quarantine,
    not a silent 'keep' of a broken tree."""
    d = tmp_path / "ComfyUI"
    _mkcomfy(d, marker=plan_stage.COMFY_PIN, main=False)
    assert plan_stage.classify_comfyui(d)["action"] == "quarantine_then_stage"


def test_comfyui_path_is_a_file_refuses(tmp_path):
    f = tmp_path / "ComfyUI"
    f.write_text("not a directory")
    r = plan_stage.classify_comfyui(f)
    assert r["action"] == "refuse"
    assert f.read_text() == "not a directory"  # untouched


def test_verify_staged_tree_accepts_valid(tmp_path):
    tree = tmp_path / "tree"
    _mkcomfy(tree)
    assert plan_stage.verify_staged_tree(tree)["ok"] is True


def test_verify_staged_tree_rejects_wrong_version(tmp_path):
    tree = tmp_path / "tree"
    _mkcomfy(tree, version="0.99.0")
    r = plan_stage.verify_staged_tree(tree)
    assert r["ok"] is False and "0.99.0" in r["reason"]


def test_verify_staged_tree_rejects_missing_main(tmp_path):
    tree = tmp_path / "tree"
    _mkcomfy(tree, main=False)
    assert plan_stage.verify_staged_tree(tree)["ok"] is False
