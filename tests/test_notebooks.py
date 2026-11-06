"""CPU-only notebook execution, evidence integrity, and opt-in safety tests.

Runs the actual executable cells of all 18 notebooks without a Jupyter kernel.
No notebook GPU flag is enabled and all write actions default to OFF.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from lab import notebook_support as support

ROOT = Path(__file__).resolve().parents[1]
CATALOG = json.loads((ROOT / "notebooks/catalog.json").read_text(encoding="utf-8"))
FILES = [p for row in CATALOG["modules"] for p in (row["english"], row["chinese"])]


@pytest.mark.parametrize("filename", FILES)
def test_notebook_cpu_run_all(filename, monkeypatch, capsys):
    monkeypatch.chdir(ROOT)
    notebook = json.loads((ROOT / "notebooks" / filename).read_text(encoding="utf-8"))
    scope = {"__name__": "__course_cpu_notebook__"}
    for i, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = cell["source"]
        text = "".join(source) if isinstance(source, list) else source
        exec(compile(text, f"{filename}:cell{i}", "exec"), scope)
    out = capsys.readouterr().out
    assert "GPU NOT_RUN" in out or "NO GPU work" in out or "GPU" in out
    assert scope.get("RUN_GPU") is not True
    assert scope.get("WRITE_STUDENT_RECORD") is not True


def test_notebook_quality_script_parity():
    import runpy
    m = runpy.run_path(str(ROOT / "scripts/course-quality/check_notebooks.py"), run_name="check")
    assert m["run"]() == []


def test_official_baseline_18_reference_only():
    with pytest.MonkeyPatch.context() as mp:
        mp.chdir(ROOT)
        raw = support.reference_warm("B0")
        assert len(raw) == 18
        assert abs(support.summary_rows()["B0"]["warm_total_median_s"] - 55.206) < 0.01


def test_fair_factorial_populations_not_conflated():
    with pytest.MonkeyPatch.context() as mp:
        mp.chdir(ROOT)
        original = support.fair_factorial()
        clean = support.fair_factorial(clean_head=True)
        assert all(v["n"] == 6 for v in original["cells"].values())
        assert all(v["n"] == 2 for v in clean["cells"].values())
        assert original["source"] != clean["source"]


def test_gpu_opt_in_denied_by_default(monkeypatch):
    monkeypatch.delenv("QIR_NOTEBOOK_GPU_ACK", raising=False)
    with pytest.raises(PermissionError):
        support.gpu_smoke("B0", "student-b0", confirmed=True, root=ROOT)
    with pytest.raises(ValueError):
        support.gpu_smoke("H2", "bad-unbounded", confirmed=True, root=ROOT)


def test_student_record_no_unprompted_write(tmp_path):
    with pytest.raises(PermissionError):
        support.student_record({"status": "example"}, "example", root=tmp_path)
    assert not (tmp_path / "runs").exists()
    out = support.student_record({"status": "REFERENCE_ANALYSIS_NOT_NEW_GPU_RUN"},
                                 "student-example", confirmed=True, root=tmp_path)
    assert out.is_file() and out.is_relative_to(tmp_path / "runs")
    with pytest.raises(FileExistsError):
        support.student_record({}, "student-example", confirmed=True, root=tmp_path)


def test_outside_repository_json_denied(tmp_path):
    with pytest.raises(ValueError):
        support.read_json("../no.json", ROOT)
