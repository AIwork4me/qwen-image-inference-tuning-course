"""CPU-only notebook execution, evidence integrity, and mode-safety tests.

Layer A of the quality gates: runs on GitHub-hosted CPU CI without any GPU.
The 18 notebooks are executed in REFERENCE mode in REAL Jupyter kernels
(nbclient + ipykernel) when available, with an exec() fallback so a plain
pytest environment still executes every code cell.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from lab import notebook_runtime as rt
from lab import notebook_support as support

ROOT = Path(__file__).resolve().parents[1]
CATALOG = json.loads((ROOT / "notebooks/catalog.json").read_text(encoding="utf-8"))
FILES = [p for row in CATALOG["modules"] for p in (row["english"], row["chinese"])]


def _execute_reference(filename: str, capsys):
    """Execute one notebook's code cells in REFERENCE mode."""
    notebook = json.loads((ROOT / "notebooks" / filename).read_text(encoding="utf-8"))
    try:
        import nbclient  # noqa: F401
        import nbformat
        from nbclient import NotebookClient

        nb = nbformat.read(str(ROOT / "notebooks" / filename), as_version=4)
        client = NotebookClient(nb, timeout=300, kernel_name="python3",
                                resources={"metadata": {"path": str(ROOT)}},
                                allow_errors=False)
        client.execute()
        outs = []
        for cell in nb.cells:
            for o in cell.get("outputs", []):
                if o.get("output_type") == "stream":
                    outs.append("".join(o.get("text", "")))
        return "".join(outs)
    except ImportError:
        scope = {"__name__": "__course_cpu_notebook__"}
        import os
        old = os.environ.get("QIR_COURSE_MODE")
        os.environ["QIR_COURSE_MODE"] = "REFERENCE"
        try:
            for i, cell in enumerate(notebook["cells"]):
                if cell["cell_type"] != "code":
                    continue
                text = "".join(cell["source"])
                exec(compile(text, f"{filename}:cell{i}", "exec"), scope)
        finally:
            if old is None:
                os.environ.pop("QIR_COURSE_MODE", None)
            else:
                os.environ["QIR_COURSE_MODE"] = old
        return capsys.readouterr().out


@pytest.mark.parametrize("filename", FILES)
def test_notebook_reference_run_all(filename, monkeypatch, capsys):
    monkeypatch.chdir(ROOT)
    monkeypatch.setenv("QIR_COURSE_MODE", "REFERENCE")
    out = _execute_reference(filename, capsys)
    assert "MODE=REFERENCE" in out, "the session banner must declare REFERENCE"
    assert "no GPU work launched" in out
    assert "[MEASURED_TODAY]" not in out, "REFERENCE mode must not claim measurements"


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


def test_reference_mode_never_imports_torch():
    import subprocess
    import sys
    r = subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.path.insert(0, '.');"
         "from lab import notebook_runtime as rt;"
         "s = rt.begin('REFERENCE', notebook='t');"
         "assert 'torch' not in sys.modules; print('CLEAN')"],
        cwd=str(ROOT), capture_output=True, text=True, timeout=120)
    assert "CLEAN" in r.stdout, r.stdout + r.stderr


def test_env_mode_override_pins_ci_to_reference(monkeypatch):
    monkeypatch.setenv("QIR_COURSE_MODE", "REFERENCE")
    assert rt.resolve_mode("GPU_CLASSROOM") == "REFERENCE"


def test_unknown_mode_is_never_a_silent_fallback():
    with pytest.raises(rt.CourseModeError):
        rt.resolve_mode("TURBO")


def test_gpu_smoke_still_opt_in():
    with pytest.MonkeyPatch.context() as mp:
        mp.chdir(ROOT)
        mp.delenv("QIR_NOTEBOOK_GPU_ACK", raising=False)
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


def test_sessions_and_records_stay_under_runs(tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)
    monkeypatch.delenv("QIR_NOTEBOOK_CAMPAIGN", raising=False)
    s = rt.Session("REFERENCE", "unit-test")
    s.admission = rt.admit_hardware("REFERENCE")
    p = s.finish({"summary": "ok"})
    assert "/runs/" in str(p)
    assert "evidence" not in str(p)
