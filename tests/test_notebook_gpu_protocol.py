"""GPU protocol safety tests (CPU-safe: no physical GPU is touched).

These are the NEGATIVE tests of the GPU-first notebook runtime: every guard
that must fail-cail is exercised here with a fake torch, so public CPU CI
still verifies the physical-GPU safety semantics.
"""
from __future__ import annotations

import json
import subprocess
import sys
import types
from pathlib import Path

import pytest

from lab import notebook_runtime as rt

ROOT = Path(__file__).resolve().parents[1]


class _FakeProps:
    def __init__(self, name="AMD Radeon Graphics", arch="gfx1100",
                 total=48 * 2**30, count=1):
        self.gcnArchName = arch
        self.total_memory = total
        _FakeProps._count = count


def _install_fake_torch(monkeypatch, *, available=True, name="AMD Radeon Graphics",
                        arch="gfx1100", device_count=1, free=None, total=48 * 2**30):
    fake = types.ModuleType("torch")
    fake.cuda = types.SimpleNamespace(
        is_available=lambda: available,
        device_count=lambda: device_count,
        get_device_name=lambda i=0: name,
        get_device_properties=lambda i=0: _FakeProps(name, arch, total),
        mem_get_info=lambda: (free if free is not None else total - 2 * 2**30, total),
        synchronize=lambda: None,
    )
    fake.__version__ = "2.9.1+fake"
    fake.version = types.SimpleNamespace(hip="7.2-fake")
    monkeypatch.setitem(sys.modules, "torch", fake)


def test_gpu_mode_requires_usable_gpu(monkeypatch):
    _install_fake_torch(monkeypatch, available=False)
    with pytest.raises(rt.AdmissionError, match="no usable ROCm/HIP device"):
        rt.admit_hardware("GPU_CLASSROOM")


def test_wrong_device_identity_blocked(monkeypatch):
    _install_fake_torch(monkeypatch, name="NVIDIA RTX 6000")
    with pytest.raises(rt.AdmissionError, match="not AMD"):
        rt.admit_hardware("GPU_CLASSROOM")


def test_busy_gpu_blocked(monkeypatch):
    total = 48 * 2**30
    _install_fake_torch(monkeypatch, free=total - 40 * 2**30)
    with pytest.raises(rt.AdmissionError, match="another workload"):
        rt.admit_hardware("GPU_CLASSROOM")


def test_multi_gpu_blocked(monkeypatch):
    _install_fake_torch(monkeypatch, device_count=2)
    with pytest.raises(rt.AdmissionError, match="exactly 1 GPU"):
        rt.admit_hardware("GPU_CLASSROOM")


def test_non_reference_class_gpu_warns_but_passes(monkeypatch):
    _install_fake_torch(monkeypatch, arch="gfx942")
    a = rt.admit_hardware("GPU_CLASSROOM")
    assert a.status == "PASS_WITH_WARNINGS"
    assert any("not gfx1100" in w for w in a.warnings)


def test_reference_execution_guard(monkeypatch):
    s = rt.Session("REFERENCE", "t")
    s.admission = rt.admit_hardware("REFERENCE")
    with pytest.raises(rt.CourseModeError, match="disabled in REFERENCE"):
        s.run("B0", profile="classroom")


def test_research_profile_denied_in_classroom_mode(monkeypatch):
    _install_fake_torch(monkeypatch)
    s = rt.Session("GPU_CLASSROOM", "t")
    s.admission = rt.admit_hardware("GPU_CLASSROOM")
    with pytest.raises(rt.CourseModeError, match="restricted to GPU_RESEARCH"):
        s.run("B0", profile="research")


def test_classroom_profile_is_bounded(tmp_path, monkeypatch):
    """The classroom invocation must carry only_prompt + bounded warmup/
    measured counts — never the full protocol."""
    _install_fake_torch(monkeypatch)
    # A simulated CLI success must also create the physical run directory,
    # so the real fail-closed provenance binder can verify it. Never fake
    # successful provenance or write test artifacts into the repository.
    monkeypatch.setattr(rt.lab_env, "REPO", tmp_path)
    monkeypatch.setattr(rt.lab_env, "STUDENT_RUNS_ROOT", tmp_path / "runs")
    monkeypatch.setattr(rt.prov, "REPO", tmp_path)
    monkeypatch.setattr(rt.prov, "source_identity",
                        lambda repo=None: {"head": "test-only-commit", "dirty": False})
    captured = {}

    def fake_run(cmd, **kw):
        captured["cmd"] = cmd
        (tmp_path / "runs" / "unit-bounded-B0").mkdir(parents=True, exist_ok=True)
        class R:
            returncode = 0
            stdout = ""
            stderr = ""
        return R()

    monkeypatch.setattr(rt, "_acquire_gpu_lock", lambda t=600: 1)
    monkeypatch.setattr(rt, "_release_gpu_lock", lambda: None)
    rec = {"experiment": "B0", "run_kind": "measure", "prompt_id": "p1-zh-scene",
           "steps": 40, "precision": "bf16", "attention_backend": "x",
           "unet": "u", "vram_peak_mb": 1,
           "runs": [{"kind": "warm", "total_s": 55.0, "text_encode_s": 0.5,
                     "denoise_s": 50.0, "vae_decode_s": 2.0, "step_times_s": [1]*40}]}
    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(rt, "load_run",
                        lambda run_id, root=None: rt.RunBundle(
                            run_id, ROOT, [rec]))
    s = rt.Session("GPU_CLASSROOM", "t")
    s.admission = rt.admit_hardware("GPU_CLASSROOM")
    bundle = s.run("B0", profile="classroom", tag="unit-bounded")
    cmd = captured["cmd"]
    assert "--only-prompt" in cmd and "p1-zh-scene" in cmd
    assert "--smoke" not in cmd
    assert bundle.stats()["population"].startswith("classroom-bounded")
    # The production provenance contract is exercised, not bypassed.
    binding = json.loads((tmp_path / "runs" / "unit-bounded-B0" /
                          "PROVENANCE.json").read_text(encoding="utf-8"))
    assert binding["session_id"] == s.session_id
    assert binding["run_origin"] == "CURRENT_SESSION_PHYSICAL"


def test_smoke_population_never_counted_as_formal(tmp_path):
    base = rt.RunBundle("x-smoke", ROOT, [_rec("B0", 55.0)])
    cand = rt.RunBundle("y-smoke", ROOT, [_rec("L1", 12.0)])
    ab = rt.ab_compare(base, cand)
    assert ab["classification"] == "BOUNDED_PHYSICAL_COMPARISON"
    assert any("smoke" in c or "bounded" in c for c in ab["caveats"])


def test_bounded_single_prompt_never_formal(tmp_path):
    base = rt.RunBundle("q1-B0", ROOT, [_rec("B0", 55.0)])
    cand = rt.RunBundle("q1-L1", ROOT, [_rec("L1", 12.0)])
    ab = rt.ab_compare(base, cand)
    assert ab["classification"] == "BOUNDED_PHYSICAL_COMPARISON"


def test_formal_classification_requires_full_population():
    recs_b = [_rec("B0", 55.0 + i * 0.1, pid=p, kind="measure")
              for p in ("p1-zh-scene", "p2-en-scene", "p3-poster-text")
              for i in range(3)]
    recs_l = [_rec("L1", 12.0 + i * 0.05, pid=p, kind="measure")
              for p in ("p1-zh-scene", "p2-en-scene", "p3-poster-text")
              for i in range(3)]
    ab = rt.ab_compare(rt.RunBundle("q1-B0", ROOT, recs_b),
                       rt.RunBundle("q1-L1", ROOT, recs_l))
    assert ab["classification"] == "FORMAL_PROTOCOL_MEASURED"
    assert len(ab["paired_per_prompt"]) == 3


def test_gpu_lock_is_pid_owned(tmp_path, monkeypatch):
    lock = ROOT / "runs" / ".notebook_gpu.lock"
    monkeypatch.setattr(rt, "_gpu_lock_path", lambda: lock)
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text("1")  # pid 1 always exists -> holder alive

    class _Clock:
        t = 0.0
        @staticmethod
        def time():
            _Clock.t += 1.0
            return _Clock.t
        @staticmethod
        def sleep(_s):
            return None
    monkeypatch.setattr(rt, "time", _Clock)
    with pytest.raises(rt.AdmissionError, match="one GPU workload at a time"):
        rt._acquire_gpu_lock(timeout_s=0.5)
    lock.write_text("999999999")  # dead pid -> stale lock is taken over
    got = rt._acquire_gpu_lock(timeout_s=5)
    assert got is not None
    rt._release_gpu_lock()
    assert not lock.exists()


def test_frozen_evidence_never_written_by_runtime(tmp_path, monkeypatch):
    """All runtime outputs must land under runs/, never evidence/."""
    monkeypatch.delenv("QIR_NOTEBOOK_CAMPAIGN", raising=False)
    monkeypatch.chdir(ROOT)
    s = rt.Session("REFERENCE", "t")
    s.admission = rt.admit_hardware("REFERENCE")
    p = s.finish({})
    assert p.is_relative_to(ROOT / "runs")
    assert "evidence" not in p.parts


def _rec(experiment: str, total: float, pid: str = "p1-zh-scene",
         kind: str = "measure") -> dict:
    return {"experiment": experiment, "run_kind": kind, "prompt_id": pid,
            "steps": 40 if experiment in ("B0", "F1", "F2") else 8,
            "precision": "bf16", "attention_backend": "pytorch-sdpa-default",
            "unet": "qwen_image_2.1_bf16.safetensors",
            "vram_peak_mb": 33000, "vram_device_peak_mb": 38000,
            "runs": [{"kind": "warm", "total_s": total, "text_encode_s": 0.5,
                      "denoise_s": total - 5, "vae_decode_s": 2.0,
                      "step_times_s": [1.3] * 40}]}
