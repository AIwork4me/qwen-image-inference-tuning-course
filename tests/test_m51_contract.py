"""CPU-only tests for M5.1 pure contracts (no GPU, ROCm, OpenVINO or 8B model)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lab import m51_contract as C


def test_trim_semantics_second_im_start():
    ids = [151644, 10, 11, 151644, 12, 13]
    assert C.trim_at_second_im_start(ids) == 3
    assert C.trim_at_second_im_start([1, 2, 3]) == 0


def test_cond_descriptor_contract():
    assert C.validate_cond_descriptor([1, 52, 4096], "torch.float32", True)["ok"]
    assert not C.validate_cond_descriptor([52, 4096], "torch.float32", True)["ok"]
    assert not C.validate_cond_descriptor([1, 52, 4095], "torch.float32", True)["ok"]
    assert not C.validate_cond_descriptor([1, 52, 4096], "torch.bfloat16", True)["ok"]
    assert not C.validate_cond_descriptor([1, 52, 4096], "torch.float32", False)["ok"]


def test_numerical_gate_fp32_bf16_int8_policies():
    th_fp32 = {"rel_l2": 1e-4, "cosine": 0.99999}
    th_bf16 = {"rel_l2": 2e-2, "cosine": 0.999}
    th_int8 = {"rel_l2": 8e-2, "cosine": 0.995}
    assert C.numerical_gate({"rel_l2": 4e-6, "cosine": 1.0, "finite": True}, th_fp32)["status"] == "PASS"
    assert C.numerical_gate({"rel_l2": 1.3e-2, "cosine": 0.9999, "finite": True}, th_bf16)["status"] == "PASS"
    assert C.numerical_gate({"rel_l2": 6.4e-2, "cosine": 0.998, "finite": True}, th_int8)["status"] == "PASS"
    assert C.numerical_gate({"rel_l2": 0.28, "cosine": 0.96, "finite": True}, th_int8)["status"] == "FAIL"


def test_admission_classification():
    go = {"gpu": {"gfx_arch": "gfx1100", "vram_mib": 49136, "device_count": 1},
          "torch": {"cuda_available": True, "device_count": 1}}
    assert C.classify_admission(go)[0] == "GO"
    no_gpu = {"gpu": {}, "torch": {"cuda_available": False, "device_count": 0}}
    assert C.classify_admission(no_gpu)[0] == "BLOCKED_ENVIRONMENT"
    wrong_arch = {"gpu": {"gfx_arch": "gfx942", "vram_mib": 49136, "device_count": 1},
                  "torch": {"cuda_available": True, "device_count": 1}}
    assert C.classify_admission(wrong_arch)[0] == "GO_WITH_DEVIATIONS"


def test_artifact_absence_and_corruption_are_blocked(tmp_path):
    req = ["model.bin", "model.xml"]
    r = C.artifact_status(tmp_path, req)
    assert r["status"] == "BLOCKED" and r["reason"] == "missing_artifacts"
    (tmp_path / "model.bin").write_bytes(b"x"); (tmp_path / "model.xml").write_text("x")
    assert C.artifact_status(tmp_path, req)["reason"] == "missing_manifest"
    (tmp_path / "conversion_manifest.json").write_text("{bad json")
    assert C.artifact_status(tmp_path, req)["reason"] == "corrupt_manifest"
    (tmp_path / "conversion_manifest.json").write_text('{"source_sha256": "abc"}')
    assert C.artifact_status(tmp_path, req)["status"] == "PASS"


def test_budget_rejection():
    assert C.budget_check(10, 5, 10, 5)["status"] == "PASS"
    assert C.budget_check(1, 5, 10, 5)["status"] == "BLOCKED"
    assert C.budget_check(10, 5, 1, 5)["status"] == "BLOCKED"


def test_safe_cond_path():
    root = Path("/tmp/run")
    assert C.safe_cond_path("/tmp/run/a.safetensors", root).name == "a.safetensors"
    for bad in ("/tmp/other/a.safetensors", "/tmp/run/../a.safetensors", "/tmp/run/a.pt"):
        try:
            C.safe_cond_path(bad, root)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted unsafe path {bad}")


def test_performance_stats_and_schema():
    s = C.performance_stats([1.0, 2.0, 3.0, 4.0])
    assert s["n"] == 4 and s["median"] == 2.5
    assert C.performance_stats([])["n"] == 0
    ok = {"experiment": "H3", "status": "PASS", "timestamp": "t", "source_head": "h",
          "artifact_digest": "d", "environment": {}, "measurements": {"x": 1},
          "evidence_paths": []}
    assert C.validate_result_schema(ok) == []
    bad = dict(ok); bad["status"] = "MAYBE"; assert C.validate_result_schema(bad)
    empty_pass = dict(ok); empty_pass["measurements"] = {}; assert C.validate_result_schema(empty_pass)


def test_never_downgrade_blocked_to_pass():
    assert C.honest_status("PASS", {"blocked": True, "measured": True}) == "BLOCKED"
    assert C.honest_status("PASS", {"blocked": False, "measured": False}) == "NOT_RUN"
    assert C.honest_status("PASS", {"blocked": False, "measured": True}) == "PASS"
