"""CPU-only tests for M5.1 pure contracts (no GPU, ROCm, OpenVINO or 8B model).

Extended for the PR #2 release closure:
- byte-level artifact verification (tamper/truncate/swap/identity/symlink/cache)
- release-gate propagation (execution != scientific success; INT8 negative
  failure blocks fidelity; BLOCKED never upgraded; human review never
  auto-PASSed; missing evidence never PASS)
- conservative RAM admission (finite + unlimited cgroups, host fallback)
- empty-negative tokenizer contract (production vs prevent_empty_text=True)
"""
import json
import sys
from pathlib import Path

import pytest

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


# ---------------------------------------------------------------------------
# P0-B: byte-level artifact verification
# ---------------------------------------------------------------------------

def _make_artifact(root: Path, xml: bytes = b"<net>", bin_: bytes = b"weights", **manifest_over):
    (root / "model.xml").write_bytes(xml)
    (root / "model.bin").write_bytes(bin_)
    m = {
        "source_sha256": "0" * 64,
        "output_xml_sha256": C.streaming_sha256(root / "model.xml"),
        "output_bin_sha256": C.streaming_sha256(root / "model.bin"),
        "output_bin_bytes": len(bin_),
    }
    m.update(manifest_over)
    (root / "conversion_manifest.json").write_text(json.dumps(m))
    return m


def test_verify_artifact_pass_and_cache(tmp_path):
    root = tmp_path / "arm"; root.mkdir()
    _make_artifact(root)
    r = C.verify_artifact(root, ["model.xml", "model.bin"])
    assert r["status"] == "PASS", r
    assert not (root / ".m51_verification_cache.json").is_file()  # default: no persistent cache
    r1 = C.verify_artifact(root, ["model.xml", "model.bin"], use_cache=True)
    assert r1["status"] == "PASS"
    assert (root / ".m51_verification_cache.json").is_file()
    r2 = C.verify_artifact(root, ["model.xml", "model.bin"], use_cache=True)
    assert r2["status"] == "PASS" and r2["files"]["model.bin"]["cached"] is True


def test_verify_artifact_modified_bin_blocked(tmp_path):
    root = tmp_path / "arm"; root.mkdir()
    _make_artifact(root)
    (root / "model.bin").write_bytes(b"weightsX")
    r = C.verify_artifact(root, ["model.xml", "model.bin"])
    assert r["status"] == "BLOCKED" and r["reason"] in ("size_mismatch", "digest_mismatch")


def test_verify_artifact_modified_xml_blocked_even_with_valid_size(tmp_path):
    root = tmp_path / "arm"; root.mkdir()
    _make_artifact(root, xml=b"<net>")  # 5 bytes
    (root / "model.xml").write_bytes(b"<net2")  # same length, different bytes
    r = C.verify_artifact(root, ["model.xml", "model.bin"])
    assert r["status"] == "BLOCKED" and r["reason"] == "digest_mismatch"


def test_verify_artifact_incorrect_digest_blocked(tmp_path):
    root = tmp_path / "arm"; root.mkdir()
    _make_artifact(root, output_bin_sha256="f" * 64)
    r = C.verify_artifact(root, ["model.xml", "model.bin"])
    assert r["status"] == "BLOCKED" and r["reason"] == "digest_mismatch"


def test_verify_artifact_truncated_file_blocked(tmp_path):
    root = tmp_path / "arm"; root.mkdir()
    _make_artifact(root, bin_=b"w" * 1024)
    with open(root / "model.bin", "r+b") as f:
        f.truncate(512)
    r = C.verify_artifact(root, ["model.xml", "model.bin"])
    assert r["status"] == "BLOCKED" and r["reason"] == "size_mismatch"


def test_verify_artifact_swapped_artifact_identity_blocked(tmp_path):
    # whole fp32 dir copied into the bf16 slot: digests match their own
    # manifest, but the expected bf16 identity does not
    root = tmp_path / "arm"; root.mkdir()
    _make_artifact(root, dtype="fp32")
    r = C.verify_artifact(root, ["model.xml", "model.bin"],
                          expected_identity={"storage": "FP16", "runtime_precision": "bf16"})
    assert r["status"] == "BLOCKED" and r["reason"] == "identity_mismatch"


def test_verify_artifact_identity_pass(tmp_path):
    root = tmp_path / "arm"; root.mkdir()
    _make_artifact(root, storage="FP16", runtime_precision="bf16")
    r = C.verify_artifact(root, ["model.xml", "model.bin"],
                          expected_identity={"storage": "FP16", "runtime_precision": "bf16"})
    assert r["status"] == "PASS"


def test_verify_artifact_missing_and_corrupt_manifest(tmp_path):
    root = tmp_path / "arm"; root.mkdir()
    assert C.verify_artifact(root, ["model.xml"])["reason"] == "missing_artifacts"
    (root / "model.xml").write_bytes(b"x")
    assert C.verify_artifact(root, ["model.xml"])["reason"] == "missing_manifest"
    (root / "conversion_manifest.json").write_text("{oops")
    assert C.verify_artifact(root, ["model.xml"])["reason"] == "corrupt_manifest"
    (root / "conversion_manifest.json").write_text("{}")
    assert C.verify_artifact(root, ["model.xml"])["reason"] == "manifest_has_no_source_digest"


def test_verify_artifact_symlink_escape_blocked(tmp_path):
    root = tmp_path / "arm"; root.mkdir()
    outside = tmp_path / "outside"; outside.mkdir()
    _make_artifact(outside)
    (root / "model.xml").write_bytes(b"<net>")
    (root / "model.bin").symlink_to(outside / "model.bin")  # file symlink escape
    m = {"source_sha256": "0" * 64,
         "output_xml_sha256": C.streaming_sha256(root / "model.xml"),
         "output_bin_sha256": C.streaming_sha256(outside / "model.bin"),
         "output_bin_bytes": 7}
    (root / "conversion_manifest.json").write_text(json.dumps(m))
    r = C.verify_artifact(root, ["model.xml", "model.bin"])
    assert r["status"] == "BLOCKED" and r["reason"] == "unsafe_path"


def test_verify_artifact_same_size_tamper_blocked_without_cache(tmp_path):
    # Default policy: no persistent cache, so a same-size tamper (the classic
    # size+mtime cache blind spot on coarse-granularity filesystems) is always
    # caught by re-hashing at admission time.
    root = tmp_path / "arm"; root.mkdir()
    _make_artifact(root)
    assert C.verify_artifact(root, ["model.xml", "model.bin"])["status"] == "PASS"
    (root / "model.bin").write_bytes(b"WEIGHTS")  # same length, different bytes
    r = C.verify_artifact(root, ["model.xml", "model.bin"])
    assert r["status"] == "BLOCKED" and r["reason"] == "digest_mismatch"
    assert not (root / ".m51_verification_cache.json").is_file()


def test_verify_artifact_optin_cache_size_change_invalidates(tmp_path):
    root = tmp_path / "arm"; root.mkdir()
    _make_artifact(root)
    assert C.verify_artifact(root, ["model.xml", "model.bin"], use_cache=True)["status"] == "PASS"
    (root / "model.bin").write_bytes(b"weights!")  # different size
    r = C.verify_artifact(root, ["model.xml", "model.bin"], use_cache=True)
    assert r["status"] == "BLOCKED" and r["reason"] in ("size_mismatch", "digest_mismatch")


def test_verify_artifact_unpinned_xml_warns_not_silent(tmp_path):
    # legacy bf16 manifest pins only the bin digest
    root = tmp_path / "arm"; root.mkdir()
    (root / "model.xml").write_bytes(b"<net>")
    (root / "model.bin").write_bytes(b"w")
    m = {"source_xml_sha256": "0" * 64,
         "output_bin_sha256": C.streaming_sha256(root / "model.bin"),
         "output_bin_bytes": 1}
    (root / "conversion_manifest.json").write_text(json.dumps(m))
    r = C.verify_artifact(root, ["model.xml", "model.bin"])
    assert r["status"] == "PASS_WITH_WARNINGS"
    assert any("model.xml" in w for w in r["warnings"])
    assert r["files"]["model.xml"]["unpinned"] is True


# ---------------------------------------------------------------------------
# P0-A: release-gate propagation
# ---------------------------------------------------------------------------

def _gates(**over):
    g = {k: "PASS" for k in C.GATE_ORDER}
    g["human_quality"] = "PENDING"
    g.update(over)
    return g


def test_successful_encoder_is_not_numerical_pass():
    g = _gates(conditioning_alignment="FAIL")
    assert g["encoder_runtime"] == "PASS"
    assert C.release_status(g) == "RELEASE_FAIL"


def test_int8_negative_failure_blocks_fidelity_claim():
    # the declared INT8 empty-negative failure (relL2 28.25%) must surface as
    # a conditioning FAIL that no execution PASS can outweigh
    g = _gates(artifact_integrity="PASS", encoder_runtime="PASS",
               token_contract="PASS", conditioning_alignment="FAIL")
    assert C.release_status(g) == "RELEASE_FAIL"


def test_missing_evidence_cannot_become_pass():
    g = _gates(conditioning_alignment="NOT_RUN", e2e_gpu="NOT_RUN")
    assert C.release_status(g) == "PENDING_HUMAN_REVIEW"  # human pending dominates
    g2 = _gates(human_quality="PASS", conditioning_alignment="NOT_RUN")
    assert C.release_status(g2) == "INCOMPLETE_SCOPE"


def test_blocked_experiment_never_upgraded_by_summary():
    g = _gates(artifact_integrity="BLOCKED", encoder_runtime="PASS",
               conditioning_alignment="PASS", human_quality="PASS")
    assert C.release_status(g) == "BLOCKED"
    g["environment"] = "BLOCKED"
    assert C.release_status(g) == "BLOCKED"


def test_human_review_not_auto_passed_by_numeric_checks():
    g = _gates()  # everything PASS except human PENDING
    assert C.release_status(g) == "PENDING_HUMAN_REVIEW"
    # numeric PASSes cannot set human_quality
    C.settle_human_quality(g, None)
    assert g["human_quality"] == "PENDING"
    C.settle_human_quality(g, {"verdict": "PASS"})  # no reviewer -> not accepted
    assert g["human_quality"] == "PENDING"
    C.settle_human_quality(g, {"reviewer": "owner", "verdict": "pass"})  # explicit human record
    assert g["human_quality"] == "PASS"
    assert C.release_status(g) == "RELEASE_PASS"


def test_release_status_rejects_unknown_gate_states():
    with pytest.raises(ValueError):
        C.release_status(_gates(encoder_runtime="MAYBE"))


def test_h3_specs_pin_identity_and_budget():
    from lab import m51_run
    assert m51_run.SPECS["H3_OV_BF16"]["identity"] == {"storage": "FP16", "runtime_precision": "bf16"}
    assert m51_run.SPECS["H3_OV_INT8"]["identity"] == {"mode": "INT8_ASYM"}
    assert m51_run.SPECS["H3_OV_FP32"]["identity"] == {"dtype": "fp32"}
    assert m51_run.SPECS["H3_OV_BF16"]["need_ram"] > 30 * 2**30


def test_artifact_root_env_portability(monkeypatch):
    from lab import m51_run
    monkeypatch.setenv("M51_OV_ARTIFACTS", "/tmp/opencode/some-other-root")
    assert m51_run.artifact_root() == Path("/tmp/opencode/some-other-root")
    monkeypatch.delenv("M51_OV_ARTIFACTS")
    assert m51_run.artifact_root() == m51_run.REPO / "runs" / "m51-ov-artifacts"


# ---------------------------------------------------------------------------
# P1-A: conservative RAM admission
# ---------------------------------------------------------------------------

GIB = 2**30


def test_ram_gate_uses_headroom_not_ceiling():
    # 55 GiB cgroup limit, 38 GiB non-reclaimable anon already used -> must
    # NOT advertise 55 GiB; usable ~ 17 - safety(2) = 15 GiB
    cg = {"limit_bytes": 55 * GIB, "current_bytes": 40 * GIB, "anon_bytes": 38 * GIB}
    r = C.conservative_ram_gate(cg, host_available=480 * GIB, required_bytes=14 * GIB)
    assert r["status"] == "PASS"
    assert r["usable_bytes"] == 15 * GIB  # 55-38-2
    r2 = C.conservative_ram_gate(cg, host_available=480 * GIB, required_bytes=16 * GIB)
    assert r2["status"] == "BLOCKED" and r2["reason"] == "insufficient_ram_headroom"


def test_ram_gate_unlimited_cgroup_falls_back_to_host():
    cg = {"limit_bytes": None, "current_bytes": None, "anon_bytes": 4 * GIB}
    r = C.conservative_ram_gate(cg, host_available=40 * GIB, required_bytes=30 * GIB)
    assert r["basis"] == "host_mem_available" and r["status"] == "PASS"
    r2 = C.conservative_ram_gate(cg, host_available=40 * GIB, required_bytes=45 * GIB)
    assert r2["status"] == "BLOCKED"


def test_ram_gate_takes_min_of_host_and_cgroup():
    cg = {"limit_bytes": 55 * GIB, "current_bytes": 1 * GIB, "anon_bytes": 0}
    r = C.conservative_ram_gate(cg, host_available=10 * GIB, required_bytes=7 * GIB,
                                safety_bytes=0)
    assert r["basis"] == "host_mem_available"
    assert C.conservative_ram_gate(cg, host_available=10 * GIB, required_bytes=11 * GIB,
                                   safety_bytes=0)["status"] == "BLOCKED"


def test_ram_gate_concurrent_reserve():
    cg = {"limit_bytes": 55 * GIB, "current_bytes": 20 * GIB, "anon_bytes": 18 * GIB}
    r = C.conservative_ram_gate(cg, host_available=480 * GIB, required_bytes=30 * GIB,
                                concurrent_reserve_bytes=5 * GIB)  # resident ComfyUI
    assert r["usable_bytes"] == (55 - 18 - 2 - 5) * GIB
    # 30 GiB usable vs 30 GiB + 1 byte required -> blocked
    r2 = C.conservative_ram_gate(cg, host_available=480 * GIB, required_bytes=30 * GIB + 1,
                                 concurrent_reserve_bytes=5 * GIB)
    assert r2["status"] == "BLOCKED"


def test_ram_gate_no_information_blocks():
    assert C.conservative_ram_gate({}, None, 1)["status"] == "BLOCKED"


def test_cgroup_reader_shapes():
    cg = C.read_cgroup_mem()
    assert set(cg) == {"limit_bytes", "current_bytes", "anon_bytes"}


# ---------------------------------------------------------------------------
# P0-C: empty-negative tokenizer contract
# ---------------------------------------------------------------------------

PROD_NEG = [151644, 8948, 198, 1092, 30782, 408, 323, 23643, 279, 3897, 9934, 13,
            151645, 198, 151644, 872, 198, 151645, 198, 151644, 77091, 198]


def test_token_contract_file_matches_frozen_production_ids():
    p = Path(__file__).resolve().parents[1] / "lab" / "m51_token_contract.json"
    c = json.loads(p.read_text())
    assert c["prompts"]["negative"]["ids"] == PROD_NEG
    assert 220 not in c["prompts"]["negative"]["ids"]
    # frozen campaign: OV probe (production policy) identical, manual fallback
    # inserted exactly one 220
    fb = c["prompts"]["negative"]["ids_fallback_preventTrue"]
    assert fb.count(220) == 1 and len(fb) == len(PROD_NEG) + 1
    assert c["edge_cases"]["empty"]["ids"] == PROD_NEG
    assert c["edge_cases"]["space"]["ids"] == c["edge_cases"]["space"]["ids_fallback_preventTrue"]


def test_empty_negative_explain_fallback_insertion():
    fb = PROD_NEG[:17] + [220] + PROD_NEG[17:]
    e = C.empty_negative_explain(PROD_NEG, fb)
    assert e["policy"] == "fallback_inserts_empty_token"
    assert e["inserted_fallback_token"] == 220 and e["inserted_at_index"] == 17
    assert C.empty_negative_explain(PROD_NEG, PROD_NEG)["policy"] == "identical"


# ---------------------------------------------------------------------------
# budgets, paths, stats, schema, honest status (original coverage)
# ---------------------------------------------------------------------------

def test_h3_reference_verdicts_pin_the_declared_gates():
    # The frozen cleanroom verdicts are the accuracy truth-source per arm:
    # INT8 negative FAIL must dominate any later reproducible-but-inaccurate run.
    from lab import m51_run
    assert m51_run.REFERENCE_VERDICT_KEYS == {"H3_OV_FP32": "C1a",
                                              "H3_OV_BF16": "C1b_compressed",
                                              "H3_OV_INT8": "C2"}
    verdicts = json.loads(m51_run.REFERENCE_ALIGNMENT.read_text())["verdict"]
    assert verdicts["C1a"]["status"] == "PASS"
    assert verdicts["C1b_compressed"]["status"] == "PASS"
    assert verdicts["C2"]["status"] == "FAIL"  # empty-negative relL2 28.25%


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
