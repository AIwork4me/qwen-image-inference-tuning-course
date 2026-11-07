"""CPU-only tests for the student-independence provenance contract.

Covers (student-independence closure):
- P1 provenance categories, session ids, run-id safety, bindings, and the
  fail-closed classification rules (tracked runs are never current-session;
  unbound/untracked runs are never promotable; prior sessions must be
  verifiable by an existing session record)
- P2 the M5.1 BF16 admission contract (PASS / PASS_WITH_WARNINGS / FAIL /
  BLOCKED incl. remediation) on synthetic artifact trees
- P5 student result records: every invalid promotion case from the release
  matrix fails validation
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lab import m51_contract as C
from lab import student_provenance as prov


# ---------------------------------------------------------------------------
# helpers: a disposable git repo that mimics the course layout
# ---------------------------------------------------------------------------

def make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "course"
    (repo / "runs").mkdir(parents=True)
    (repo / "evidence" / "results").mkdir(parents=True)
    (repo / "lab").mkdir()
    (repo / "lab" / "reference_runs.json").write_text("{}")
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "t@t"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "t"], check=True)
    return repo


def make_run(repo: Path, run_id: str, records=None, commit=False) -> Path:
    d = repo / "runs" / run_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "RESULTS.json").write_text(json.dumps(records or {"experiment": "X"}))
    if commit:
        subprocess.run(["git", "-C", str(repo), "add", "runs"], check=True)
        subprocess.run(["git", "-C", str(repo), "commit", "-qm", "add " + run_id],
                       check=True)
    return d


def make_session_record(repo: Path, session_id: str) -> None:
    root = repo / "runs" / "notebook-sessions"
    root.mkdir(parents=True, exist_ok=True)
    (root / f"{session_id}.json").write_text(
        json.dumps({"session_id": session_id, "notebook": "06", "mode": "GPU_CLASSROOM"}))


def make_artifact(root: Path, *, pin_xml=True, identity=None, **manifest_over):
    root.mkdir(parents=True, exist_ok=True)
    (root / "language_model_bf16.xml").write_bytes(b"<net>")
    (root / "language_model_bf16.bin").write_bytes(b"weights")
    m = {"source_sha256": "0" * 64,
         "output_bin_sha256": C.streaming_sha256(root / "language_model_bf16.bin"),
         "output_bin_bytes": 7,
         "storage": "FP16", "runtime_precision": "bf16"}
    if pin_xml:
        m["output_xml_sha256"] = C.streaming_sha256(root / "language_model_bf16.xml")
    m.update(manifest_over)
    (root / "conversion_manifest.json").write_text(json.dumps(m))
    return m


# ---------------------------------------------------------------------------
# P1 — categories, ids, path safety
# ---------------------------------------------------------------------------

def test_categories_are_exact_and_distinct():
    assert prov.PROVENANCE_CATEGORIES == (
        "CURRENT_SESSION_PHYSICAL", "SAME_STUDENT_PRIOR_PHYSICAL",
        "SHARED_CAMPAIGN_PHYSICAL", "COMMITTED_PHYSICAL_REFERENCE",
        "FROZEN_REFERENCE", "SIMULATED", "NOT_RUN", "BLOCKED")
    assert len(set(prov.PROVENANCE_CATEGORIES)) == 8
    assert len(prov.PROVENANCE_CATEGORIES) == len(set(prov.PROVENANCE_CATEGORIES))


def test_session_ids_are_unique_and_scoped():
    ids = {prov.new_session_id() for _ in range(200)}
    assert len(ids) == 200
    assert all(i.startswith("stu-") for i in ids)


def test_run_id_traversal_and_escapes_rejected(tmp_path):
    repo = make_repo(tmp_path)
    for bad in ("../evidence", "a/../b", "x/y", "", "/abs", ".hidden-ok", "sp ace"):
        with pytest.raises(prov.ProvenanceError):
            prov.run_dir(bad, repo)


def test_missing_run_is_not_run():
    info = prov.classify_run("20990101-000000-nbq-B0", repo=Path("/nonexistent-repo"))
    assert info["category"] == "NOT_RUN"


def test_committed_run_never_current_session(tmp_path):
    repo = make_repo(tmp_path)
    make_run(repo, "q1-075553-B0", commit=True)
    sid = prov.new_session_id()
    # even WITH a binding to this session, a tracked run stays COMMITTED
    prov.write_binding("q1-075553-B0", sid, kind="classroom-bounded", repo=repo)
    info = prov.classify_run("q1-075553-B0", session_id=sid, repo=repo)
    assert info["category"] == "COMMITTED_PHYSICAL_REFERENCE"


def test_current_session_physical_requires_binding_match(tmp_path):
    repo = make_repo(tmp_path)
    sid = prov.new_session_id()
    make_run(repo, "120000-nbq-m51pipe")
    prov.write_binding("120000-nbq-m51pipe", sid, kind="m51-pipeline:A,C", repo=repo)
    assert prov.classify_run("120000-nbq-m51pipe",
                             session_id=sid, repo=repo)["category"] == "CURRENT_SESSION_PHYSICAL"
    # a DIFFERENT session cannot claim it (stale/foreign session)
    other = prov.new_session_id()
    with pytest.raises(prov.ProvenanceError):
        prov.classify_run("120000-nbq-m51pipe", session_id=other, repo=repo)


def test_prior_session_physical_requires_session_record(tmp_path):
    repo = make_repo(tmp_path)
    prior = prov.new_session_id()
    make_run(repo, "130000-nbq-m51pipe")
    prov.write_binding("130000-nbq-m51pipe", prior, kind="m51-pipeline:A,C", repo=repo)
    # no session record yet -> fail closed
    with pytest.raises(prov.ProvenanceError):
        prov.classify_run("130000-nbq-m51pipe", repo=repo)
    make_session_record(repo, prior)
    info = prov.classify_run("130000-nbq-m51pipe", repo=repo)
    assert info["category"] == "SAME_STUDENT_PRIOR_PHYSICAL"


def test_unbound_untracked_run_is_never_promotable(tmp_path):
    repo = make_repo(tmp_path)
    make_run(repo, "140000-nbq-B0")  # no binding, not committed, no campaign
    with pytest.raises(prov.ProvenanceError):
        prov.classify_run("140000-nbq-B0", repo=repo)


def test_campaign_run_classifies_shared(tmp_path):
    repo = make_repo(tmp_path)
    camp = repo / "runs" / "notebook-gpu-qualification" / "camp-1"
    camp.mkdir(parents=True)
    (camp / "x.json").write_text("{}")
    info = prov.classify_run(str(camp.name), repo=None) if False else None
    # direct path form: classify by location under the campaign root
    real = prov.classify_run("notebook-gpu-qualification", repo=repo)
    assert real["category"] == "SHARED_CAMPAIGN_PHYSICAL"


def test_require_category_enforces_expectation(tmp_path):
    repo = make_repo(tmp_path)
    make_run(repo, "q1-075553-H3_OV_BF16", commit=True)
    with pytest.raises(prov.ProvenanceError):
        prov.require_category("q1-075553-H3_OV_BF16", "CURRENT_SESSION_PHYSICAL", repo=repo)
    ok = prov.require_category("q1-075553-H3_OV_BF16",
                               ("COMMITTED_PHYSICAL_REFERENCE",), repo=repo)
    assert ok["category"] == "COMMITTED_PHYSICAL_REFERENCE"
    with pytest.raises(prov.ProvenanceError):
        prov.require_category("q1-075553-H3_OV_BF16", "NOT_A_CATEGORY", repo=repo)


def test_frozen_vs_committed_paths(tmp_path):
    repo = make_repo(tmp_path)
    f = repo / "runs" / "m51-20261105-cleanroom-01" / "performance" / "encoder_ab.json"
    f.parent.mkdir(parents=True)
    f.write_text("{}")
    subprocess.run(["git", "-C", str(repo), "add", "runs"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "refs"], check=True)
    assert prov.classify_path("runs/m51-20261105-cleanroom-01/performance/"
                              "encoder_ab.json", repo)["category"] == \
        "COMMITTED_PHYSICAL_REFERENCE"
    assert prov.classify_path("evidence/results/SUMMARY.json",
                              repo)["category"] == "FROZEN_REFERENCE"
    assert prov.classify_path("runs/never-tracked.json", repo)["category"] == "UNVERIFIED"


def test_binding_is_append_only_across_sessions(tmp_path):
    repo = make_repo(tmp_path)
    make_run(repo, "150000-nbq-B0")
    prov.write_binding("150000-nbq-B0", "stu-a", kind="k", repo=repo)
    with pytest.raises(prov.ProvenanceError):
        prov.write_binding("150000-nbq-B0", "stu-b", kind="k", repo=repo)


# ---------------------------------------------------------------------------
# P2 — the M5.1 BF16 admission contract (fail closed)
# ---------------------------------------------------------------------------

def _admit(root, repo, **kw):
    return C.admit_ov_bf16(root, repo=repo, runtime_check=False, **kw)


def test_admission_pass(tmp_path, monkeypatch):
    repo = make_repo(tmp_path)
    root = tmp_path / "artifacts"
    make_artifact(root / "bf16")
    monkeypatch.setattr(C, "read_cgroup_mem", lambda *a, **k: {"limit_bytes": 60 * 2**30,
                                                               "anon_bytes": 0})
    monkeypatch.setattr(C, "host_mem_available_bytes", lambda *a, **k: 200 * 2**30)
    res = _admit(root, repo)
    assert res["status"] == "PASS", res
    assert res["remediation"] is None
    assert res["identity_manifest"] == {"storage": "FP16", "runtime_precision": "bf16"}


def test_admission_unpinned_xml_is_pass_with_warnings_never_pass(tmp_path, monkeypatch):
    repo = make_repo(tmp_path)
    root = tmp_path / "artifacts"
    make_artifact(root / "bf16", pin_xml=False)
    monkeypatch.setattr(C, "read_cgroup_mem", lambda *a, **k: {"limit_bytes": 60 * 2**30,
                                                               "anon_bytes": 0})
    monkeypatch.setattr(C, "host_mem_available_bytes", lambda *a, **k: 200 * 2**30)
    res = _admit(root, repo)
    assert res["status"] == "PASS_WITH_WARNINGS"
    assert any("pins no digest" in w for w in res["verification"]["warnings"])


def test_admission_missing_root_xml_bin_manifest(tmp_path):
    repo = make_repo(tmp_path)
    res = _admit(tmp_path / "nothing", repo)
    assert res["status"] == "BLOCKED" and "provision_ov.sh" in res["remediation"]

    root = tmp_path / "artifacts"
    make_artifact(root / "bf16")
    (root / "bf16" / "language_model_bf16.xml").unlink()  # missing XML
    res = _admit(root, repo)
    assert res["status"] == "BLOCKED" and res["verification"]["reason"] == "missing_artifacts"

    make_artifact(root / "bf16")
    (root / "bf16" / "language_model_bf16.bin").unlink()  # missing BIN
    res = _admit(root, repo)
    assert res["status"] == "BLOCKED" and res["verification"]["reason"] == "missing_artifacts"

    make_artifact(root / "bf16")
    (root / "bf16" / "conversion_manifest.json").unlink()  # missing manifest
    res = _admit(root, repo)
    assert res["status"] == "BLOCKED" and res["verification"]["reason"] == "missing_manifest"


def test_admission_corrupted_bin_is_fail_not_blocked(tmp_path):
    repo = make_repo(tmp_path)
    root = tmp_path / "artifacts"
    make_artifact(root / "bf16")
    (root / "bf16" / "language_model_bf16.bin").write_bytes(b"tamper!")  # same size
    res = _admit(root, repo)
    assert res["status"] == "FAIL", res
    assert res["verification"]["reason"] == "digest_mismatch"
    assert "do NOT continue" in res["remediation"]


def test_admission_truncated_bin_is_fail(tmp_path):
    repo = make_repo(tmp_path)
    root = tmp_path / "artifacts"
    make_artifact(root / "bf16")
    (root / "bf16" / "language_model_bf16.bin").write_bytes(b"wt")  # size_mismatch
    res = _admit(root, repo)
    assert res["status"] == "FAIL" and res["verification"]["reason"] == "size_mismatch"


def test_admission_precision_identity_mismatch_is_fail(tmp_path):
    repo = make_repo(tmp_path)
    root = tmp_path / "artifacts"
    make_artifact(root / "bf16", runtime_precision="int8")  # wrong artifact identity
    res = _admit(root, repo)
    assert res["status"] == "FAIL"
    assert res["verification"]["reason"] == "identity_mismatch"
    assert res["verification"]["mismatch"]["runtime_precision"]["expected"] == "bf16"


def test_admission_invalid_json_manifest_is_fail(tmp_path):
    repo = make_repo(tmp_path)
    root = tmp_path / "artifacts"
    make_artifact(root / "bf16")
    (root / "bf16" / "conversion_manifest.json").write_text("{broken")
    res = _admit(root, repo)
    assert res["status"] == "FAIL" and res["verification"]["reason"] == "corrupt_manifest"


def test_admission_inadequate_ram_is_blocked_with_remediation(tmp_path, monkeypatch):
    repo = make_repo(tmp_path)
    root = tmp_path / "artifacts"
    make_artifact(root / "bf16")
    monkeypatch.setattr(C, "read_cgroup_mem", lambda *a, **k: {"limit_bytes": 10 * 2**30,
                                                               "anon_bytes": 0})
    monkeypatch.setattr(C, "host_mem_available_bytes", lambda *a, **k: 12 * 2**30)
    res = _admit(root, repo)
    assert res["status"] == "BLOCKED"
    assert "insufficient RAM headroom" in res["remediation"]
    assert res["ram_gate"]["status"] == "BLOCKED"


def test_admission_missing_ov_runtime_is_blocked(tmp_path, monkeypatch):
    repo = make_repo(tmp_path)
    root = tmp_path / "artifacts"
    make_artifact(root / "bf16")
    monkeypatch.setattr(C, "read_cgroup_mem", lambda *a, **k: {"limit_bytes": 60 * 2**30,
                                                               "anon_bytes": 0})
    monkeypatch.setattr(C, "host_mem_available_bytes", lambda *a, **k: 200 * 2**30)
    res = C.admit_ov_bf16(root, repo=repo, runtime_check=True)
    assert res["status"] == "BLOCKED"
    assert "openvino==2026.4.1" in res["remediation"]


def test_admission_status_enums_are_strict(tmp_path):
    repo = make_repo(tmp_path)
    res = _admit(tmp_path / "nothing", repo)
    assert res["status"] in ("PASS", "PASS_WITH_WARNINGS", "FAIL", "BLOCKED")
    assert res["contract"] == "m51-ov-admission/v1"


# ---------------------------------------------------------------------------
# P5 — student result record validation (invalid promotions must fail)
# ---------------------------------------------------------------------------

def _base_record(**over):
    rec = {
        "experiment_id": "M51", "session_id": "stu-test",
        "run_ids": ["120000-nbq-m51pipe"],
        "provenance": "CURRENT_SESSION_PHYSICAL", "execution_status": "MEASURED",
        "gpu": {"arch": "gfx1100", "model": "W7900"},
        "source_head": "a" * 40,
        "workload": {"n_requests": 4, "seed": 42, "prompt_digest": "d" * 16},
        "populations": {"measured_count": 4},
        "measurements": {"a_batch_wall_s": 67.2, "c_batch_wall_s": 50.8},
        "throughput": {"numerator": 4, "denominator": 50.8, "unit": "images/s"},
        "speedup": 1.323,
        "memory": {"cpu_worker_rss_mb": 30000, "gpu_vram_peak_mb": 34000},
        "numerical_status": "PASS", "human_quality_status": "PENDING_HUMAN_REVIEW",
        "classification": "BOUNDED_PHYSICAL_COMPARISON",
        "ac_pair": {"a_run_id": "120000-nbq-m51pipe", "c_run_id": "120000-nbq-m51pipe",
                    "a_workload": {"seed": 42}, "c_workload": {"seed": 42}},
        "result_status": "PASS", "reproduce_command": "lab/m51_pipeline.py --arms A,C",
    }
    rec.update(over)
    return rec


def _current_session_repo(tmp_path):
    repo = make_repo(tmp_path)
    make_run(repo, "120000-nbq-m51pipe")
    prov.write_binding("120000-nbq-m51pipe", "stu-test", kind="m51-pipeline:A,C",
                       repo=repo)
    return repo


def test_valid_current_session_record_passes(tmp_path):
    repo = _current_session_repo(tmp_path)
    assert prov.validate_result_record(_base_record(), session_id="stu-test",
                                       repo=repo) == []


def test_reference_data_classified_as_current_fails(tmp_path):
    repo = make_repo(tmp_path)
    make_run(repo, "q1-075553-m51pipe", commit=True)  # instructor-owned, committed
    errs = prov.validate_result_record(
        _base_record(run_ids=["q1-075553-m51pipe"]), session_id="stu-test", repo=repo)
    assert any("COMMITTED_PHYSICAL_REFERENCE" in e or "not CURRENT_SESSION" in e
               for e in errs)


def test_instructor_run_as_student_owned_fails_without_run_ids():
    errs = prov.validate_result_record(
        _base_record(provenance="CURRENT_SESSION_PHYSICAL", run_ids=[]))
    assert any("requires run_ids" in e for e in errs)


def test_smoke_population_cannot_be_formal():
    errs = prov.validate_result_record(
        _base_record(classification="FORMAL_PROTOCOL_MEASURED",
                     populations={"measured_count": 2}))
    assert any("FORMAL" in e for e in errs)


def test_missing_source_head_for_new_claim_fails():
    errs = prov.validate_result_record(_base_record(source_head=None))
    assert any("source Git head" in e for e in errs)


def test_reused_record_claimed_measured_fails():
    errs = prov.validate_result_record(
        _base_record(provenance="COMMITTED_PHYSICAL_REFERENCE",
                     run_ids=["q1-075553-m51pipe"]))
    assert any("never MEASURED" in e for e in errs)


def test_ac_workload_mismatch_fails():
    pair = {"a_run_id": "r1", "c_run_id": "r2",
            "a_workload": {"seed": 42}, "c_workload": {"seed": 43}}
    errs = prov.validate_result_record(_base_record(ac_pair=pair))
    assert any("workload mismatch" in e for e in errs)


def test_duplicate_run_id_conflicting_metadata_rejected():
    prov.register_run_metadata("r-dup", {"head": "a" * 40})
    prov.register_run_metadata("r-dup", {"head": "a" * 40})  # identical ok
    with pytest.raises(prov.ProvenanceError):
        prov.register_run_metadata("r-dup", {"head": "b" * 40})


def test_invalid_enums_fail():
    assert prov.validate_result_record(_base_record(provenance="MEASURED_TODAY"))
    assert prov.validate_result_record(_base_record(execution_status="GREEN"))
    assert prov.validate_result_record(_base_record(classification="ULTRA"))


def test_stale_session_cannot_claim_current(tmp_path):
    repo = make_repo(tmp_path)
    make_run(repo, "120000-nbq-m51pipe")
    prov.write_binding("120000-nbq-m51pipe", "stu-OTHER-session", kind="k", repo=repo)
    errs = prov.validate_result_record(_base_record(), session_id="stu-test", repo=repo)
    assert any("120000-nbq-m51pipe" in e for e in errs)  # fail closed, no silent claim


def test_numerical_fail_never_human_pass():
    errs = prov.validate_result_record(
        _base_record(numerical_status="FAIL", human_quality_status="PASS"))
    assert any("numerical FAIL" in e for e in errs)


def test_m6_pending_human_review_cannot_release_pass():
    errs = prov.validate_result_record(
        _base_record(experiment_id="M6", result_status="PASS",
                     human_quality_status="PENDING_HUMAN_REVIEW"))
    assert any("PENDING_HUMAN_REVIEW" in e for e in errs)


def test_reference_only_cannot_carry_speedup():
    errs = prov.validate_result_record(
        _base_record(provenance="COMMITTED_PHYSICAL_REFERENCE",
                     execution_status="REUSED", classification="REFERENCE_ONLY",
                     ac_pair=None))
    assert any("must not carry a speedup" in e for e in errs)


def test_throughput_requires_numerator_denominator_unit():
    errs = prov.validate_result_record(
        _base_record(throughput={"numerator": 4}))
    assert any("throughput missing" in e for e in errs)


# ---------------------------------------------------------------------------
# notebook_runtime integration (fail-closed consume, unique tags, bindings)
# ---------------------------------------------------------------------------

def test_session_consume_classifies_and_labels(tmp_path, monkeypatch, capsys):
    import lab.lab_env as lab_env
    from lab import notebook_runtime as rt
    repo = make_repo(tmp_path)
    monkeypatch.setattr(prov, "REPO", repo)
    monkeypatch.setattr(lab_env, "REPO", repo)
    monkeypatch.setattr(lab_env, "STUDENT_RUNS_ROOT", repo / "runs")
    make_run(repo, "q1-075553-H3_OV_BF16",
             records={"experiment": "H3_OV_BF16", "runs": [
                 {"kind": "warm", "total_s": 1.0}], "run_kind": "measure",
                 "prompt_id": "p1"}, commit=True)
    s = rt.Session("GPU_CLASSROOM", "06")
    bundle = s.consume("q1-075553-H3_OV_BF16", why="explicit reference")
    out = capsys.readouterr().out
    assert "[CONSUMED PHYSICAL RECORD] run_id=q1-075553-H3_OV_BF16" in out
    assert "[PROVENANCE] COMMITTED_PHYSICAL_REFERENCE | run_id=q1-075553-H3_OV_BF16" in out
    assert s.consumed[-1]["category"] == "COMMITTED_PHYSICAL_REFERENCE"
    assert bundle.run_id == "q1-075553-H3_OV_BF16"


def test_session_consume_unverified_run_fails_closed(tmp_path, monkeypatch):
    import lab.lab_env as lab_env
    from lab import notebook_runtime as rt
    repo = make_repo(tmp_path)
    monkeypatch.setattr(prov, "REPO", repo)
    monkeypatch.setattr(lab_env, "REPO", repo)
    monkeypatch.setattr(lab_env, "STUDENT_RUNS_ROOT", repo / "runs")
    make_run(repo, "160000-nbq-B0")  # unbound, untracked, non-campaign
    s = rt.Session("GPU_CLASSROOM", "01")
    with pytest.raises(prov.ProvenanceError):
        s.consume("160000-nbq-B0", why="mystery run")


def test_fresh_tag_deconflicts_existing_runs(tmp_path, monkeypatch):
    import lab.lab_env as lab_env
    from lab import notebook_runtime as rt
    repo = make_repo(tmp_path)
    monkeypatch.setattr(lab_env, "STUDENT_RUNS_ROOT", repo / "runs")
    make_run(repo, "000000-nbq-B0")  # collide with a forced 000000 timestamp
    s = rt.Session.__new__(rt.Session)  # bypass __init__ (no admission side effects)
    s.mode, s.notebook = "GPU_CLASSROOM", "01"
    import datetime as _dt
    class FixedDt(_dt.datetime):
        @classmethod
        def now(cls, *a, **k):
            return cls(2026, 11, 6, 0, 0, 0)
    monkeypatch.setattr(rt, "datetime", FixedDt)
    tag = s._fresh_tag("B0")
    assert tag != "000000-nbq"
    assert not (repo / "runs" / f"{tag}-B0").exists()


def test_session_record_carries_identity(tmp_path, monkeypatch):
    import lab.lab_env as lab_env
    from lab import notebook_runtime as rt
    repo = make_repo(tmp_path)
    monkeypatch.setattr(prov, "REPO", repo)
    monkeypatch.setattr(lab_env, "REPO", repo)
    monkeypatch.setattr(lab_env, "STUDENT_RUNS_ROOT", repo / "runs")
    monkeypatch.setenv("QIR_NOTEBOOK_SESSIONS_ROOT", "")
    s = rt.Session("REFERENCE", "05")
    assert s.session_id.startswith("stu-")
    p = s.finish({"mode": "REFERENCE"})
    d = json.loads(p.read_text())
    assert d["session_id"] == s.session_id
    assert "source" in d and "head" in d["source"]
    make_session_record(repo, s.session_id)
    assert prov.session_record_exists(s.session_id, repo)


def test_workload_identity_distinguishes_prompts():
    a = {"prompts": [{"id": "r1", "text": "x", "seed": 42}]}
    b = {"prompts": [{"id": "r1", "text": "x", "seed": 42}]}
    c = {"prompts": [{"id": "r1", "text": "OTHER", "seed": 42}]}
    d = {"prompts": [{"id": "r1", "text": "x", "seed": 43}]}
    assert prov.workload_identity(a) == prov.workload_identity(b)
    assert prov.workload_identity(a) != prov.workload_identity(c)
    assert prov.workload_identity(a) != prov.workload_identity(d)


def test_binding_failure_propagates_instead_of_stamping_measured(tmp_path, monkeypatch, capsys):
    """A successful GPU CLI cannot be labelled CURRENT_SESSION_PHYSICAL if
    its PROVENANCE.json binding fails. No GPU is used in this regression.
    """
    import lab.lab_env as lab_env
    from lab import notebook_runtime as rt
    repo = make_repo(tmp_path)
    monkeypatch.setattr(prov, "REPO", repo)
    monkeypatch.setattr(lab_env, "REPO", repo)
    monkeypatch.setattr(lab_env, "STUDENT_RUNS_ROOT", repo / "runs")
    s = rt.Session("GPU_CLASSROOM", "06")
    s.admission = rt.Admission("PASS", "GPU_CLASSROOM", {}, [])

    def refuse_binding(*args, **kwargs):
        raise prov.ProvenanceError("simulated binding conflict")

    monkeypatch.setattr(prov, "write_binding", refuse_binding)
    monkeypatch.setattr(rt.subprocess, "run", lambda *args, **kwargs:
                        subprocess.CompletedProcess(args[0], 0, "synthetic success", ""))
    with pytest.raises(rt.AdmissionError, match="provenance binding FAILED"):
        s._run_cli(["synthetic-gpu-cli"], timeout=1.0, kind="m51-pipeline",
                   lock=False, bind="student-m51")
    assert s.executions == []
    assert "CURRENT_SESSION_PHYSICAL" not in capsys.readouterr().out


def test_binding_failure_does_not_mutate_executions(tmp_path, monkeypatch):
    import lab.lab_env as lab_env
    from lab import notebook_runtime as rt
    repo = make_repo(tmp_path)
    monkeypatch.setattr(prov, "REPO", repo)
    monkeypatch.setattr(lab_env, "REPO", repo)
    monkeypatch.setattr(lab_env, "STUDENT_RUNS_ROOT", repo / "runs")
    s = rt.Session("GPU_CLASSROOM", "06")
    monkeypatch.setattr(prov, "write_binding", lambda *a, **k: (_ for _ in ()).throw(
        prov.ProvenanceError("a prior session already owns this run")))
    with pytest.raises(rt.AdmissionError, match="prior session"):
        s._bind("student-m51", kind="m51-pipeline")
    assert not s.executions
