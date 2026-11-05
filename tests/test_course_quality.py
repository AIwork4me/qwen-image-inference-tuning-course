"""Course-quality gate tests — including NEGATIVE tests proving each gate
detects actual bad input (a gate that cannot fail protects nothing)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "course-quality"))

import check_bilingual  # noqa: E402
import check_claims  # noqa: E402
import check_examples  # noqa: E402
import check_links  # noqa: E402
import check_metric_pairs  # noqa: E402


# --- the real repository must pass all four gates --------------------------

def test_real_repo_passes_all_gates():
    assert check_links.run(REPO)[0] == []
    assert check_bilingual.run(REPO) == []
    assert check_claims.run(REPO) == []
    assert check_examples.run(REPO) == []
    assert check_metric_pairs.run(REPO) == []


# --- check_links: negative --------------------------------------------------

def test_check_links_detects_broken_markdown_link(tmp_path):
    (tmp_path / "doc.md").write_text("see [guide](missing/GUIDE.md)\n")
    problems, _ = check_links.run.__wrapped__(tmp_path) if hasattr(check_links.run, "__wrapped__") else (None, None)
    # run() relies on git ls-files; test the per-file function directly
    problems = check_links.check_file(tmp_path / "doc.md", tmp_path)
    assert any("broken link -> missing/GUIDE.md" in p for p in problems)


def test_check_links_accepts_anchor_and_http(tmp_path):
    (tmp_path / "doc.md").write_text("x [a](#sec) y [b](https://e.invalid/x) <img src=\"img.png\">\n")
    (tmp_path / "img.png").write_bytes(b"png")
    assert check_links.check_file(tmp_path / "doc.md", tmp_path) == []


def test_check_links_detects_missing_bare_path(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text("output goes to `docs/missing_dir/FILE.md`\n")
    problems = check_links.check_file(docs / "a.md", tmp_path)
    assert any("docs/missing_dir/FILE.md" in p for p in problems)


def test_check_links_skips_host_only_and_future(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text("host-only `runs/raw/x-B0/a.png`; future "
                               "`evidence/quality/VERDICTS.md`\n")
    assert check_links.check_file(docs / "a.md", tmp_path) == []


# --- check_bilingual: negative ----------------------------------------------

def test_bilingual_detects_missing_counterpart(tmp_path):
    (tmp_path / "README.md").write_text("# hi\n")
    problems = check_bilingual.check_pair(tmp_path, "README.md", "README.zh-CN.md")
    assert any("missing Chinese counterpart" in p for p in problems)


def test_bilingual_detects_lost_command(tmp_path):
    (tmp_path / "README.md").write_text(
        "# t\n```bash\n.venv-lab/bin/python -m lab run --experiment BEST --smoke\n```\n")
    (tmp_path / "README.zh-CN.md").write_text(
        "# 标题\n```bash\n# 中文注释行，命令丢失\ntrue\n```\n" + "段落\n" * 20)
    problems = check_bilingual.check_pair(tmp_path, "README.md", "README.zh-CN.md")
    assert any("documented command(s) absent" in p for p in problems)


def test_bilingual_detects_summary_translation(tmp_path):
    en = "# t\n" + "\n".join(f"## s{i}\ntext\n" for i in range(10))
    zh = "# 标\n一段话"
    (tmp_path / "A.md").write_text(en)
    (tmp_path / "A.zh-CN.md").write_text(zh)
    problems = check_bilingual.check_pair(tmp_path, "A.md", "A.zh-CN.md")
    assert any("structure shrank" in p for p in problems)


def test_bilingual_command_parity_ignores_language_specific_comments(tmp_path):
    (tmp_path / "A.md").write_text("```bash\n.venv-lab/bin/python -m lab doctor  # check\n```\n")
    (tmp_path / "A.zh-CN.md").write_text("```bash\n.venv-lab/bin/python -m lab doctor  # 检查\n```\n")
    problems = check_bilingual.check_pair(tmp_path, "A.md", "A.zh-CN.md")
    assert not any("command(s) absent" in p for p in problems)


# --- check_claims: negative --------------------------------------------------

def test_claims_rejects_invalid_registry(tmp_path):
    (tmp_path / "course").mkdir()
    (tmp_path / "course/data").mkdir()
    (tmp_path / "course/data/verified_claims.json").write_text(
        json.dumps({"claims": [{"id": "X", "status": "guessed"}]}))
    problems = check_claims.validate_registry(tmp_path)
    assert any("missing field" in p or "not in" in p for p in problems)


def test_claims_rejects_int8_fail_promoted_to_pass(tmp_path):
    (tmp_path / "course").mkdir(); (tmp_path / "course/data").mkdir()
    (tmp_path / "course/data/verified_claims.json").write_text(json.dumps({"claims": [
        {"id": "M51-INT8-NEGATIVE-FAIL", "status": "measured",
         "evidence_path": "x", "quality_status": "PASS"}]}))
    problems = check_claims.validate_registry(tmp_path)
    assert problems, "the gate must refuse to register the INT8 FAIL as PASS"


def test_claims_detects_bilingual_numeric_drift(tmp_path):
    for rel, body in (("README.md", "6.12x faster"),
                      ("README.zh-CN.md", "快 6.1 倍（约）")):
        (tmp_path / rel).write_text(body)
    problems = check_claims.check_parity(tmp_path)
    assert any("README.zh-CN.md" in p and "6.12" in p for p in problems)


# --- check_examples: negative ------------------------------------------------

def test_examples_detects_unknown_flag(tmp_path):
    lab = tmp_path / "lab"; lab.mkdir()
    (lab / "cli.py").write_text("import argparse\nap.add_argument('--real')\n")
    doc = tmp_path / "README.md"
    doc.write_text("```bash\n.venv-lab/bin/python -m lab run --experiment B0 --fabricated\n```\n")
    problems = check_examples.run(tmp_path, docs=["README.md"])
    assert any("--fabricated" in p for p in problems)


def test_examples_detects_bash_syntax_error(tmp_path):
    lab = tmp_path / "lab"; lab.mkdir(); (lab / "cli.py").write_text("")
    (tmp_path / "README.md").write_text("```bash\nif [ -f x\nthen\necho hi\n```\n")
    problems = check_examples.run(tmp_path, docs=["README.md"])
    assert any("bash syntax" in p for p in problems)


def test_examples_accepts_documented_placeholders(tmp_path):
    lab = tmp_path / "lab"; lab.mkdir()
    (lab / "cli.py").write_text("import argparse\nap.add_argument('--experiment')\nap.add_argument('--date')\n")
    (tmp_path / "README.md").write_text(
        "```bash\n.venv-lab/bin/python -m lab run --experiment B0 --date <your-tag>\n```\n")
    assert check_examples.run(tmp_path, docs=["README.md"]) == []


# --- P2 study helper ----------------------------------------------------------

def test_attribute_request_derives_phases_and_tolerates_missing():
    from lab.m51_pipeline_profile import attribute_request
    rec = {"t_pre_post": 0.0, "t_post": 0.05, "t_first_exec": 0.4, "t_ws_done": 12.4,
           "t_hist": 12.9, "t_fetched": 12.95, "t_written": 12.97, "t_wall_end": 12.97,
           "history_polls": 13, "mode": "poll"}
    a = attribute_request(rec)
    assert a["post_http_s"] == 0.05
    assert a["poll_wait_after_done_s"] == 0.5
    assert a["wall_s"] == 12.97
    # missing fields must yield None, never invented numbers
    b = attribute_request({"mode": "ws"})
    assert b["post_http_s"] is None and b["wall_s"] is None


def test_check_links_uses_git_tracked_fallback(tmp_path):
    # a bare path that is gitignored (host-only) must not fail when it is
    # absent, and must still fail when it is neither present nor tracked.
    import subprocess
    docs = tmp_path / "docs"; docs.mkdir()
    (docs / "a.md").write_text("see `runs/m51-ov-artifacts` and `runs/ghost-dir`\n")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    tracked = check_links.git_tracked_paths(tmp_path)
    problems = check_links.check_file(docs / "a.md", tmp_path, tracked)
    assert any("runs/ghost-dir" in p for p in problems)
    assert not any("m51-ov-artifacts" in p for p in problems)


# --- P3 strict status enums: negative battery --------------------------------
# Every test proves the gate FAILS on real tamper/mistake patterns (mission
# publication-closure Phase P3). Fixtures build a minimal VALID registry and
# mutate exactly one aspect.

OWNER_RECORD = {"reviewer": "owner-fixture", "verdict": "PASS",
                "date": "2026-11-05", "scope": "M5.1 C1b deployment vs G_native"}


def _fixture_registry(tmp_path, mutate=None):
    """Minimal valid registry + owner-record fixture; returns problems."""
    (tmp_path / "course/data").mkdir(parents=True, exist_ok=True)
    (tmp_path / "evidence").mkdir(exist_ok=True)
    (tmp_path / "evidence/ev.json").write_text("{}")
    (tmp_path / "docs/release-closure").mkdir(parents=True, exist_ok=True)
    (tmp_path / "docs/release-closure/PR2_OWNER_QUALITY_VERDICT.json").write_text(
        json.dumps(OWNER_RECORD))
    claims = [
        {"id": "OK-1", "status": "measured", "evidence_path": "evidence/ev.json",
         "measurement_status": "MEASURED", "numerical_verdict": "PASS",
         "human_quality_verdict": "NOT_APPLICABLE", "release_decision": "NOT_APPLICABLE"},
        {"id": "M51-INT8-NEGATIVE-FAIL", "status": "measured",
         "evidence_path": "evidence/ev.json",
         "measurement_status": "MEASURED", "numerical_verdict": "FAIL",
         "human_quality_verdict": "NOT_RUN", "release_decision": "RELEASE_FAIL"},
        {"id": "M51-OWNER-QUALITY-VERDICT", "status": "measured",
         "evidence_path": "evidence/ev.json",
         "measurement_status": "MEASURED", "numerical_verdict": "NOT_APPLICABLE",
         "human_quality_verdict": "OWNER_ACCEPTED_SCOPE_ONLY",
         "verdict_scope": "M5.1 C1b deployment vs G_native",
         "release_decision": "RELEASE_PENDING"},
    ]
    if mutate:
        claims, owner = mutate(claims)
        if owner is not None:
            (tmp_path / "docs/release-closure/PR2_OWNER_QUALITY_VERDICT.json").write_text(
                json.dumps(owner))
    (tmp_path / "course/data/verified_claims.json").write_text(
        json.dumps({"claims": claims}))
    return check_claims.validate_registry(tmp_path)


def test_claims_rejects_not_run_promoted_to_pass(tmp_path):
    def m(claims):
        claims.append({"id": "SWEEP-40STEP-CFG", "status": "NOT_RUN",
                       "evidence_path": "evidence/ev.json",
                       "measurement_status": "NOT_RUN", "numerical_verdict": "PASS",
                       "human_quality_verdict": "PASS", "release_decision": "RELEASE_PASS"})
        return claims, OWNER_RECORD
    problems = _fixture_registry(tmp_path, m)
    assert any("NOT_RUN but numerical_verdict='PASS'" in p for p in problems)
    assert any("NOT_RUN but human_quality_verdict='PASS'" in p for p in problems)
    assert any("NOT_RUN but release_decision='RELEASE_PASS'" in p for p in problems)


def test_claims_rejects_missing_quality_verdict_field(tmp_path):
    def m(claims):
        claims.append({"id": "NO-VERDICT", "status": "measured",
                       "evidence_path": "evidence/ev.json",
                       "measurement_status": "MEASURED"})
        return claims, OWNER_RECORD
    problems = _fixture_registry(tmp_path, m)
    assert any("missing canonical field 'numerical_verdict'" in p for p in problems)
    assert any("missing canonical field 'human_quality_verdict'" in p for p in problems)


def test_claims_rejects_empty_and_unknown_status(tmp_path):
    def m(claims):
        claims[0]["numerical_verdict"] = ""
        claims[0]["human_quality_verdict"] = "LOOKS_FINE_TO_ME"
        return claims, OWNER_RECORD
    problems = _fixture_registry(tmp_path, m)
    assert any("numerical_verdict ''" in p or "numerical_verdict" in p
               and "not an exact enum" in p for p in problems)
    assert any("'LOOKS_FINE_TO_ME'" in p for p in problems)


def test_claims_rejects_mixed_status_strings(tmp_path):
    # "FAIL/PASS" in a legacy field must be rejected (no substring acceptance)
    def m(claims):
        claims.append({"id": "MIXED-1", "status": "measured",
                       "evidence_path": "evidence/ev.json",
                       "quality_status": "FAIL/PASS"})
        return claims, OWNER_RECORD
    problems = _fixture_registry(tmp_path, m)
    assert any("quality_status rejected" in p for p in problems)


def test_claims_rejects_sentence_containing_fail_and_pass(tmp_path):
    def m(claims):
        claims.append({"id": "MIXED-2", "status": "measured",
                       "evidence_path": "evidence/ev.json",
                       "quality_status": "PASS (was FAIL, promoted by owner)"})
        return claims, OWNER_RECORD
    problems = _fixture_registry(tmp_path, m)
    assert any("quality_status rejected" in p and "conflicting" in p for p in problems)


def test_claims_accepts_honest_legacy_prose(tmp_path):
    def ok(claims):
        claims.append({"id": "LEGACY-OK", "status": "measured",
                       "evidence_path": "evidence/ev.json",
                       "quality_status": "PASS (numerics unchanged - kernel swap)"})
        return claims, OWNER_RECORD
    assert _fixture_registry(tmp_path, ok) == []


def test_claims_rejects_duplicate_claim_ids(tmp_path):
    def m(claims):
        claims.append(dict(claims[0]))
        return claims, OWNER_RECORD
    problems = _fixture_registry(tmp_path, m)
    assert any("duplicate claim id OK-1" in p for p in problems)


def test_claims_rejects_owner_scope_mismatch(tmp_path):
    def m(claims):
        for c in claims:
            if c["id"] == "M51-OWNER-QUALITY-VERDICT":
                c["verdict_scope"] = "everything, all experiments, forever"
        return claims, OWNER_RECORD
    problems = _fixture_registry(tmp_path, m)
    assert any("verdict_scope must equal" in p for p in problems)


def test_claims_rejects_owner_verdict_record_scope_drift(tmp_path):
    drifted = dict(OWNER_RECORD, scope="M5.1 C1b + INT8 (widened)")
    def m(claims):
        return claims, drifted
    problems = _fixture_registry(tmp_path, m)
    assert any("owner record" in p and "scope mismatch" in p for p in problems)


def test_claims_rejects_m51_human_verdict_applied_to_int8(tmp_path):
    def m(claims):
        for c in claims:
            if c["id"] == "M51-INT8-NEGATIVE-FAIL":
                c["human_quality_verdict"] = "OWNER_ACCEPTED_SCOPE_ONLY"
        return claims, OWNER_RECORD
    problems = _fixture_registry(tmp_path, m)
    assert any("scope leak" in p for p in problems)
    assert any("owner acceptance explicitly does NOT cover" in p for p in problems)


def test_claims_rejects_m51_human_verdict_applied_to_m6_best(tmp_path):
    def m(claims):
        claims.append({"id": "M6-BEST-QUALITY", "status": "measured",
                       "evidence_path": "evidence/ev.json",
                       "measurement_status": "MEASURED",
                       "numerical_verdict": "NOT_APPLICABLE",
                       "human_quality_verdict": "OWNER_ACCEPTED_SCOPE_ONLY",
                       "release_decision": "RELEASE_PASS"})
        return claims, OWNER_RECORD
    problems = _fixture_registry(tmp_path, m)
    assert any("scope leak" in p for p in problems)
    assert any("M6-BEST-QUALITY" in p and "does NOT cover" in p for p in problems)


def test_claims_rejects_missing_source_evidence(tmp_path):
    def m(claims):
        claims[0]["evidence_path"] = "evidence/not_committed_anywhere.json"
        return claims, OWNER_RECORD
    problems = _fixture_registry(tmp_path, m)
    assert any("evidence_path does not resolve" in p for p in problems)


def test_claims_rejects_derived_claim_without_derivation(tmp_path):
    def m(claims):
        claims.append({"id": "DERIVED-BAD", "status": "derived",
                       "evidence_path": "evidence/ev.json",
                       "measurement_status": "DERIVED",
                       "numerical_verdict": "PASS",
                       "human_quality_verdict": "NOT_APPLICABLE",
                       "release_decision": "NOT_APPLICABLE"})
        return claims, OWNER_RECORD
    problems = _fixture_registry(tmp_path, m)
    assert any("requires derivation" in p for p in problems)


def test_claims_rejects_derived_claim_with_unknown_refs(tmp_path):
    def m(claims):
        claims.append({"id": "DERIVED-REF", "status": "derived",
                       "evidence_path": "evidence/ev.json",
                       "measurement_status": "DERIVED",
                       "derivation": {"numerator_ref": "GHOST-A", "denominator_ref": "GHOST-B"},
                       "numerical_verdict": "PASS",
                       "human_quality_verdict": "NOT_APPLICABLE",
                       "release_decision": "NOT_APPLICABLE"})
        return claims, OWNER_RECORD
    problems = _fixture_registry(tmp_path, m)
    assert any("references unknown claims" in p for p in problems)


def test_claims_valid_fixture_passes(tmp_path):
    assert _fixture_registry(tmp_path) == []


# --- P1 metric-pair population gate: negative battery ------------------------

def _pair_repo(tmp_path, readme_body):
    prov = json.loads((REPO / "docs/publication-closure/P1_METRIC_PROVENANCE.json")
                      .read_text(encoding="utf-8"))
    d = tmp_path / "docs/publication-closure"; d.mkdir(parents=True)
    (d / "P1_METRIC_PROVENANCE.json").write_text(json.dumps(prov))
    (tmp_path / "README.md").write_text(readme_body)
    return check_metric_pairs.check_pairs(tmp_path)


def test_metric_pairs_reject_h2_time_with_rebaseline_throughput(tmp_path):
    problems = _pair_repo(tmp_path, 'A4["4 reqs: 737 s<br/>0.35 img/min"]')
    assert any("MIXED-POPULATION pair (737.0 s, 0.35)" in p for p in problems)


def test_metric_pairs_reject_cold_time_with_warm_throughput(tmp_path):
    problems = _pair_repo(tmp_path, 'B4["4 reqs: 50 s<br/>4.55 img/min"]')
    assert any("MIXED-POPULATION pair (50.0 s, 4.55)" in p for p in problems)


def test_metric_pairs_reject_arithmetic_but_unregistered_pair(tmp_path):
    # 100 s -> 2.4 img/min is arithmetically consistent but matches no
    # registered population: fabricated provenance must still fail
    problems = _pair_repo(tmp_path, 'C1["4 reqs: 100 s<br/>2.4 img/min"]')
    assert any("matches NO registered population" in p for p in problems)


def test_metric_pairs_accept_same_population_pairs(tmp_path):
    ok = ('A4["4 reqs: 737 s<br/>0.33 img/min"] '
          'B4["4 reqs: 50.05 s cold-incl.<br/>4.80 img/min"] '
          'C1["warm: 52.7 s / 4.55 img/min"] '
          'D1["rebaseline B: 680.0 s / 0.35 img/min"]')
    assert _pair_repo(tmp_path, ok) == []


# --- P4 experiment module: CPU-safety and scope regression -------------------

def test_pipeline_profile_helpers_bind_lazy_imports():
    # Regression for the 1d797ad lazy-import refactor that left batch_generic
    # / batch_fast / main referencing unbound PROMPTS/SEED/E: parse the module
    # and require every function that uses those names to bind them locally.
    import ast
    src = (REPO / "lab/m51_pipeline_profile.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    watched = {"PROMPTS", "SEED", "E"}
    offenders = []
    for fn in tree.body:
        if not isinstance(fn, ast.FunctionDef):
            continue
        used = {n.id for n in ast.walk(fn)
                if isinstance(n, ast.Name) and n.id in watched}
        if not used:
            continue
        bound = set()
        for n in ast.walk(fn):
            if isinstance(n, ast.ImportFrom):
                bound |= {a.asname or a.name for a in n.names}
        missing = used - bound
        if missing:
            offenders.append((fn.name, sorted(missing)))
    assert offenders == [], f"unbound lazy-import names: {offenders}"


def test_fair_ab_module_is_cpu_importable_and_preregistered():
    import importlib
    mod = importlib.import_module("lab.m51_pipeline_fair_ab")
    assert mod.SEED == 42
    src = (REPO / "lab/m51_pipeline_fair_ab.py").read_text(encoding="utf-8")
    assert "exclusion_rule" in src and "never performance-based" in src
