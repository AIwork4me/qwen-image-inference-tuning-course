#!/usr/bin/env python3
"""Course-quality gate 3: scientific-claim status semantics (CPU-only).

Publication-closure Phase P3 replaced fragile substring verdict validation
("PASS" somewhere inside free prose) with EXACT canonical status enums and
fail-closed cross-field semantics.

Canonical per-claim contract (course/data/verified_claims.json):

  measurement_status     MEASURED | DERIVED | ESTIMATED | HISTORICAL | NOT_RUN
  numerical_verdict      PASS | FAIL | NOT_APPLICABLE | NOT_RUN
  human_quality_verdict  PASS | FAIL | PENDING_HUMAN_REVIEW |
                         OWNER_ACCEPTED_SCOPE_ONLY | NOT_APPLICABLE | NOT_RUN
  release_decision       RELEASE_PASS | RELEASE_FAIL | RELEASE_PENDING |
                         BLOCKED | NOT_APPLICABLE | NOT_RUN
  explanation            free human prose — NEVER parsed for verdicts
  evidence_path          must resolve to a repo path (first token)
  derivation             REQUIRED for measurement_status=DERIVED:
                         {numerator_ref, denominator_ref} referencing claim ids

Backward compatibility: a legacy entry carrying only `quality_status` is
accepted IFF the string is EXACTLY an enum value, or starts with an enum value
followed by " (" AND the parenthetical contains no other status token from a
different family. This rejects "FAIL/PASS", "PASS (was FAIL)", empty, unknown
and sentence-statuses while accepting the historical honest prose form.

Immutable verdicts (fail-closed; tampering is a hard error):
  M51-INT8-NEGATIVE-FAIL       numerical_verdict FAIL     (never PASS)
  M51-C1B-UNCOMPRESSED-FAIL    numerical_verdict FAIL     (never PASS)
  SWEEP-40STEP-CFG             measurement_status NOT_RUN (never any PASS)

Human-verdict scope containment (the owner acceptance may never leak):
  the ONLY claim that may carry human_quality_verdict =
  OWNER_ACCEPTED_SCOPE_ONLY is M51-OWNER-QUALITY-VERDICT, its verdict_scope
  must exactly match the committed record
  (docs/release-closure/PR2_OWNER_QUALITY_VERDICT.json:scope), and the claims
  for H3_OV_INT8 / the uncompressed variant / M6 BEST / the 40-step sweep can
  never reference the owner acceptance.

Bilingual parity: the critical values taught in both languages (PARITY_VALUES)
must appear in both carriers. This is a presence check, deliberately NOT a
global numeric diff; arithmetic/population consistency is enforced by
check_metric_pairs.py (gate 5).

Usage:  python scripts/course-quality/check_claims.py [--root REPO]
Exit:   0 = registry valid + parity holds, 1 otherwise.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REQUIRED_FIELDS = ("id", "evidence_path")
STATUS_VOCAB = ("measured", "derived", "estimated", "historical", "NOT_RUN")

MEASUREMENT_ENUM = ("MEASURED", "DERIVED", "ESTIMATED", "HISTORICAL", "NOT_RUN")
NUMERICAL_ENUM = ("PASS", "FAIL", "NOT_APPLICABLE", "NOT_RUN")
HUMAN_ENUM = ("PASS", "FAIL", "PENDING_HUMAN_REVIEW", "OWNER_ACCEPTED_SCOPE_ONLY",
              "NOT_APPLICABLE", "NOT_RUN")
RELEASE_ENUM = ("RELEASE_PASS", "RELEASE_FAIL", "RELEASE_PENDING", "BLOCKED",
                "NOT_APPLICABLE", "NOT_RUN")

# A legacy quality_status may lead with one enum token; the remainder must be
# parenthetical prose WITHOUT any other cross-family status token.
_ALL_STATUS_TOKENS = set(MEASUREMENT_ENUM) | set(NUMERICAL_ENUM) | set(HUMAN_ENUM)

IMMUTABLE = {
    "M51-INT8-NEGATIVE-FAIL": {"field": "numerical_verdict", "value": "FAIL"},
    "M51-C1B-UNCOMPRESSED-FAIL": {"field": "numerical_verdict", "value": "FAIL"},
    "SWEEP-40STEP-CFG": {"field": "measurement_status", "value": "NOT_RUN"},
}
OWNER_VERDICT_CLAIM = "M51-OWNER-QUALITY-VERDICT"
OWNER_VERDICT_RECORD = "docs/release-closure/PR2_OWNER_QUALITY_VERDICT.json"
OWNER_VERDICT_SCOPE = "M5.1 C1b deployment vs G_native"
# Claims that must NEVER inherit the owner acceptance:
OWNER_VERDICT_FORBIDDEN = ("M51-INT8-NEGATIVE-FAIL", "M51-C1B-UNCOMPRESSED-FAIL",
                           "M6-BEST-QUALITY", "SWEEP-40STEP-CFG")
EVIDENCE_HOST_ONLY_PREFIXES = ("runs/pr2-p4-pipewarm-120822/pipeline_warm/raw/",)

# value -> (EN carrier, ZH carrier) — the parity the mission demands
PARITY_VALUES = [
    ("55.21", "README.md", "README.zh-CN.md"),
    ("9.02", "README.md", "README.zh-CN.md"),
    ("6.12", "README.md", "README.zh-CN.md"),
    ("1.081", "README.md", "README.zh-CN.md"),
    ("1.029", "README.md", "README.zh-CN.md"),
    ("13.031", "README.md", "README.zh-CN.md"),
    ("12.060", "README.md", "README.zh-CN.md"),
    ("4.42", "README.md", "README.zh-CN.md"),
    ("4.55", "README.md", "README.zh-CN.md"),
    ("331", "README.md", "README.zh-CN.md"),
    ("480", "README.md", "README.zh-CN.md"),
    ("28.25", "README.md", "README.zh-CN.md"),
    ("1.33", "README.md", "README.zh-CN.md"),
    ("16.18", "course/M05_parallel_heterogeneous/M5_1_OPENVINO_CPU_TE.md",
     "course/M05_parallel_heterogeneous/M5_1_OPENVINO_CPU_TE.zh-CN.md"),
    ("17.26", "README.md", "README.zh-CN.md"),
    ("1.081", "course/M05_parallel_heterogeneous/M5_1_OPENVINO_CPU_TE.md",
     "course/M05_parallel_heterogeneous/M5_1_OPENVINO_CPU_TE.zh-CN.md"),
    ("1.029", "course/M05_parallel_heterogeneous/M5_1_OPENVINO_CPU_TE.md",
     "course/M05_parallel_heterogeneous/M5_1_OPENVINO_CPU_TE.zh-CN.md"),
    ("331", "course/METHODOLOGY.md", "course/METHODOLOGY.zh-CN.md"),
    ("4.55", "course/METHODOLOGY.md", "course/METHODOLOGY.zh-CN.md"),
    ("6.12", "course/METHODOLOGY.md", "course/METHODOLOGY.zh-CN.md"),
    ("55.21", "course/SYLLABUS.md", "course/SYLLABUS.zh-CN.md"),
]


def parse_legacy_quality_status(qs: str) -> tuple[str | None, str]:
    """Strictly parse a legacy `quality_status` string.

    Returns (verdict_or_None, reason). Accepted: exact enum, or
    'ENUM (prose)' where prose contains no other status token.
    Everything else — mixed, empty, sentence, unknown — is rejected.
    """
    qs = (qs or "").strip()
    if not qs:
        return None, "empty status"
    for token in _ALL_STATUS_TOKENS:
        if qs == token:
            return token, ""
        if qs.startswith(token):
            rest = qs[len(token):].lstrip()
            if rest.startswith("(") and rest.endswith(")"):
                inner = rest[1:-1]
                others = {t for t in _ALL_STATUS_TOKENS if t != token and t in inner}
                if others:
                    return None, (f"mixed-status prose contains conflicting tokens "
                                  f"{sorted(others)}")
                return token, ""
            return None, f"not an exact enum value (got {qs!r})"
    return None, f"unknown status {qs!r}"


def _evidence_resolves(root: Path, evidence_path: str) -> bool:
    first = re.split(r"\s+", (evidence_path or "").strip(), maxsplit=1)[0].rstrip(".,;:")
    if not first:
        return False
    if first.startswith(EVIDENCE_HOST_ONLY_PREFIXES):
        return True
    return (root / first).exists()


def validate_registry(root: Path) -> list[str]:
    path = root / "course/data/verified_claims.json"
    if not path.is_file():
        return ["registry missing: course/data/verified_claims.json"]
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        return [f"registry is not valid JSON: {exc}"]
    problems: list[str] = []
    claims = data.get("claims")
    if not isinstance(claims, list) or not claims:
        return ["registry has no claims list"]
    seen_ids: set[str] = set()
    for c in claims:
        cid = c.get("id", "<no id>")
        if cid in seen_ids:
            problems.append(f"duplicate claim id {cid}")
        seen_ids.add(cid)

    owner_record_ok = _owner_record_scope_ok(root)
    for c in claims:
        cid = c.get("id", "<no id>")
        for f in REQUIRED_FIELDS:
            if not c.get(f):
                problems.append(f"claim {cid}: missing field '{f}'")
        if c.get("status") not in STATUS_VOCAB:
            problems.append(f"claim {cid}: status {c.get('status')!r} not in {STATUS_VOCAB}")

        canonical = c.get("measurement_status") or c.get("numerical_verdict") \
            or c.get("human_quality_verdict") or c.get("release_decision")
        if canonical is None:
            # legacy-only entry: strict-parse quality_status
            verdict, reason = parse_legacy_quality_status(str(c.get("quality_status", "")))
            if verdict is None:
                problems.append(f"claim {cid}: quality_status rejected: {reason}")
        else:
            for field, enum in (("measurement_status", MEASUREMENT_ENUM),
                                ("numerical_verdict", NUMERICAL_ENUM),
                                ("human_quality_verdict", HUMAN_ENUM),
                                ("release_decision", RELEASE_ENUM)):
                v = c.get(field)
                if v is None:
                    problems.append(f"claim {cid}: missing canonical field '{field}'")
                elif v not in enum:
                    problems.append(f"claim {cid}: {field} {v!r} not an exact enum value")

        # legacy quality_status, when present alongside canonical fields or
        # alone, must still strictly parse (no mixed prose anywhere)
        if "quality_status" in c:
            verdict, reason = parse_legacy_quality_status(str(c["quality_status"]))
            if verdict is None:
                problems.append(f"claim {cid}: quality_status rejected: {reason}")

        # immutable verdicts — tampering is a hard error (canonical field if
        # present, else the legacy parse)
        if cid in IMMUTABLE:
            rule = IMMUTABLE[cid]
            got = c.get(rule["field"])
            if got is None and rule["field"] == "measurement_status":
                got = c.get("status", "").upper() if c.get("status") else None
            if got is None and "quality_status" in c:
                got, _ = parse_legacy_quality_status(str(c["quality_status"]))
            if got != rule["value"]:
                problems.append(
                    f"claim {cid}: immutable {rule['field']}={rule['value']} "
                    f"altered to {got!r}")

        # cross-field fail-closed semantics: an experiment that never ran can
        # hold no verdicts and no release decision of any kind
        ms_now = c.get("measurement_status") or (
            {"measured": "MEASURED", "derived": "DERIVED", "estimated": "ESTIMATED",
             "historical": "HISTORICAL", "NOT_RUN": "NOT_RUN"}.get(str(c.get("status"))))
        if ms_now == "NOT_RUN":
            for field in ("numerical_verdict", "human_quality_verdict", "release_decision"):
                v = c.get(field)
                if v is not None and v != "NOT_RUN":
                    problems.append(
                        f"claim {cid}: measurement_status is NOT_RUN but {field}={v!r} "
                        f"— an unrun experiment can hold no verdict")

        # derived claims need a verifiable derivation
        ms = c.get("measurement_status") or ("DERIVED" if c.get("status") == "derived" else None)
        if ms == "DERIVED":
            d = c.get("derivation") or {}
            if not (d.get("numerator_ref") and d.get("denominator_ref")):
                problems.append(
                    f"claim {cid}: DERIVED claim requires derivation with "
                    f"numerator_ref and denominator_ref")
            elif not {d["numerator_ref"], d["denominator_ref"]} <= seen_ids | {cid}:
                pending = {d["numerator_ref"], d["denominator_ref"]}
                problems.append(
                    f"claim {cid}: derivation references unknown claims "
                    f"{sorted(pending - seen_ids)}")

        # evidence must resolve inside the repo
        if c.get("evidence_path") and not _evidence_resolves(root, str(c["evidence_path"])):
            problems.append(f"claim {cid}: evidence_path does not resolve: "
                            f"{str(c['evidence_path'])[:70]!r}")

        # human-verdict scope containment
        hv = c.get("human_quality_verdict")
        if hv == "OWNER_ACCEPTED_SCOPE_ONLY":
            if cid != OWNER_VERDICT_CLAIM:
                problems.append(
                    f"claim {cid}: OWNER_ACCEPTED_SCOPE_ONLY may only appear on "
                    f"{OWNER_VERDICT_CLAIM} (scope leak)")
            elif str(c.get("verdict_scope", "")) != OWNER_VERDICT_SCOPE:
                problems.append(f"claim {cid}: verdict_scope must equal the committed "
                                f"owner record scope {OWNER_VERDICT_SCOPE!r}")
        if cid in OWNER_VERDICT_FORBIDDEN and hv in ("PASS", "OWNER_ACCEPTED_SCOPE_ONLY"):
            problems.append(
                f"claim {cid}: the M5.1 owner acceptance explicitly does NOT cover "
                f"this experiment; human_quality_verdict must not be {hv!r}")

    # the owner-verdict claim itself must match the committed record
    owner_claim = next((c for c in claims if c.get("id") == OWNER_VERDICT_CLAIM), None)
    if owner_claim is not None:
        if not owner_record_ok:
            problems.append(f"claim {OWNER_VERDICT_CLAIM}: owner record "
                            f"{OWNER_VERDICT_RECORD} missing or scope mismatch")
    return problems


def _owner_record_scope_ok(root: Path) -> bool:
    p = root / OWNER_VERDICT_RECORD
    if not p.is_file():
        return False
    try:
        rec = json.loads(p.read_text(encoding="utf-8"))
    except ValueError:
        return False
    return rec.get("scope") == OWNER_VERDICT_SCOPE and rec.get("verdict") in ("PASS", "FAIL")


def check_parity(root: Path) -> list[str]:
    problems = []
    for value, en_rel, zh_rel in PARITY_VALUES:
        for rel in (en_rel, zh_rel):
            p = root / rel
            if not p.is_file():
                problems.append(f"parity carrier missing: {rel}")
                continue
            if not re.search(re.escape(value), p.read_text(encoding="utf-8")):
                problems.append(f"{rel}: critical value {value} not found (bilingual numeric drift)")
    return problems


def run(root: Path) -> list[str]:
    return validate_registry(root) + check_parity(root)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[2]))
    args = ap.parse_args()
    problems = run(Path(args.root).resolve())
    if problems:
        print(f"CLAIMS CHECK FAILED ({len(problems)} problem(s)):")
        for p in problems:
            print("  " + p)
        return 1
    print("claims check: OK (registry valid; strict status enums; bilingual parity holds)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
