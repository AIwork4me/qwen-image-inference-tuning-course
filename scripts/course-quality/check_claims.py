#!/usr/bin/env python3
"""Course-quality gate 3: numerical parity of critical claims (CPU-only).

Two jobs, both deliberately non-brittle (semantic scopes, never a global
numeric-string diff):

1. Registry validation: course/data/verified_claims.json must parse, every
   claim carries the mandatory provenance fields, statuses come from the
   declared vocabularies, and immutable verdicts keep their states
   (INT8 negative FAIL can never be registered as PASS).
2. Bilingual parity: each critical value taught by the course (from the
   registry's parity list below) must appear in BOTH language READMEs and,
   where declared, in both language lesson carriers.

Usage:  python scripts/course-quality/check_claims.py [--root REPO]
Exit:   0 = registry valid + parity holds, 1 otherwise.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REQUIRED_FIELDS = ("id", "status", "evidence_path")
STATUS_VOCAB = ("measured", "derived", "estimated", "historical", "NOT_RUN")
IMMUTABLE = {
    "M51-INT8-NEGATIVE-FAIL": ("FAIL",),          # relL2 28.25% verdict
    "M51-C1B-UNCOMPRESSED-FAIL": ("FAIL",),
    "SWEEP-40STEP-CFG": ("NOT_RUN",),
}
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


def validate_registry(root: Path) -> list[str]:
    path = root / "course/data/verified_claims.json"
    if not path.is_file():
        return ["registry missing: course/data/verified_claims.json"]
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        return [f"registry is not valid JSON: {exc}"]
    problems = []
    claims = data.get("claims")
    if not isinstance(claims, list) or not claims:
        return ["registry has no claims list"]
    seen_ids = set()
    for c in claims:
        cid = c.get("id", "<no id>")
        if cid in seen_ids:
            problems.append(f"duplicate claim id {cid}")
        seen_ids.add(cid)
        for f in REQUIRED_FIELDS:
            if not c.get(f):
                problems.append(f"claim {cid}: missing field '{f}'")
        if c.get("status") not in STATUS_VOCAB:
            problems.append(f"claim {cid}: status {c.get('status')!r} not in {STATUS_VOCAB}")
        qs = str(c.get("quality_status", ""))
        if "PASS" in IMMUTABLE.get(cid, ()) and "PASS" not in qs:
            problems.append(f"claim {cid}: this verdict may not be registered as PASS (got {qs!r})")
        for cid_imm, allowed in IMMUTABLE.items():
            if cid == cid_imm:
                ok = any(a in qs for a in allowed) or qs == allowed[0]
                if not ok and any(a in qs for a in ("PASS",)) and "FAIL" not in qs and "NOT_RUN" not in qs:
                    problems.append(f"claim {cid}: immutable status {allowed} altered to {qs!r}")
    return problems


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
    print("claims check: OK (registry valid; bilingual numerical parity holds)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
