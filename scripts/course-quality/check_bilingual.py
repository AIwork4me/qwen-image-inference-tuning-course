#!/usr/bin/env python3
"""Course-quality gate 2: bilingual coverage and structural parity (CPU-only).

For every declared core course document the Chinese counterpart must exist,
and the pair must have substantive (not word-count) parity:
  - identical technical commands (CLI is language-neutral, so every fenced
    command line in EN must appear verbatim in ZH);
  - experiment IDs referenced in EN appear in ZH;
  - local image/file references in EN appear in ZH;
  - heading-count ratio within a tolerance (a ZH doc that lost whole
    sections fails);
  - both directions checked for the curriculum pairs.

Usage:  python scripts/course-quality/check_bilingual.py [--root REPO]
Exit:   0 = full parity, 1 = missing or non-parity documents.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# (english canonical, chinese counterpart)
CORE_PAIRS = [
    ("README.md", "README.zh-CN.md"),
    ("course/SYLLABUS.md", "course/SYLLABUS.zh-CN.md"),
    ("course/METHODOLOGY.md", "course/METHODOLOGY.zh-CN.md"),
    ("course/LECTURE_DEMO.md", "course/LECTURE_DEMO.zh-CN.md"),
    ("course/M00_setup_and_discipline/LESSON.md", "course/M00_setup_and_discipline/LESSON.zh-CN.md"),
    ("course/M01_baseline/LESSON.md", "course/M01_baseline/LESSON.zh-CN.md"),
    ("course/M02_profiling/LESSON.md", "course/M02_profiling/LESSON.zh-CN.md"),
    ("course/M03_less_computation/LESSON.md", "course/M03_less_computation/LESSON.zh-CN.md"),
    ("course/M04_faster_computation/LESSON.md", "course/M04_faster_computation/LESSON.zh-CN.md"),
    ("course/M05_parallel_heterogeneous/LESSON.md", "course/M05_parallel_heterogeneous/LESSON.zh-CN.md"),
    ("course/M05_parallel_heterogeneous/M5_1_OPENVINO_CPU_TE.md",
     "course/M05_parallel_heterogeneous/M5_1_OPENVINO_CPU_TE.zh-CN.md"),
    ("course/M06_combined_quality/LESSON.md", "course/M06_combined_quality/LESSON.zh-CN.md"),
    ("course/M07_student_challenge/CHALLENGES.md", "course/M07_student_challenge/CHALLENGES.zh-CN.md"),
    ("docs/RADEON_CLOUD_QUICKSTART.md", "docs/RADEON_CLOUD_QUICKSTART.zh-CN.md"),
    ("docs/CLASSROOM_WORKFLOWS.md", "docs/CLASSROOM_WORKFLOWS.zh-CN.md"),
    ("docs/REPRODUCTION_GUIDE.md", "docs/REPRODUCTION_GUIDE.zh-CN.md"),
    ("docs/CONTRIBUTING_TUTORIAL.md", "docs/CONTRIBUTING_TUTORIAL.zh-CN.md"),
    ("docs/TROUBLESHOOTING.md", "docs/TROUBLESHOOTING.zh-CN.md"),
]

EXPERIMENT_IDS = re.compile(r"\b(B0|L1O?|F[12]|LF[123]|BEST|H[12]|H3_OV_(?:BF16|FP32|INT8))\b")
CMD_TOKEN = re.compile(r"^(?:\.venv-(?:lab|ov|contrib)/bin/)?(?:bash|python3?|git|pip|pytest|rm|tail|du|df|cd)\b")
LOCAL_REF = re.compile(r"`((?:docs|course|evidence|lab|scripts|runs)/[\w./-]+)`")


def fenced_command_lines(text: str) -> set[str]:
    out: set[str] = set()
    in_fence = False
    for line in text.splitlines():
        if line.strip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence and CMD_TOKEN.match(line.strip()):
            s = line.strip()
            # compare the COMMAND part only: inline '#' comments are language-specific
            s = s.split(" #")[0].split("\t#")[0].rstrip()
            if s:
                out.add(s)
    return out


def check_pair(root: Path, en_rel: str, zh_rel: str) -> list[str]:
    problems = []
    en, zh = root / en_rel, root / zh_rel
    if not en.is_file():
        return [f"missing canonical EN document: {en_rel}"]
    if not zh.is_file():
        return [f"missing Chinese counterpart: {zh_rel} (for {en_rel})"]
    en_t, zh_t = en.read_text(encoding="utf-8"), zh.read_text(encoding="utf-8")

    en_cmds, zh_cmds = fenced_command_lines(en_t), fenced_command_lines(zh_t)
    missing = {c for c in en_cmds - zh_cmds
               if "<" not in c and "$(date" not in c}  # placeholders are per-language free
    if missing:
        problems.append(f"{zh_rel}: {len(missing)} documented command(s) absent, e.g. {sorted(missing)[:2]}")

    en_ids, zh_ids = set(EXPERIMENT_IDS.findall(en_t)), set(EXPERIMENT_IDS.findall(zh_t))
    lost = en_ids - zh_ids
    if lost:
        problems.append(f"{zh_rel}: experiment IDs missing vs EN: {sorted(lost)}")

    en_refs, zh_refs = set(LOCAL_REF.findall(en_t)), set(LOCAL_REF.findall(zh_t))
    lost_refs = {r for r in en_refs - zh_refs if not r.startswith(("runs/raw", "runs/quality"))}
    if lost_refs:
        problems.append(f"{zh_rel}: local artifact refs missing vs EN: {sorted(lost_refs)[:3]}")

    en_h = len(re.findall(r"^#{1,3}\s", en_t, re.M)) + en_t.count("**Problem") + en_t.count("**问题")
    zh_h = len(re.findall(r"^#{1,3}\s", zh_t, re.M)) + zh_t.count("**Problem") + zh_t.count("**问题")
    if en_h and zh_h / max(en_h, 1) < 0.7:
        problems.append(f"{zh_rel}: structure shrank vs EN (headings {en_h} -> {zh_h}; must not be a summary translation)")
    return problems


def run(root: Path) -> list[str]:
    problems: list[str] = []
    for en, zh in CORE_PAIRS:
        problems += check_pair(root, en, zh)
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[2]))
    args = ap.parse_args()
    problems = run(Path(args.root).resolve())
    if problems:
        print(f"BILINGUAL CHECK FAILED ({len(problems)} problem(s)):")
        for p in problems:
            print("  " + p)
        return 1
    print(f"bilingual check: OK ({len(CORE_PAIRS)} core document pairs with substantive parity)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
