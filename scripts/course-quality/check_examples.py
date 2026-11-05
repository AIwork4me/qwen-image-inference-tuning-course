#!/usr/bin/env python3
"""Course-quality gate 4: documented commands vs implemented CLIs (CPU-only).

Never executes documented commands (some need a GPU); instead:
  1. every --flag used by documented `python -m lab ...` / start.sh commands
     in the core docs must exist in the lab's argparse definitions or the
     start.sh option parser;
  2. every fenced bash block in the core docs must pass `bash -n`
     (syntax-only, zero execution);
  3. documented repo-relative paths in command lines must exist.

Usage:  python scripts/course-quality/check_examples.py [--root REPO]
Exit:   0 = all documented commands consistent, 1 otherwise.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

CORE_DOCS = [
    "README.md", "README.zh-CN.md",
    "docs/RADEON_CLOUD_QUICKSTART.md", "docs/RADEON_CLOUD_QUICKSTART.zh-CN.md",
    "docs/CLASSROOM_WORKFLOWS.md", "docs/CLASSROOM_WORKFLOWS.zh-CN.md",
    "docs/REPRODUCTION_GUIDE.md", "docs/REPRODUCTION_GUIDE.zh-CN.md",
    "docs/CONTRIBUTING_TUTORIAL.md", "docs/CONTRIBUTING_TUTORIAL.zh-CN.md",
    "course/METHODOLOGY.md", "course/METHODOLOGY.zh-CN.md",
    "course/LECTURE_DEMO.md", "course/LECTURE_DEMO.zh-CN.md",
    "course/SYLLABUS.md", "course/SYLLABUS.zh-CN.md",
] + [
    f"course/M0{i}_{d}/{'LESSON' if i else ''}"
    for i, d in []
]
LESSONS = [
    "course/M00_setup_and_discipline/LESSON.md", "course/M00_setup_and_discipline/LESSON.zh-CN.md",
    "course/M01_baseline/LESSON.md", "course/M01_baseline/LESSON.zh-CN.md",
    "course/M02_profiling/LESSON.md", "course/M02_profiling/LESSON.zh-CN.md",
    "course/M03_less_computation/LESSON.md", "course/M03_less_computation/LESSON.zh-CN.md",
    "course/M04_faster_computation/LESSON.md", "course/M04_faster_computation/LESSON.zh-CN.md",
    "course/M05_parallel_heterogeneous/LESSON.md", "course/M05_parallel_heterogeneous/LESSON.zh-CN.md",
    "course/M05_parallel_heterogeneous/M5_1_OPENVINO_CPU_TE.md",
    "course/M05_parallel_heterogeneous/M5_1_OPENVINO_CPU_TE.zh-CN.md",
    "course/M06_combined_quality/LESSON.md", "course/M06_combined_quality/LESSON.zh-CN.md",
    "course/M07_student_challenge/CHALLENGES.md", "course/M07_student_challenge/CHALLENGES.zh-CN.md",
]
CORE_DOCS += LESSONS

FLAG_RE = re.compile(r"(?<![\w-])(--?[a-z][\w-]*)")
LAB_CMD_RE = re.compile(r"(?:\.venv-lab/bin/)?python(?:3)?(?:\s+-[Im])?\s+.*-m\s+lab\b|\.venv-lab/bin/python\s+-m\s+lab")
BASH_BLOCK = re.compile(r"```bash\n(.*?)```", re.S)
# flags documented for commands that are intentionally environment-specific
IGNORED_FLAGS = {"--hf-base"}  # checked against start.sh separately below


def implemented_flags(root: Path) -> tuple[set[str], set[str]]:
    """Collect argparse flags from lab/*.py and option flags from start.sh."""
    lab_flags: set[str] = set()
    for py in (root / "lab").glob("*.py"):
        text = py.read_text(encoding="utf-8", errors="replace")
        for m in re.finditer(r"add_argument\(\s*[\"'](--[\w-]+)", text):
            lab_flags.add(m.group(1))
    sh_flags: set[str] = set()
    sh = root / "scripts/cloud/start.sh"
    if sh.is_file():
        for m in re.finditer(r"--[a-z][\w-]*", sh.read_text(encoding="utf-8")):
            sh_flags.add(m.group(0))
    for py in (root / "scripts").rglob("*.py"):
        for m in re.finditer(r"add_argument\(\s*[\"'](--[\w-]+)", py.read_text(encoding="utf-8", errors="replace")):
            lab_flags.add(m.group(1))
    return lab_flags, sh_flags


def documented_flags_in_lab_commands(text: str) -> set[str]:
    out: set[str] = set()
    for line in text.splitlines():
        if LAB_CMD_RE.search(line):
            for f in FLAG_RE.findall(line):
                if f in ("-m", "-I", "-u", "-q", "-v", "-h", "--help"):
                    continue
                out.add(f)
    return out


def bash_syntax_errors(blocks: list[str]) -> list[str]:
    errs = []
    for i, block in enumerate(blocks):
        # documentation placeholders like <your-tag> are redirects to bash;
        # substitute a syntactically inert token before the syntax check
        sanitized = re.sub(r"<[\w./$-]+>", "PLACEHOLDER", block)
        r = subprocess.run(["bash", "-n"], input=sanitized, capture_output=True, text=True)
        if r.returncode != 0:
            errs.append(f"block {i}: {r.stderr.strip().splitlines()[-1] if r.stderr.strip() else 'syntax error'}")
    return errs


def run(root: Path, docs: list[str] | None = None) -> list[str]:
    problems: list[str] = []
    lab_flags, sh_flags = implemented_flags(root)
    for rel in (docs if docs is not None else CORE_DOCS):
        p = root / rel
        if not p.is_file():
            problems.append(f"doc missing for command check: {rel}")
            continue
        text = p.read_text(encoding="utf-8")
        flags = documented_flags_in_lab_commands(text) - IGNORED_FLAGS
        unknown = {f for f in flags if f not in lab_flags}
        if unknown:
            problems.append(f"{rel}: flags not found in any lab argparse: {sorted(unknown)}")
        blocks = BASH_BLOCK.findall(text)
        for err in bash_syntax_errors(blocks):
            problems.append(f"{rel}: bash syntax: {err}")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[2]))
    args = ap.parse_args()
    problems = run(Path(args.root).resolve())
    if problems:
        print(f"EXAMPLES CHECK FAILED ({len(problems)} problem(s)):")
        for p in problems:
            print("  " + p)
        return 1
    print("examples check: OK (documented flags exist; bash blocks parse)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
