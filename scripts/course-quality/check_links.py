#!/usr/bin/env python3
"""Course-quality gate 1: local link integrity (CPU-only, no network).

Checks every Markdown link and <img src> in the tracked .md files of the
repository core (course/, docs/, README*): relative targets must resolve,
anchors (#...) are ignored, http(s)/mailto targets are reported separately
but never failed (network-free gate).

Usage:  python scripts/course-quality/check_links.py [--root REPO] [--list-external]
Exit:   0 = no broken links, 1 = broken links found.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

MD_LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
IMG_SRC = re.compile(r"""<img\s+[^>]*src=["']([^"']+)["']""", re.I)
BARE_PATH = re.compile(r"`((?:docs|course|evidence|runs|lab|scripts)/[\w./-]+)`")

# Paths that legitimately exist only on the measurement host (documented as
# host-only in the evidence audits): never required to resolve in git.
HOST_ONLY_PREFIXES = ("runs/raw/", "runs/quality/", "runs/m51-20261105-cleanroom-01/e2e/raw/")
# Forward-looking destinations described in instructional text (files the
# owner or students will create): not existence claims, never flagged.
FUTURE_DESTINATIONS = (
    "evidence/quality/VERDICTS.md",       # created by the owner at verdict time
    "course/worksheets/M1_lecture_worksheet.md",  # proposed in the readiness gap table
    "evidence/p0/",                       # host-only env snapshots (verbatim journals)
)
# Verbatim historical engineering journals reference host-only artifacts by
# design; they are reading material, not navigation docs.
BARE_PATH_SKIP_DIRS = ("case-study",)


def tracked_markdown(root: Path) -> list[Path]:
    out = subprocess.run(["git", "ls-files", "*.md"], cwd=root,
                         capture_output=True, text=True).stdout.split("\n")
    files = [root / p for p in out if p.endswith(".md") and (root / p).is_file()]
    # the two vendored trees are not course content
    return [f for f in files if ".venv" not in f.parts]


def extract_targets(md_text: str) -> tuple[set[str], set[str], set[str]]:
    """Return (local_targets, anchors, external_urls) found in the text."""
    local, anchors, ext = set(), set(), set()
    for m in MD_LINK.finditer(md_text):
        t = m.group(1).strip()
        if t.startswith("#"):
            anchors.add(t)
        elif t.startswith(("http://", "https://", "mailto:")):
            ext.add(t)
        else:
            local.add(t)
    for m in IMG_SRC.finditer(md_text):
        t = m.group(1).strip()
        if t.startswith(("http://", "https://")):
            ext.add(t)
        else:
            local.add(t)
    return local, anchors, ext


def check_file(path: Path, root: Path) -> list[str]:
    problems = []
    text = path.read_text(encoding="utf-8", errors="replace")
    local, _anchors, _ext = extract_targets(text)
    for target in sorted(local):
        if target.startswith(HOST_ONLY_PREFIXES):
            continue
        resolved = (path.parent / target.split("#")[0]).resolve()
        if not resolved.exists():
            problems.append(f"{path.relative_to(root)}: broken link -> {target}")
    # bare repo-path references in backticks (common drift source)
    rel_parts = path.relative_to(root).parts
    if rel_parts[0] in ("docs", "course", ".") or path.name.startswith("README"):
        if not any(d in rel_parts for d in BARE_PATH_SKIP_DIRS):
            for m in BARE_PATH.finditer(text):
                raw = m.group(1)
                if raw.startswith(HOST_ONLY_PREFIXES):
                    continue
                t = raw.rstrip("/.")
                if t.startswith(HOST_ONLY_PREFIXES):
                    continue
                if "*" in t or "<" in t or "..." in t:
                    continue
                if any(raw == f or raw.startswith(f) or t == f or t.startswith(f)
                       for f in FUTURE_DESTINATIONS):
                    continue
                if not (root / t).exists():
                    problems.append(f"{path.relative_to(root)}: bare path does not exist -> {t}")
    return problems


def run(root: Path, list_external: bool = False) -> tuple[list[str], set[str]]:
    problems: list[str] = []
    external: set[str] = set()
    for md in tracked_markdown(root):
        try:
            text = md.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            problems.append(f"{md}: unreadable ({exc})")
            continue
        _l, _a, ext = extract_targets(text)
        external |= ext
        problems += check_file(md, root)
    return problems, external


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[2]))
    ap.add_argument("--list-external", action="store_true")
    args = ap.parse_args()
    root = Path(args.root).resolve()
    problems, external = run(root)
    if args.list_external:
        for u in sorted(external):
            print(f"external: {u}")
    if problems:
        print(f"LINK CHECK FAILED ({len(problems)} problem(s)):")
        for p in problems:
            print("  " + p)
        return 1
    print("link check: OK (all local targets resolve)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
