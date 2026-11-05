#!/usr/bin/env python3
"""Course-quality gate 5: metric-pair population consistency (CPU-only).

Publication rule (publication-closure Phase P1): every (seconds, images/min)
pair that appears CO-LOCATED in one Mermaid node, table cell or slash-pair in
the publication set must be a same-population pair — i.e. it must satisfy the
arithmetic identity  throughput = n_images * 60 / wall_s  (within rounding)
AND match a registered entry in docs/publication-closure/P1_METRIC_PROVENANCE.json.

This gate exists because both READMEs shipped diagrams pairing 737 s (H2
campaign, 0.33 img/min) with 0.35 img/min (the M5.1 rebaseline's 680.0 s arm),
and 50 s (C1b cold-inclusive, 4.80 img/min) with 4.55 img/min (the balanced
warm audit's 52.74 s) — mathematically impossible same-run pairs.

Usage:  python scripts/course-quality/check_metric_pairs.py [--root REPO]
Exit:   0 = all co-located pairs are same-population, 1 otherwise.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

PROVENANCE = "docs/publication-closure/P1_METRIC_PROVENANCE.json"
# Publication set: documents whose displayed numbers are publication-level
# claims (README front matter, syllabus summaries, lessons, lecture, results).
SCAN_GLOBS = (
    "README.md", "README.zh-CN.md",
    "course/SYLLABUS.md", "course/SYLLABUS.zh-CN.md",
    "course/METHODOLOGY.md", "course/METHODOLOGY.zh-CN.md",
    "course/LECTURE_DEMO.md", "course/LECTURE_DEMO.zh-CN.md",
    "course/M05_parallel_heterogeneous/*.md",
    "docs/M51_RESULTS.md", "docs/M51_E2E_AND_PIPELINE.md",
    "docs/quality-v2/PIPELINE_OVERHEAD_STUDY.md",
    "docs/publication-closure/P4_FAIR_CLIENT_AB.md",
)
# time like 737 s / 52.7 s / 50.05 s ; throughput like 0.33 img/min / 4.80 张/分
TIME = r"(\d+(?:\.\d+)?)\s*s\b"
TPUT = r"(\d+(?:\.\d+)?)\s*(?:img/min|images/min|张/分)"
# A co-located pair: time and throughput inside the same mermaid node label
# [...], the same table cell |...|, or the same slash-pair "X s / Y img/min".
NODE_RE = re.compile(r"\[([^\[\]]*)\]")          # mermaid node labels
CELL_RE = re.compile(r"\|([^|\n]*)\|")           # markdown table cells
PAIR_RE = re.compile(TIME + r"\s*/\s*" + TPUT)   # explicit slash pairs
REL_TOL = 0.02                                    # 2% rounding tolerance


def _pairs_from(text: str):
    """Yield (time_s, tput, context) for every co-located pair."""
    spans: list[tuple[float, float, str]] = []
    for regex in (NODE_RE, CELL_RE):
        for m in regex.finditer(text):
            seg = m.group(1)
            for t in re.finditer(TIME, seg):
                for p in re.finditer(TPUT, seg):
                    spans.append((float(t.group(1)), float(p.group(1)), seg.strip()[:80]))
    for m in PAIR_RE.finditer(text):
        spans.append((float(m.group(1)), float(m.group(2)), m.group(0)))
    # de-duplicate identical (time, tput) triples
    return sorted(set(spans))


def check_pairs(root: Path) -> list[str]:
    prov_path = root / PROVENANCE
    if not prov_path.is_file():
        return [f"provenance registry missing: {PROVENANCE}"]
    prov = json.loads(prov_path.read_text(encoding="utf-8"))
    entries = [e for e in prov["entries"]
               if e.get("numerator_s") is not None
               and e.get("throughput_img_per_min") is not None
               and e.get("n_images")]
    problems: list[str] = []
    for rel in _scan_files(root):
        text = (root / rel).read_text(encoding="utf-8", errors="replace")
        for time_s, tput, ctx in _pairs_from(text):
            computed = None
            matched = None
            for e in entries:
                n = e["n_images"]
                if abs(e["numerator_s"] - time_s) / time_s <= 0.015 \
                        and abs(e["throughput_img_per_min"] - tput) / tput <= 0.02:
                    matched = e["id"]
                    break
                if abs(n * 60.0 / time_s - tput) / tput <= REL_TOL and computed is None:
                    computed = f"{n}*60/{time_s}={n * 60.0 / time_s:.3f}"
            if matched:
                continue
            if computed:
                problems.append(
                    f"{rel}: pair ({time_s} s, {tput}) is arithmetically consistent "
                    f"({computed}) but matches NO registered population entry: "
                    f"\"{ctx}\" — register it or fix its provenance")
            else:
                problems.append(
                    f"{rel}: MIXED-POPULATION pair ({time_s} s, {tput}) — 4*60/{time_s}="
                    f"{240 / time_s:.3f} != {tput}: \"{ctx}\" — time and throughput "
                    f"come from different runs; pair them within one population")
    return problems


def _scan_files(root: Path) -> list[str]:
    out: list[str] = []
    for pat in SCAN_GLOBS:
        for p in sorted(root.glob(pat)):
            if p.is_file():
                out.append(str(p.relative_to(root)))
    return sorted(set(out))


def run(root: Path) -> list[str]:
    return check_pairs(root)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[2]))
    args = ap.parse_args()
    problems = run(Path(args.root).resolve())
    if problems:
        print(f"METRIC-PAIR CHECK FAILED ({len(problems)} problem(s)):")
        for p in problems:
            print("  " + p)
        return 1
    print("metric-pair check: OK (every co-located time/throughput pair is same-population)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
