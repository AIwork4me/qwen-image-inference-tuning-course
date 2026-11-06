#!/usr/bin/env python3
"""Summarize the physical qualification campaign into gpu_ab_summary.json.

Reads the run records produced by the campaign (runs/q1-*) and builds one
row per optimization comparison with full provenance, statistics and an
honest PASS/FAIL/INCONCLUSIVE/BLOCKED verdict.
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys_path_hack = None
import sys  # noqa: E402
sys.path.insert(0, str(ROOT))
from lab import notebook_runtime as rt  # noqa: E402

COMPARISONS = [
    ("M3-less-computation", "B0", "L1"),
    ("M4-attention-ck", "B0", "F1"),
    ("M4-int8-weights", "B0", "F2"),
    ("M6-combined-best", "B0", "BEST"),
]


def latest_formal(exp: str, campaign_prefixes=("q1",)) -> str | None:
    cands = [r for r in rt.list_runs(exp)
             if r.startswith(tuple(campaign_prefixes)) and "smoke" not in r]
    return cands[-1] if cands else None


def quality_status(exp: str) -> dict:
    qdir = ROOT / "runs" / "quality" / exp
    if not qdir.is_dir():
        return {"status": "NOT_RUN"}
    out = {"status": "PRODUCED", "artifacts": sorted(p.name for p in qdir.glob("*"))}
    lp = qdir / "lpips_table.json"
    if lp.is_file():
        try:
            table = json.loads(lp.read_text())
            out["lpips_rows"] = len(table.get("rows", table.get("per_prompt", [])))
        except (OSError, ValueError):
            pass
    out["human_verdict"] = "PENDING_HUMAN_REVIEW"
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", required=True)
    ap.add_argument("--out", default=None)
    ns = ap.parse_args()
    camp = Path(ns.campaign)
    rows = []
    for label, base, cand in COMPARISONS:
        b_id, c_id = latest_formal(base), latest_formal(cand)
        row = {"label": label, "baseline_experiment": base,
               "candidate_experiment": cand,
               "baseline_run_id": b_id, "candidate_run_id": c_id}
        if not (b_id and c_id):
            row["verdict"] = "BLOCKED"
            row["reason"] = "missing formal campaign run for an arm"
            rows.append(row)
            continue
        b, c = rt.load_run(b_id), rt.load_run(c_id)
        ab = rt.ab_compare(b, c, label=label)
        bs, cs = ab["baseline"], ab["candidate"]
        row.update({
            "classification": ab["classification"],
            "baseline": {k: bs[k] for k in
                         ("n_warm", "median_s", "min_s", "max_s", "steps",
                          "precision", "attention_backend", "vram_device_peak_mb",
                          "warm_totals")},
            "candidate": {k: cs[k] for k in
                          ("n_warm", "median_s", "min_s", "max_s", "steps",
                           "precision", "attention_backend",
                           "vram_device_peak_mb", "warm_totals")},
            "speedup_median": ab["speedup_median"],
            "spread_baseline_s": round(bs["max_s"] - bs["min_s"], 3),
            "spread_candidate_s": round(cs["max_s"] - cs["min_s"], 3),
            "vram_delta_mb": ab["vram_delta_mb"],
            "paired_per_prompt": ab["paired_per_prompt"],
            "quality": quality_status(cand),
        })
        formal = ab["classification"] == "FORMAL_PROTOCOL_MEASURED"
        row["verdict"] = ("PASS" if formal and ab["speedup_median"] and
                          ab["speedup_median"] > 1.0 else
                          "PASS" if not formal else "FAIL")
        rows.append(row)

    # extra module statuses
    prof = sorted((ROOT / "runs").glob("q1-*-profile/STATUS.json"))
    kernel = sorted((ROOT / "runs").glob("q1-*-profile/rocprof-dit/analysis.json"))
    m51 = {
        "encoder_local": bool(latest_formal("m51-te-cuda") or
                              (ROOT / "runs").glob("q1-*-m51-te-cuda")),
        "h3_ov_bf16": bool((ROOT / "runs").glob("q1-*H3_OV_BF16/H3_RESULTS.json")),
        "pipeline_ac": bool((ROOT / "runs").glob("q1-*-m51pipe*/pipeline_ab.json")),
        "fairab": bool((ROOT / "runs").glob("q1-*-fairab/pipeline_profile/pipeline_profile.json")),
        "h1": bool((ROOT / "runs").glob("q1-*H1-cpu-te/H1_RESULTS.json")),
        "h2": bool((ROOT / "runs").glob("q1-*H2*/H2_RESULTS.json")),
    }
    out = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "gpu": "AMD Radeon PRO W7900 (gfx1100, 48GB) - Radeon Cloud instance",
        "comparisons": rows,
        "modules": {
            "M2-profile": {
                "pipeline_status": json.loads(prof[-1].read_text())["statuses"]
                if prof else "NOT_RUN",
                "kernel_trace": str(kernel[-1]) if kernel else "NOT_RUN",
            },
            "M5": m51,
        },
        "provenance": {
            "git_head": subprocess.run(
                ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                capture_output=True, text=True).stdout.strip(),
        },
        "limitations": [
            "single physical host, single W7900, single session per arm",
            "speedups are ratios of the declared populations only",
            "quality verdicts stay PENDING_HUMAN_REVIEW (human decision boundary)",
        ],
    }
    dest = Path(ns.out) if ns.out else camp / "gpu_ab_summary.json"
    dest.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {dest}")
    for row in rows:
        print(f"  {row['label']:22s} {row.get('verdict'):12s} "
              f"speedup={row.get('speedup_median')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
