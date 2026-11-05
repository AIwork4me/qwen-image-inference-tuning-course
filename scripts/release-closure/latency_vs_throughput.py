"""Service latency vs actual batch throughput — corrected distinction (PR2 polish).

Reads committed raw timestamps only and separates:
- per-request SERVICE LATENCY: wall_s / gen_wall_s (what one requester waits)
- INTER-REQUEST GAP: submission/monitoring/queueing overhead between requests,
  derived from raw t_end / gen_start / gen_end timestamps
- ACTUAL BATCH THROUGHPUT: n images / true batch wall (gap-inclusive)

Sources:
- campaign  runs/m51-20261105-cleanroom-01/pipeline/pipeline_ab.json (arms A-D)
- audit     runs/pr2-p4-pipewarm-120822/pipeline_warm/pipeline_warm.json (A/A2/C)

Usage: .venv-lab/bin/python scripts/release-closure/latency_vs_throughput.py
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "docs/release-closure/PR2_LATENCY_VS_THROUGHPUT.json"


def campaign_arm_A(d: dict) -> dict:
    reqs = d["arms"]["A"]["requests"]
    walls = [r["wall_s"] for r in reqs]
    t_end = [r["t_end"] for r in reqs]
    gaps = [round(t_end[i + 1] - t_end[i] - walls[i + 1], 3) for i in range(len(reqs) - 1)]
    batch_span = t_end[-1] - t_end[0] + walls[0]  # first-request start .. last end
    return {"arm": "A (campaign, cold-inclusive session)", "n": len(reqs),
            "service_latency_s": [round(w, 3) for w in walls],
            "first_request_s": walls[0],
            "steady_service_median_s": round(statistics.median(walls[1:]), 3),
            "inter_request_gaps_s": gaps,
            "sum_service_s": round(sum(walls), 3),
            "batch_span_from_timestamps_s": round(batch_span, 3),
            "recorded_total_s": d["arms"]["A"]["total_s"],
            "prebatch_overhead_s": round(d["arms"]["A"]["total_s"] - batch_span, 3),
            "actual_batch_throughput_img_per_min": round(len(reqs) * 60 / d["arms"]["A"]["total_s"], 2),
            "naive_sum_throughput_img_per_min": round(len(reqs) * 60 / sum(walls), 2)}


def campaign_pipelined(d: dict, arm: str) -> dict:
    a = d["arms"][arm]
    reqs = a["requests"]
    gen = [r["gen_wall_s"] for r in reqs]
    gaps = [round(reqs[i + 1]["gen_start"] - reqs[i]["gen_end"], 3) for i in range(len(reqs) - 1)]
    return {"arm": f"{arm} (campaign, cold-inclusive session)", "n": len(reqs),
            "service_latency_s": [round(w, 3) for w in gen],
            "steady_service_median_s": round(statistics.median(gen), 3),
            "inter_request_gaps_s": gaps,
            "sum_service_s": round(sum(gen), 3),
            "recorded_total_s": a["total_s"],
            "fill_s": a.get("fill_s"),
            "actual_batch_throughput_img_per_min": round(len(reqs) * 60 / a["total_s"], 2),
            "naive_sum_throughput_img_per_min": round(len(reqs) * 60 / (sum(gen) + (a.get("fill_s") or 0)), 2)}


def warm_arm(d: dict, arm: str) -> dict:
    a = d["arms"][arm]
    key = "wall_s" if arm in ("A", "A2") else "gen_wall_s"
    batches = a["batches"]
    rows, gaps_all = [], []
    for b in batches:
        reqs = b["requests"]
        walls = [r[key] for r in reqs]
        if arm in ("A", "A2"):
            t_end = [r["t_end"] for r in reqs]
            gaps = [round(t_end[i + 1] - t_end[i] - walls[i + 1], 3) for i in range(len(reqs) - 1)]
        else:
            gaps = [round(reqs[i + 1]["gen_start"] - reqs[i]["gen_end"], 3) for i in range(len(reqs) - 1)]
        rows.append({"batch": b["batch"], "service_latency_s": [round(w, 3) for w in walls],
                     "inter_request_gaps_s": gaps,
                     "batch_wall_s": b["batch_wall_s"],
                     "gap_total_s": round(b["batch_wall_s"] - sum(walls), 3),
                     "actual_batch_img_per_min": round(len(reqs) * 60 / b["batch_wall_s"], 2),
                     "naive_sum_img_per_min": round(len(reqs) * 60 / sum(walls), 2)})
        gaps_all += gaps
    allwalls = [r[key] for b in batches for r in b["requests"]]
    return {"arm": f"{arm} (audit warm session)", "n": len(allwalls),
            "service_latency_median_s": round(statistics.median(allwalls), 3),
            "service_latency_min_max_s": [round(min(allwalls), 3), round(max(allwalls), 3)],
            "per_batch": rows,
            "inter_request_gap_median_s": round(statistics.median(gaps_all), 3) if gaps_all else None,
            "inter_request_gap_max_s": round(max(gaps_all), 3) if gaps_all else None}


def main() -> int:
    camp = json.loads((REPO / "runs/m51-20261105-cleanroom-01/pipeline/pipeline_ab.json").read_text())
    warm = json.loads((REPO / "runs/pr2-p4-pipewarm-120822/pipeline_warm/pipeline_warm.json").read_text())

    wa, wc = warm_arm(warm, "A"), warm_arm(warm, "C")
    a_eff = statistics.mean([b["actual_batch_img_per_min"] for b in wa["per_batch"]])
    c_eff = statistics.mean([b["actual_batch_img_per_min"] for b in wc["per_batch"]])

    report = {
        "definitions": {
            "service_latency": "wall_s/gen_wall_s of a single request (what one requester waits)",
            "inter_request_gap": "time between consecutive requests that is NOT service: "
                                 "workflow submission, cond-file reads, monitoring, queueing",
            "actual_batch_throughput": "n images divided by the true batch wall (gaps included); "
                                       "the naive sum-of-walls figure OVERSTATES it",
        },
        "campaign_cold_inclusive": {
            "A": campaign_arm_A(camp),
            "C": campaign_pipelined(camp, "C"),
            "D": campaign_pipelined(camp, "D"),
            "B": campaign_pipelined(camp, "B"),
        },
        "audit_warm_sessions": {"A": wa, "A2_control": warm_arm(warm, "A2"), "C": wc},
        "corrected_summary": {
            "warm_service_latency_ratio_A_over_C": round(
                wa["service_latency_median_s"] / wc["service_latency_median_s"], 4),
            "warm_actual_batch_throughput_img_per_min": {"A": round(a_eff, 2), "C": round(c_eff, 2)},
            "warm_actual_batch_throughput_ratio_C_over_A": round(c_eff / a_eff, 4),
            "interpretation": (
                "Warm per-request service latency improves 8.05% (13.031 -> 12.060 s, n=8/8, "
                "non-overlapping). ACTUAL four-request batch throughput improves only ~3% "
                "(~4.41 -> ~4.55 img/min) because the precomputed arm carries ~0.54 s/request "
                "more inter-request overhead (cond-file staging + workflow submission). "
                "Both numbers are honest; they answer different questions. Any 'images/min' "
                "computed from sum-of-walls must be labeled naive/upper-bound."),
        },
    }
    OUT.write_text(json.dumps(report, indent=2))
    print(json.dumps(report["corrected_summary"], indent=1))
    print("campaign A gaps:", report["campaign_cold_inclusive"]["A"]["inter_request_gaps_s"],
          "prebatch_overhead_s:", report["campaign_cold_inclusive"]["A"]["prebatch_overhead_s"])
    print("WROTE", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
