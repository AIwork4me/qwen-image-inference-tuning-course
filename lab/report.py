"""Report aggregation: records -> SUMMARY.json/md + charts.

Charts are generated ONLY from the aggregated summary numbers (never
hand-edited) — a chart that disagrees with SUMMARY.json is a bug.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

from . import lab_env


def _load_records() -> list[dict]:
    records = []
    root = lab_env.RESULTS_ROOT
    if not root.is_dir():
        return records
    for run_dir in sorted(root.iterdir()):
        if not run_dir.is_dir() or run_dir.name in ("raw", "env"):
            continue
        for f in sorted(run_dir.glob("*.json")):
            if f.name.startswith(("server", "H1_", "H2_", "STATUS", "client")):
                continue
            try:
                rec = json.loads(f.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if isinstance(rec, dict) and "experiment" in rec and "runs" in rec:
                records.append(rec)
    return records


def _med_field(records: list[dict], field: str):
    vals = [r.get(field) for r in records if r.get(field) is not None]
    return round(statistics.median(vals), 3) if vals else None


def summarize_experiment(records: list[dict]) -> dict:
    exp = records[0]["experiment"]
    warm = [r for r in records if r["run_kind"] == "measure"]
    fresh = [r for r in records if r["run_kind"] == "fresh_prompt"]
    cold = [r for r in records if r["run_kind"] == "cold"]

    def med(rs, key):
        vals = [r["runs"][0][key] for r in rs if r["runs"][0].get(key) is not None]
        return round(statistics.median(vals), 3) if vals else None

    steps = [r.get("steady_step_median_s") for r in warm
             if r.get("steady_step_median_s")]

    def mx(key):
        vals = [r.get(key) for r in records if r.get(key) is not None]
        return round(max(vals), 1) if vals else None

    return {
        "experiment": exp,
        "category": records[0].get("category"),
        "n_measure": len(warm),
        "n_fresh": len(fresh),
        "warm_total_median_s": med(warm, "total_s"),
        "warm_ksampler_median_s": _med_field(warm, "ksampler_s"),
        "warm_vae_median_s": med(warm, "vae_decode_s"),
        "warm_step_median_s": round(statistics.median(steps), 3) if steps else None,
        "fresh_total_median_s": med(fresh, "total_s"),
        "fresh_text_encode_median_s": med(fresh, "text_encode_s"),
        "cold_total_s": cold[0]["runs"][0]["total_s"] if cold else None,
        "cold_server_boot_s": cold[0].get("server_boot_s") if cold else None,
        "vram_peak_mb": mx("vram_peak_mb"),
        "vram_device_peak_mb": mx("vram_device_peak_mb"),
        "gpu_util_pct_p50": _med_field(warm, "gpu_util_pct_p50"),
        "attention_backend": records[0].get("attention_backend"),
        "run_ids": sorted({r.get("session_id") for r in records}),
        "quality_status": _quality_status(exp),
    }


def _quality_status(exp: str) -> str:
    q = lab_env.QUALITY_ROOT / exp
    if not (q / "quality.md").exists():
        return "PENDING_HUMAN_REVIEW" if exp != "B0" else "REFERENCE"
    return "RECORDED_OR_PENDING_HUMAN_REVIEW"


def _charts(summary: dict) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    exps = [e for e in summary["per_experiment"] if e["warm_total_median_s"]]
    if not exps:
        return []
    written = []
    out_dir = lab_env.RESULTS_ROOT

    order = {"B0": 0, "L1": 1, "L1O": 2, "F1": 3, "F2": 4,
             "LF1": 5, "LF2": 6, "LF3": 7, "BEST": 8}
    exps.sort(key=lambda e: order.get(e["experiment"], 99))
    names = [e["experiment"] for e in exps]
    vals = [e["warm_total_median_s"] for e in exps]
    ns = [e["n_measure"] for e in exps]

    fig, ax = plt.subplots(figsize=(9, 4.2))
    bars = ax.bar(names, vals, color="#7b2cbf")
    for b, v, n in zip(bars, vals, ns):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.6, f"{v:.2f}s\n(n={n})",
                ha="center", fontsize=8)
    ax.set_ylabel("warm generation, median (s)")
    ax.set_title("1024x1024 warm latency per experiment (generated from SUMMARY.json)")
    ax.spines[["top", "right"]].set_visible(False)
    p = out_dir / "chart_warm_latency.png"
    fig.tight_layout()
    fig.savefig(p, dpi=130)
    plt.close(fig)
    written.append(str(p))

    by = {e["experiment"]: e for e in exps}
    stages = [(lab, by[lab]["warm_total_median_s"])
              for lab in ("B0", "L1", "LF1", "LF2") if lab in by]
    if len(stages) >= 3:
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        labels = [s[0] for s in stages]
        values = [s[1] for s in stages]
        ax.step(range(len(stages)), values, where="mid", color="#7b2cbf", lw=2)
        ax.plot(range(len(stages)), values, "o", color="#7b2cbf")
        for i, v in enumerate(values):
            ax.annotate(f"{v:.2f}s", (i, v), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=9)
        ax.set_xticks(range(len(stages)), labels)
        ax.set_ylabel("warm generation (s)")
        ax.set_title("Staged optimization (each stage measured end-to-end)")
        ax.spines[["top", "right"]].set_visible(False)
        p = out_dir / "chart_waterfall.png"
        fig.tight_layout()
        fig.savefig(p, dpi=130)
        plt.close(fig)
        written.append(str(p))
    return written


def _markdown(s: dict) -> str:
    lines = [
        "# Lab results summary (auto-generated)",
        "",
        f"Generated: {s['generated_at']}  ·  records: {s['n_records']}",
        "",
        "| Exp | Category | warm med (s) | step med (s) | fresh (s) | TE (s) | cold (s) | VRAM peak (MB) | util% p50 | attn | n | quality |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for e in s["per_experiment"]:
        lines.append(
            f"| {e['experiment']} | {e['category']} | {e['warm_total_median_s']} "
            f"| {e['warm_step_median_s']} | {e['fresh_total_median_s']} "
            f"| {e['fresh_text_encode_median_s']} | {e['cold_total_s']} "
            f"| {e['vram_device_peak_mb'] or e['vram_peak_mb']} "
            f"| {e['gpu_util_pct_p50']} | {e['attention_backend']} "
            f"| {e['n_measure']} | {e['quality_status']} |")
    return "\n".join(lines) + "\n"


def build(write: bool = False) -> dict:
    records = _load_records()
    by_exp: dict[str, list[dict]] = {}
    for r in records:
        by_exp.setdefault(r["experiment"], []).append(r)
    summary = {
        "generated_at": lab_env.build_snapshot()["created_at"],
        "n_records": len(records),
        "per_experiment": [summarize_experiment(rs) for _, rs in sorted(by_exp.items())],
    }
    if write:
        out = lab_env.RESULTS_ROOT
        out.mkdir(parents=True, exist_ok=True)
        (out / "SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        (out / "SUMMARY.md").write_text(_markdown(summary), encoding="utf-8")
        summary["charts"] = _charts(summary)
    return summary
