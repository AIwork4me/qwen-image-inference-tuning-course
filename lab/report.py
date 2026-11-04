"""Report aggregation: records -> SUMMARY.json/md + charts.

Scopes (evidence integrity):
- ``reference`` (default): ONLY the frozen official campaign run dirs listed
  in ``lab/reference_runs.json``. This is what the published README numbers
  must equal — smoke, duplicate and student runs can never contaminate it.
  READ-ONLY by default: writing into ``evidence/`` requires the explicit
  ``--write`` flag (the deliberate maintenance path), so a plain
  ``python -m lab report`` can never churn the frozen SUMMARY files.
- ``student``: only new runs under ``runs/`` (the student's own
  measurements), further split by run kind:
  ``formal`` (default — full-protocol sessions, quotable medians) vs
  ``smoke`` (bounded classroom runs) vs ``all`` kinds. Smoke medians are
  NEVER pooled with formal ones by default — the same contamination rule
  that protects the reference campaign, applied to your own data.
- ``all``: reference and student side by side in one document, NEVER pooled
  into shared medians (mixing hosts/sessions would corrupt them).

Charts are generated ONLY from the aggregated summary numbers (never
hand-edited) — a chart that disagrees with SUMMARY.json is a bug.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

from . import lab_env

SCOPES = ("reference", "student", "all")
KINDS = ("formal", "smoke", "all")


def _reference_run_dirs() -> dict[str, list[str]]:
    p = lab_env.LAB_DIR / "reference_runs.json"
    return json.loads(p.read_text(encoding="utf-8"))["reference_run_dirs"]


def _run_kind_of(dirname: str) -> str:
    """Classify a run directory: bounded smoke runs are suffixed '-smoke'."""
    return "smoke" if dirname.endswith("-smoke") else "formal"


def _iter_record_files(scope: str, kinds: tuple[str, ...] = ("formal",)) -> list[Path]:
    """Record JSON files for the requested scope (deterministic order).

    kind filtering applies to the student scope only — the reference
    campaign is pinned to its formal run dirs by construction."""
    files: list[Path] = []
    roots: list[Path] = []
    if scope in ("reference", "all"):
        roots.append(lab_env.RESULTS_ROOT)
    if scope in ("student", "all"):
        roots.append(lab_env.STUDENT_RUNS_ROOT)
    for root in roots:
        if not root.is_dir():
            continue
        if scope in ("reference", "all") and root == lab_env.RESULTS_ROOT:
            allowed = {d for dirs in _reference_run_dirs().values() for d in dirs}
            run_dirs = [root / d for d in sorted(allowed) if (root / d).is_dir()]
        else:
            run_dirs = []
            for d in sorted(root.iterdir()):
                if not d.is_dir() or d.name in ("raw", "env"):
                    continue
                if "all" not in kinds and _run_kind_of(d.name) not in kinds:
                    continue
                run_dirs.append(d)
        for run_dir in run_dirs:
            for f in sorted(run_dir.glob("*.json")):
                files.append(f)
    return files


def _load_records(scope: str = "reference",
                  kinds: tuple[str, ...] = ("formal",)) -> list[dict]:
    records = []
    for f in _iter_record_files(scope, kinds):
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


def _charts(summary: dict, out_dir: Path | None = None) -> list[str]:
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
    out_dir = out_dir or lab_env.RESULTS_ROOT

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
        f"Generated: {s['generated_at']}  ·  scope: **{s['scope']}**  ·  "
        f"kinds: **{s.get('kinds', 'formal')}**  ·  records: {s['n_records']}",
        "",
        "Scope `reference` = the frozen official W7900 campaign only "
        "(lab/reference_runs.json). Student runs live under `runs/` and are "
        "aggregated separately (`python -m lab report --scope student`); "
        "smoke runs are excluded from formal medians by default "
        "(`--kind smoke` to inspect them).",
        "",
    ]
    for section in s.get("sections", [s]):
        lines += _markdown_table(section)
    return "\n".join(lines) + "\n"


def _markdown_table(section: dict) -> list[str]:
    header = section.get("title")
    lines = []
    if header:
        lines += [f"## {header}", ""]
    lines += [
        "| Exp | Category | warm med (s) | step med (s) | fresh (s) | TE (s) | cold (s) | VRAM peak (MB) | util% p50 | attn | n | quality |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for e in section["per_experiment"]:
        lines.append(
            f"| {e['experiment']} | {e['category']} | {e['warm_total_median_s']} "
            f"| {e['warm_step_median_s']} | {e['fresh_total_median_s']} "
            f"| {e['fresh_text_encode_median_s']} | {e['cold_total_s']} "
            f"| {e['vram_device_peak_mb'] or e['vram_peak_mb']} "
            f"| {e['gpu_util_pct_p50']} | {e['attention_backend']} "
            f"| {e['n_measure']} | {e['quality_status']} |")
    return lines + [""]


def _summarize_scope(scope: str, kinds: tuple[str, ...]) -> dict:
    records = _load_records(scope, kinds)
    by_exp: dict[str, list[dict]] = {}
    for r in records:
        by_exp.setdefault(r["experiment"], []).append(r)
    return {
        "n_records": len(records),
        "per_experiment": [summarize_experiment(rs)
                           for _, rs in sorted(by_exp.items())],
    }


def build(write: bool = False, scope: str = "reference",
          kinds: str | tuple[str] = "formal") -> dict:
    if scope not in SCOPES:
        raise ValueError(f"scope must be one of {SCOPES}, got {scope!r}")
    if isinstance(kinds, str):
        kinds = (kinds,)
    kinds = tuple(kinds)
    if kinds not in (("formal",), ("smoke",), ("all",)):
        raise ValueError(f"kinds must be one of {KINDS}, got {kinds!r}")
    kind_label = kinds[0] if len(kinds) == 1 else "all"

    if scope == "all":
        # NEVER pool: the reference campaign and student runs are different
        # measurement populations — render them side by side instead.
        ref = _summarize_scope("reference", ("formal",))
        stu = _summarize_scope("student", kinds)
        summary = {
            "generated_at": lab_env.build_snapshot()["created_at"],
            "scope": "all",
            "kinds": kind_label,
            "n_records": ref["n_records"] + stu["n_records"],
            "reference": ref,
            "student": stu,
            "per_experiment": ref["per_experiment"],  # compat view
            "sections": [
                dict(ref, title="Reference campaign (frozen, formal only)"),
                dict(stu, title=f"Student runs ({kind_label})"),
            ],
        }
    else:
        part = _summarize_scope(scope, kinds)
        summary = {
            "generated_at": lab_env.build_snapshot()["created_at"],
            "scope": scope,
            "kinds": kind_label,
            **part,
        }
    if write:
        if scope == "reference":
            out = lab_env.RESULTS_ROOT
            print("NOTE: writing the reference SUMMARY regenerates the frozen "
                  f"files under {out} (deliberate maintenance action)")
        else:
            out = lab_env.STUDENT_RUNS_ROOT
        out.mkdir(parents=True, exist_ok=True)
        name = "SUMMARY" if scope in ("reference", "student") else "SUMMARY-all"
        (out / f"{name}.json").write_text(json.dumps(summary, indent=2),
                                          encoding="utf-8")
        (out / f"{name}.md").write_text(_markdown(summary), encoding="utf-8")
        summary["charts"] = _charts(summary, out_dir=out)
    return summary
