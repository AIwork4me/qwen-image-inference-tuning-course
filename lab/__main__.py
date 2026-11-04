"""Command-line interface for the tuning lab.

    python -m lab doctor            # strict preflight (must PASS for formal runs)
    python -m lab doctor --cloud    # Radeon Cloud staged mode (stage-1 weights)
    python -m lab baseline          # run the frozen B0 baseline
    python -m lab profile           # pipeline + GPU kernel profiling
    python -m lab run --experiment L1 [--smoke]
    python -m lab quality --experiment F2
    python -m lab report [--scope reference|student|all]

Every stage writes explicit PASS/FAIL/BLOCKED/WARN status and evidence
paths. NEW measurements always land under runs/ (never inside the frozen
evidence/ campaign).
"""

from __future__ import annotations

import argparse
import json
import sys

from . import manifests


def cmd_doctor(args) -> int:
    from . import doctor
    return doctor.run(quick=args.quick, cloud=args.cloud)


def cmd_baseline(args) -> int:
    from . import runner
    res = runner.run_experiment("B0", date=args.date, port=args.port,
                                smoke=args.smoke)
    print(f"BASELINE_DONE {res}")
    return 0


def cmd_profile(args) -> int:
    from . import profile
    return profile.run(layers=args.layers, date=args.date, port=args.port)


def cmd_run(args) -> int:
    if args.experiment == "H1":
        from .pipeline import run_h1
        run_h1(date=args.date)
        return 0
    if args.experiment == "H2":
        from .pipeline import run_h2
        run_h2(date=args.date, port=args.port)
        return 0
    from . import runner
    res = runner.run_experiment(args.experiment, date=args.date, port=args.port,
                                only_prompt=args.only_prompt, smoke=args.smoke)
    print(f"RUN_DONE {res}")
    return 0


def _resolve_raw_dir(experiment: str, date: str | None = None) -> str:
    """Resolve the RAW IMAGE directory for a run (fixes the historical bug
    where the records dir — which contains no PNGs — was passed to the
    quality gate).

    Order: an explicit --date (run-id) first; then the student's own latest
    run under runs/raw/; then any reference raw dir that still exists on
    this host (the shipped campaign's raw PNGs stayed on the measurement
    host, so a fresh clone has none)."""
    runs_raw = manifests.lab_env.STUDENT_RUNS_ROOT / "raw"
    ref_raw = manifests.lab_env.RESULTS_ROOT / "raw"
    if date:
        d = date if date.endswith(f"-{experiment}") else f"{date}-{experiment}"
        for root in (runs_raw, ref_raw):
            cand = root / d
            if cand.is_dir() and any(cand.glob("*.png")):
                return str(cand)
        raise SystemExit(
            f"no raw images under {runs_raw / d} (or the reference "
            f"{ref_raw / d}) — run the experiment first, or pass the exact "
            "run-id via --date")
    for root in (runs_raw, ref_raw):
        if root.is_dir():
            cands = sorted(root.glob(f"*-{experiment}"))
            cands = [c for c in cands if any(c.glob("*.png"))]
            if cands:
                return str(cands[-1])
    raise SystemExit(
        f"no raw image dir for {experiment!r} under runs/raw/ (the shipped "
        "campaign's PNGs stayed on the measurement host — only their hashes "
        "are committed). Generate your own first:\n"
        f"  python -m lab run --experiment {experiment} --date <your-tag>")


def cmd_quality(args) -> int:
    """Quality gate for one experiment vs the B0 baseline.

    Semantics are derived from the manifest (never guessed):
    fidelity = quantized weights with the same schedule; distance =
    everything that changes the schedule/numerics path.
    """
    import subprocess

    m = manifests.load_manifest(args.experiment)
    semantics = m.get("quality_semantics",
                      "fidelity" if (args.experiment == "F2" and m["steps"] == 40)
                      else "distance")
    baseline = args.baseline or _resolve_raw_dir("B0")
    experiment = _resolve_raw_dir(args.experiment, args.date)
    if not any(x.glob("*.png") for x in map(__import__("pathlib").Path,
                                            (baseline, experiment))):
        raise SystemExit(
            "no raw images at the resolved paths — raw generation PNGs stay "
            "on the measurement host (hashes are in the records).\n"
            "- The SHIPPED campaign's quality evidence is the precomputed grids\n"
            "  and crops under evidence/quality/<experiment>/.\n"
            "- To compute new gates, run the experiment first\n"
            "  (python -m lab run --experiment <ID>) so raw images exist.")
    r = subprocess.run([
        sys.executable, "-I", str(manifests.lab_env.LAB_DIR / "quality_gate.py"),
        "--baseline", baseline,
        "--experiment", experiment,
        "--experiment-name", args.experiment,
        "--semantics", semantics,
        "--baseline-desc", args.baseline_desc or "B0 frozen baseline",
        # student gates land under runs/quality/ — the shipped
        # evidence/quality/<exp>/ grids of the frozen campaign are never
        # overwritten by new measurements
        "--out", str(manifests.lab_env.STUDENT_RUNS_ROOT / "quality" / args.experiment),
    ])
    return r.returncode


def _latest_run(experiment: str) -> str:  # retained for backwards compat
    runs = sorted(manifests.lab_env.RESULTS_ROOT.glob(f"*-{experiment}"))
    if not runs:
        raise SystemExit(f"no run dir for {experiment!r} under evidence/results/")
    return str(runs[-1])


def cmd_report(args) -> int:
    from . import report
    out = report.build(write=True, scope=args.scope)
    print(json.dumps({k: v for k, v in out.items() if k != "per_experiment"} |
                     {"per_experiment_count": len(out.get("per_experiment", []))},
                      indent=2, default=str))
    return 0


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m lab", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("doctor", help="environment preflight")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--cloud", action="store_true",
                   help="Radeon Cloud staged mode: stage-1 weights + "
                        "non-W7900 Radeon GPUs produce warnings, not failures")
    p.set_defaults(func=cmd_doctor)

    p = sub.add_parser("baseline", help="run the frozen B0 baseline")
    p.add_argument("--date", default=None)
    p.add_argument("--port", type=int, default=8188)
    p.add_argument("--smoke", action="store_true")
    p.set_defaults(func=cmd_baseline)

    p = sub.add_parser("profile", help="pipeline + kernel profiling")
    p.add_argument("--layers", default="pipeline,kernel")
    p.add_argument("--date", default=None)
    p.add_argument("--port", type=int, default=8199)
    p.set_defaults(func=cmd_profile)

    p = sub.add_parser("run", help="run one experiment manifest")
    p.add_argument("--experiment", required=True,
                   help=f"one of {manifests.KNOWN_IDS}")
    p.add_argument("--date", default=None)
    p.add_argument("--port", type=int, default=8188)
    p.add_argument("--only-prompt", default=None)
    p.add_argument("--smoke", action="store_true")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("quality", help="quality gate for one experiment")
    p.add_argument("--experiment", required=True)
    p.add_argument("--baseline", default=None, help="raw dir of baseline runs")
    p.add_argument("--date", default=None)
    p.add_argument("--baseline-desc", default=None)
    p.set_defaults(func=cmd_quality)

    p = sub.add_parser("report", help="aggregate results into SUMMARY + charts "
                      "(default scope: frozen reference campaign only)")
    p.add_argument("--scope", choices=("reference", "student", "all"),
                   default="reference",
                   help="reference = frozen official campaign (published "
                        "numbers); student = your runs/ only; all = pooled")
    p.set_defaults(func=cmd_report)

    ns = ap.parse_args(argv)
    rc = ns.func(ns)
    sys.exit(rc)


if __name__ == "__main__":
    main()
