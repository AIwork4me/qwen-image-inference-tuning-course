"""P4 fair-client factorial A/B: architecture (GPU-native TE vs OpenVINO CPU
TE) × client (original per-request monitor vs persistent monitor).

Research question (course-v2 P4, pre-registered 2026-11-06):
  When BOTH arms A and C use the same optimized client, how much end-to-end
  four-request batch-throughput improvement remains attributable to OpenVINO
  CPU offloading — and how much of the previously observed A/C batch gap was
  the measurement client itself?

Why this exists: the P2 study (docs/quality-v2/PIPELINE_OVERHEAD_STUDY.md)
measured C[poll] 52.761 s vs C[fast] 50.320 s (+4.85%) and noted that
comparing C-fast against A-original is NOT a fair architecture estimate,
because A-original also pays per-request monitor teardown. This module adds
the missing A-fast arm so the 2×2 factorial is complete:

    A-original : GPU-native TE, per-request RunMonitor (production path)
    A-fast     : GPU-native TE, persistent RunMonitor (this module)
    C-original : OV BF16 CPU TE, per-request RunMonitor (production path)
    C-fast     : OV BF16 CPU TE, persistent RunMonitor (P2 C-opt)

Pre-registered protocol (before any measurement):
  * Frozen 4 prompts / token IDs / seed / latent / DiT / VAE / scheduler /
    8 steps / cfg=1 / 1024x1024 — byte-identical workflows to the P2 study.
  * One warm server session per architecture per block; two warmup requests
    for session A (the documented post-warmup CLIP-eviction reload, if it
    fires, must land in a warmup, never in a measured batch) and one warmup
    + negative for session C (no GPU CLIP exists in that session).
  * Within each warm session, batches alternate poll,fast,poll,fast,... so
    monitor-mode comparisons are interleaved for drift control.
  * Two blocks (A-then-C, then C-then-A) => 3 batches per arm×mode cell
    per block, 6 per cell total if both blocks complete.
  * Exclusion rule (pre-declared): a batch is excluded ONLY on fatal
    technical failure (server execution error / worker crash), is retained
    in the raw record marked failed, and is excluded from every view with
    the failure documented. NO performance-based exclusion ever. Model
    reloads inside measured batches are captured in node timings and
    reported; the primary view includes them, a clearly-labeled derived
    warm-resident view separates them.
  * Correctness contract: conditioning-file SHA256 per request (arm C),
    identical seeds/workflows, image bytes differ only by the documented
    run-to-run kernel nondeterminism.

Run (physical W7900-class host, GPU + .venv-ov + M51_OV_ARTIFACTS):
  M51_OV_ARTIFACTS=/opt/m51-artifacts \
    .venv-lab/bin/python -m lab.m51_pipeline_fair_ab --run runs/<new-run-id>

Experiment-mode only: the frozen production harness (lab.m51_pipeline_warm,
lab.m51_e2e.run_one) is not modified; this module reuses the P2 experiment
helpers (run_one_fast/batch_fast/batch_generic) unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

from lab.lab_env import REPO
from lab.source_provenance import capture as capture_source_provenance

SEED = 42


# Project modules that materially affect this experiment's measurements
# (runner, reused helpers, pipeline plumbing, worker, custom node).
EXPERIMENT_MODULES = [
    "lab/m51_pipeline_fair_ab.py",
    "lab/m51_pipeline_profile.py",
    "lab/m51_pipeline.py",
    "lab/m51_e2e.py",
    "lab/comfyui_client.py",
    "lab/server.py",
    "lab/ov_worker.py",
    "lab/lab_env.py",
    "lab/custom_nodes/linuxlab_cond_safetensors.py",
]


def _dependency_probes() -> dict:
    """Fail-safe version probes for the two runtimes that materially affect
    results (OpenVINO worker venv; lab/ComfyUI torch). Never raises."""
    out = {}
    for label, py in (("openvino_venv", REPO / ".venv-ov/bin/python"),
                      ("lab_venv", REPO / ".venv-lab/bin/python")):
        try:
            r = subprocess.run(
                [str(py), "-c",
                 "import sys;d={};\n"
                 "import importlib\n"
                 "for m in ('openvino','torch'):\n"
                 "    try:\n"
                 "        d[m]=importlib.import_module(m).__version__\n"
                 "    except Exception:\n"
                 "        d[m]=None\n"
                 "print(sys.version.split()[0], d.get('openvino'), d.get('torch'))"],
                capture_output=True, text=True, timeout=60)
            if r.returncode == 0:
                pyver, ov, torch = (r.stdout.strip().split() + [None, None, None])[:3]
                out[label] = {"python": pyver, "openvino": ov, "torch": torch}
        except Exception:
            continue
    return out


def batch_fast_native(srv, batch_id: str, outdir: Path) -> dict:
    """A-fast: one persistent RunMonitor for a 4-request GPU-native batch.

    Single variable vs batch_generic(mode=poll, worker=None): the monitor is
    persistent (no per-request RunMonitor teardown in the batch gap). The
    workflows, seeds, image fetch and file write are identical to the
    production A path.
    """
    from lab import comfyui_client, m51_e2e as E
    from lab.m51_pipeline import PROMPTS
    from lab.m51_pipeline_profile import run_one_fast

    mon = comfyui_client.RunMonitor(srv.base_url)
    mon.start()
    reqs = []
    t_all = time.perf_counter()
    try:
        for rid, text in PROMPTS:
            t_loop = time.perf_counter()
            t_wf0 = time.perf_counter()
            wf = E.wf_native(text, "", SEED, 1.0)
            t_wf1 = time.perf_counter()
            rec = run_one_fast(mon, srv, f"{batch_id}|{rid}", wf,
                               outdir / "raw", ev0=len(mon.events))
            rec["wf_build_s"] = round(t_wf1 - t_wf0, 4)
            rec["loop_head_s"] = round(t_loop - (reqs[-1]["t_wall_end"] if reqs else t_all), 4)
            reqs.append(rec)
    finally:
        mon.stop()
    return {"batch": batch_id, "mode": "fast", "batch_wall_s": round(time.perf_counter() - t_all, 3),
            "requests": reqs, "note": "persistent monitor; GPU-native TE (arm A, fast client)"}


def reload_in_batch(batch: dict) -> bool:
    """True if any request re-executed a model loader (>=5 s) — the
    documented CLIP-eviction session-start mechanism landing in a measured
    batch. Detection is from node timings, pre-declared before running."""
    for r in batch["requests"]:
        s = r.get("summary", {}).get("node_timings_s", {})
        if any(k.startswith("CLIPLoader") and v >= 5.0 for k, v in s.items()):
            return True
    return False


def session_A(outdir: Path, port: int, pairs: int) -> dict:
    """Warm GPU-native session: 2 warmups (eviction reload lands here),
    then <pairs> interleaved (poll, fast) measured batches."""
    from lab.server import ComfyServer
    from lab import m51_e2e as E
    from lab.m51_pipeline import server_rss
    from lab.m51_pipeline_profile import batch_generic

    srv = ComfyServer(port=port, log_path=outdir / "server_A.log")
    srv.start()
    out = {"warmups": [], "batches": []}
    try:
        t0 = time.perf_counter()
        E.run_one(srv, "warmup__A_1", E.wf_native("warmup image, unique text", "", 43, 1.0),
                  outdir / "raw")
        w2 = E.run_one(srv, "warmup__A_2", E.wf_native("warmup image, second unique text", "", 44, 1.0),
                       outdir / "raw")
        out["warmups"] = ["warmup__A_1", "warmup__A_2"]
        out["warmup_wall_s"] = round(time.perf_counter() - t0, 3)
        for i in range(pairs):
            for mode in ("poll", "fast"):
                bid = f"A{2 * i + 1}_{mode}"
                row = (batch_generic(srv, bid, "poll", outdir, {})
                       if mode == "poll" else batch_fast_native(srv, bid, outdir))
                row["server_rss_mb"] = server_rss(srv.proc.pid)
                row["model_reload_in_batch"] = reload_in_batch(row)
                out["batches"].append(row)
                print(json.dumps({"batch": row["batch"], "wall": row["batch_wall_s"],
                                  "reqs": [r["attrib"]["wall_s"] for r in row["requests"]],
                                  "reload": row["model_reload_in_batch"]}), flush=True)
    finally:
        srv.stop()
    return out


def session_C(outdir: Path, port: int, pairs: int) -> dict:
    """Warm precomputed session: OV BF16 worker + 1 warmup, then <pairs>
    interleaved (poll, fast) measured batches."""
    from lab.server import ComfyServer
    from lab import m51_e2e as E
    from lab.m51_pipeline import WorkerClient, artifact_root, server_rss
    from lab.m51_pipeline_profile import batch_fast, batch_generic

    art = artifact_root()
    worker = WorkerClient([str(REPO / ".venv-ov/bin/python"), "-u", str(REPO / "lab/ov_worker.py"),
                           "--model-dir", str(art / "bf16"), "--xml", "language_model_bf16.xml",
                           "--precision", "bf16", "--threads", "16"], str(REPO))
    srv = ComfyServer(port=port, log_path=outdir / "server_C.log")
    srv.start()
    out = {"worker_load_s": None, "warmups": [], "batches": []}
    results_stub = {}
    try:
        wready = worker.start()
        out["worker_load_s"] = wready.get("load_s")
        cond_dir = (outdir / "cond" / "C").resolve()
        cond_dir.mkdir(parents=True, exist_ok=True)
        worker.request({"id": "neg", "text": "", "out": str(cond_dir / "negative.safetensors")})
        worker.request({"id": "warmup", "text": "warmup image, unique text",
                        "out": str(cond_dir / "warmup.safetensors")})
        t0 = time.perf_counter()
        E.run_one(srv, "warmup__C", E.wf_precomputed(str(cond_dir / "warmup.safetensors"),
                                                     str(cond_dir / "negative.safetensors"), 43, 1.0),
                  outdir / "raw")
        out["warmups"] = ["negative+cond warmup (worker)", "warmup__C (server)"]
        out["warmup_wall_s"] = round(time.perf_counter() - t0, 3)
        for i in range(pairs):
            for mode in ("poll", "fast"):
                bid = f"C{2 * i + 1}_{mode}"
                row = (batch_fast(srv, bid, outdir, worker, cond_dir) if mode == "fast"
                       else batch_generic(srv, bid, "poll", outdir, results_stub,
                                          worker=worker, cond_dir=cond_dir))
                row["server_rss_mb"] = server_rss(srv.proc.pid)
                row["worker_rss_mb"] = worker.rss_mb()
                row["model_reload_in_batch"] = reload_in_batch(row)
                out["batches"].append(row)
                print(json.dumps({"batch": row["batch"], "wall": row["batch_wall_s"],
                                  "reqs": [r["attrib"]["wall_s"] for r in row["requests"]],
                                  "cond_shas": {r["tag"].split("|")[1]: r.get("cond_sha256", "")[:12]
                                                for r in row["requests"]}}),
                      flush=True)
    finally:
        worker.stop()
        srv.stop()
    return out


def analyze(results: dict) -> dict:
    """Descriptive factorial analysis. Views: primary (all measured batches,
    no exclusions) and warm_resident (derived: batches without an in-batch
    model reload). Both always reported; never silently substituted."""
    def cell_walls(arm, mode, reload_filter=None):
        return [b["batch_wall_s"] for b in results["arms"][arm]["batches"]
                if b.get("mode") == mode and not b.get("failed")
                and (reload_filter is None or b["model_reload_in_batch"] == reload_filter)]

    def stats(v):
        return {"n": len(v), "mean": round(statistics.mean(v), 3) if v else None,
                "median": round(statistics.median(v), 3) if v else None,
                "min": min(v) if v else None, "max": max(v) if v else None}

    def tput(wall):
        return round(4 * 60.0 / wall, 3) if wall else None

    out = {"views": {}}
    for view, rf in (("primary_all_batches", None), ("warm_resident", False)):
        A_poll = cell_walls("A", "poll", rf)
        A_fast = cell_walls("A", "fast", rf)
        C_poll = cell_walls("C", "poll", rf)
        C_fast = cell_walls("C", "fast", rf)
        cell = {
            "A_original": {**stats(A_poll), "img_per_min": tput(statistics.mean(A_poll)) if A_poll else None},
            "A_fast": {**stats(A_fast), "img_per_min": tput(statistics.mean(A_fast)) if A_fast else None},
            "C_original": {**stats(C_poll), "img_per_min": tput(statistics.mean(C_poll)) if C_poll else None},
            "C_fast": {**stats(C_fast), "img_per_min": tput(statistics.mean(C_fast)) if C_fast else None},
        }
        if A_poll and C_poll:
            cell["architecture_gain_equivalent_original_clients_C_over_A"] = round(
                statistics.mean(A_poll) / statistics.mean(C_poll), 4)
        if A_fast and C_fast:
            cell["architecture_gain_equivalent_fast_clients_C_over_A"] = round(
                statistics.mean(A_fast) / statistics.mean(C_fast), 4)
        if A_poll and A_fast:
            cell["monitor_gain_within_A"] = round(
                statistics.mean(A_poll) / statistics.mean(A_fast), 4)
        if C_poll and C_fast:
            cell["monitor_gain_within_C"] = round(
                statistics.mean(C_poll) / statistics.mean(C_fast), 4)
        if all([A_poll, A_fast, C_poll, C_fast]):
            cell["interaction_effect_s"] = round(
                (statistics.mean(A_fast) - statistics.mean(A_poll))
                - (statistics.mean(C_fast) - statistics.mean(C_poll)), 3)
        out["views"][view] = cell
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--port", type=int, default=8213)
    ap.add_argument("--pairs", type=int, default=3,
                    help="measured (poll, fast) pairs per arm per block")
    ap.add_argument("--blocks", type=int, default=2,
                    help="blocks = (A-session + C-session); block 2 reverses order")
    args = ap.parse_args()

    run = Path(args.run)
    outdir = run / "pipeline_profile"
    (outdir / "raw").mkdir(parents=True, exist_ok=True)
    # Source identity is captured BEFORE anything executes, with exact-byte
    # snapshots of any dirty/untracked module (publication-closure P2).
    source_provenance = capture_source_provenance(
        EXPERIMENT_MODULES, REPO, snapshot_dir=outdir / "source_snapshot")
    from lab import m51_e2e as E
    from lab.m51_pipeline import PROMPTS
    E.install_node()

    results = {
        "run": str(run),
        "module": "lab/m51_pipeline_fair_ab.py",
        "preregistration": {
            "question": "2x2 factorial: architecture (GPU-native TE vs OV BF16 CPU TE) x "
                        "client (per-request RunMonitor vs persistent RunMonitor); primary "
                        "endpoint = actual 4-request batch wall (images/min = 240/wall)",
            "design": "warm sessions; interleaved poll,fast batches within each session; "
                      f"{args.blocks} block(s) of (A session + C session), block>=2 reversed",
            "exclusion_rule": "fatal technical failure only (server error / worker crash); "
                              "batch retained marked failed; never performance-based",
            "reload_policy": "2 warmups in A sessions so the documented CLIP-eviction reload "
                             "lands outside measured batches; any in-batch reload detected via "
                             "node timings, reported in both primary and warm-resident views",
            "frozen": "prompts/token IDs/seed/latent/DiT/VAE/scheduler/8 steps/cfg=1/1024px; "
                      "workflows byte-identical to the P2 study (same helpers)",
        },
        "environment": {
            "source_provenance": source_provenance,
            "git": source_provenance["git"],
            "argv": list(sys.argv),
            "dependencies": _dependency_probes(),
            "cgroup_cpu_max": Path("/sys/fs/cgroup/cpu.max").read_text().strip()
            if Path("/sys/fs/cgroup/cpu.max").exists() else None,
            "cgroup_memory_max": Path("/sys/fs/cgroup/memory.max").read_text().strip()
            if Path("/sys/fs/cgroup/memory.max").exists() else None,
            "ov_artifacts_root": str(__import__("lab.m51_pipeline", fromlist=["artifact_root"])
                                     .artifact_root()),
            "bf16_manifest": json.loads(
                (Path(__import__("lab.m51_pipeline", fromlist=["artifact_root"])
                      .artifact_root()) / "bf16" / "conversion_manifest.json").read_text())
        },
        "prompts": [{"id": i, "text": t, "seed": SEED} for i, t in PROMPTS],
        "arms": {},
        "timeline": [],
    }

    t_start = time.time()
    for block in range(1, args.blocks + 1):
        order = ("A", "C") if block % 2 == 1 else ("C", "A")
        for arm in order:
            fn = session_A if arm == "A" else session_C
            t0 = time.time()
            results["arms"].setdefault(arm, {"sessions": []})
            sess = fn(outdir, args.port, args.pairs)
            sess["block"] = block
            sess["started_at"] = time.strftime("%H:%M:%S", time.gmtime(t0))
            results["arms"][arm]["sessions"].append(sess)
            results["timeline"].append({"block": block, "arm": arm,
                                        "started_at": sess["started_at"],
                                        "batches": [b["batch"] for b in sess["batches"]]})
            print(f"[block {block}] session {arm} done "
                  f"({round(time.time() - t0)} s)", flush=True)

    # flatten sessions -> batches for analyze()
    flat = {"A": {"batches": [b for s in results["arms"]["A"]["sessions"] for b in s["batches"]]},
            "C": {"batches": [b for s in results["arms"]["C"]["sessions"] for b in s["batches"]]}}
    results["analysis"] = analyze({"arms": flat})
    # correctness contract: per-prompt conditioning SHA256 identical across all
    # C batches and both client modes (deterministic OV worker)
    per_prompt: dict[str, set[str]] = {}
    for b in flat["C"]["batches"]:
        for r in b["requests"]:
            if r.get("cond_sha256"):
                per_prompt.setdefault(r["tag"].split("|")[1], set()).add(r["cond_sha256"])
    results["cond_sha256_stable_across_all_C_batches"] = {
        k: {"stable": len(v) == 1, "n_distinct": len(v)} for k, v in sorted(per_prompt.items())}
    results["wall_clock_s"] = round(time.time() - t_start, 1)
    (outdir / "pipeline_profile.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results["analysis"], indent=1))
    print(f"WROTE {outdir / 'pipeline_profile.json'}")


if __name__ == "__main__":
    main()
