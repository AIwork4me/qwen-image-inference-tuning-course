"""Balanced WARM steady-state pipeline experiment: arm A vs arm C (PR2 P1-E).

Fixes the attribution gap of the 2026-11-05 campaign, where the "steady 1.04x"
was an estimate (A steady median x 4 vs C measured total) and A's cold-inclusive
figure mixed in a recurring GPU CLIP reload.

Design:
- Two independent server sessions, both WARM before measurement:
  - Session A: GPU-native TE + DiT + VAE. Warmup request, then 2 batches of
    the 4 frozen requests (sequential, model-resident, no idle).
  - Session C: precomputed-cond DiT + VAE only (warmup itself uses a
    precomputed workflow — no CLIPLoader is EVER instantiated in this
    session); persistent OV BF16-deployment worker encodes each request's
    conditioning overlapped with GPU denoise.
- A third A-batch (A2) at the end controls for session-order drift.
- Reports per-request walls, batch walls, steady medians/dispersion,
  images/min, VRAM peak/residency, server/worker RSS, measured overlap.

Run (physical host, GPU + .venv-ov):
  .venv-lab/bin/python -m lab.m51_pipeline_warm --run runs/<new-run-id>
"""
from __future__ import annotations

import argparse, json, statistics, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != REPO / "lab"]
sys.path.insert(0, str(REPO))

from lab.m51_pipeline import PROMPTS, SEED, WorkerClient, artifact_root, server_rss  # noqa: E402
from lab import m51_e2e as E  # noqa: E402

BATCHES_PER_ARM = 2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--port", type=int, default=8199)
    args = ap.parse_args()
    run = Path(args.run)
    outdir = run / "pipeline_warm"
    (outdir / "raw").mkdir(parents=True, exist_ok=True)
    from lab.server import ComfyServer
    E.install_node()

    def start_server(tag):
        srv = ComfyServer(port=args.port, log_path=outdir / f"server_{tag}.log")
        srv.start()
        return srv

    def batch_A(srv, batch_id):
        reqs = []
        t0 = time.time()
        for i, text in PROMPTS:
            tag = f"A{batch_id}|{i}|cfg1"
            r = E.run_one(srv, tag, E.wf_native(text, "", SEED, 1.0), outdir / "raw")
            reqs.append({"req": i, "wall_s": r["wall_s"], "text_encode_s": r["text_encode_s"],
                         "ksampler_s": r["ksampler_s"], "vae_s": r["vae_decode_s"],
                         "vram_peak_mb": r["vram_peak_mb"], "vram_residency_mb": r["vram_residency_mb"],
                         "t_end": time.time()})
        return {"batch": f"A{batch_id}", "batch_wall_s": round(time.time() - t0, 3),
                "requests": reqs, "server_rss_mb": None}

    results = {"run": str(run), "design": {
        "sessions": "A warm -> 2x4 requests; C warm -> 2x4 requests (precomputed only); "
                    "A2 warm -> 1x4 requests (order-drift control)",
        "warm_definition": "one full generation completed in-session before measurement",
        "steady_definition": "all measured requests run back-to-back, models resident, no idle",
        "prompts": [{"id": i, "text": t, "seed": SEED} for i, t in PROMPTS]},
               "arms": {}}

    # ---------------- Session A ----------------
    srv = start_server("A_warm")
    try:
        t0 = time.time()
        E.run_one(srv, "warmup__A_warm", E.wf_native("warmup image, unique text", "", 43, 1.0), outdir / "raw")
        warm_a = time.time() - t0
        results["arms"]["A"] = {"server_warmup_s": round(warm_a, 3), "batches": []}
        for b in range(1, BATCHES_PER_ARM + 1):
            row = batch_A(srv, b)
            row["server_rss_mb"] = server_rss(srv.proc.pid)
            results["arms"]["A"]["batches"].append(row)
            print(json.dumps({"A batch": b, "wall": row["batch_wall_s"],
                              "reqs": [r["wall_s"] for r in row["requests"]]}), flush=True)
    finally:
        srv.stop()

    # ---------------- Session C (no CLIPLoader ever) ----------------
    art = artifact_root()
    worker = WorkerClient([str(REPO / ".venv-ov/bin/python"), "-u", str(REPO / "lab/ov_worker.py"),
                           "--model-dir", str(art / "bf16"), "--xml", "language_model_bf16.xml",
                           "--precision", "bf16", "--threads", "16"], str(REPO))
    srv = start_server("C_warm")
    try:
        wready = worker.start()
        cond_dir = (outdir / "cond" / "C").resolve()
        cond_dir.mkdir(parents=True, exist_ok=True)
        neg = worker.request({"id": "neg", "text": "", "out": str(cond_dir / "negative.safetensors")})
        # warmup uses the PRECOMPUTED workflow: DiT+VAE only, no GPU CLIP
        wu = worker.request({"id": "warmup", "text": "warmup image, unique text",
                             "out": str(cond_dir / "warmup.safetensors")})
        t0 = time.time()
        E.run_one(srv, "warmup__C_warm", E.wf_precomputed(str(cond_dir / "warmup.safetensors"),
                                                          str(cond_dir / "negative.safetensors"), 43, 1.0),
                  outdir / "raw")
        warm_c = time.time() - t0
        results["arms"]["C"] = {"server_warmup_s": round(warm_c, 3),
                                "worker_load_s": wready.get("load_s"), "worker_meta": wready.get("meta"),
                                "batches": []}
        for b in range(1, BATCHES_PER_ARM + 1):
            reqs = []
            t_all = time.time()
            pool = ThreadPoolExecutor(max_workers=1)
            # pipeline fill: encode request 1 before its generation
            fut = pool.submit(worker.request, {"id": PROMPTS[0][0], "text": PROMPTS[0][1],
                                               "out": str(cond_dir / f"{PROMPTS[0][0]}.safetensors")})
            for idx, (rid, text) in enumerate(PROMPTS):
                cur = fut.result()  # this request's conditioning is ready
                if idx + 1 < len(PROMPTS):
                    nxt_rid, nxt_text = PROMPTS[idx + 1]
                    fut = pool.submit(worker.request, {"id": nxt_rid, "text": nxt_text,
                                                       "out": str(cond_dir / f"{nxt_rid}.safetensors")})
                else:
                    fut = None
                t_gen0 = time.time()
                wf = E.wf_precomputed(str(cond_dir / f"{rid}.safetensors"),
                                      str(cond_dir / "negative.safetensors"), SEED, 1.0)
                r = E.run_one(srv, f"C{b}|{rid}|cfg1", wf, outdir / "raw")
                gen_end = time.time()
                nxt = fut.result() if fut is not None else None  # encoded during this gen
                rec = {"req": rid, "gen_start": t_gen0, "gen_end": gen_end,
                       "gen_wall_s": r["wall_s"], "ksampler_s": r["ksampler_s"], "vae_s": r["vae_decode_s"],
                       "vram_peak_mb": r["vram_peak_mb"], "vram_residency_mb": r["vram_residency_mb"],
                       "cpu_encode_start": cur["t_start"], "cpu_encode_end": cur["t_end"],
                       "cpu_encode_s": round(cur["encode_s"], 4)}
                if nxt is not None:
                    rec["next_overlap_s"] = round(max(0.0, min(gen_end, nxt["t_end"]) -
                                                      max(t_gen0, nxt["t_start"])), 3)
                reqs.append(rec)
            pool.shutdown(wait=False)
            results["arms"]["C"]["batches"].append(
                {"batch": f"C{b}", "batch_wall_s": round(time.time() - t_all, 3), "requests": reqs,
                 "server_rss_mb": server_rss(srv.proc.pid), "worker_rss_mb": worker.rss_mb()})
            print(json.dumps({"C batch": b,
                              "reqs": [r["gen_wall_s"] for r in reqs]}), flush=True)
    finally:
        worker.stop()
        srv.stop()

    # ---------------- Session A2 (order-drift control) ----------------
    srv = start_server("A2_warm")
    try:
        t0 = time.time()
        E.run_one(srv, "warmup__A2_warm", E.wf_native("warmup image, another unique text", "", 44, 1.0),
                  outdir / "raw")
        results["arms"]["A2"] = {"server_warmup_s": round(time.time() - t0, 3), "batches": []}
        row = batch_A(srv, 3)
        row["batch"] = "A2"
        row["server_rss_mb"] = server_rss(srv.proc.pid)
        results["arms"]["A2"]["batches"].append(row)
        print(json.dumps({"A2 batch": 3, "wall": row["batch_wall_s"],
                          "reqs": [r["wall_s"] for r in row["requests"]]}), flush=True)
    finally:
        srv.stop()

    # ---------------- summary ----------------
    def walls(arm, key="wall_s"):
        return [r[key] for b in results["arms"][arm]["batches"] for r in b["requests"]]

    a_w, c_w = walls("A"), walls("C", "gen_wall_s")
    a2_w = walls("A2")
    summary = {
        "A_steady_per_request_s": {"n": len(a_w), "median": round(statistics.median(a_w), 3),
                                   "min": round(min(a_w), 3), "max": round(max(a_w), 3),
                                   "values": [round(x, 3) for x in a_w]},
        "A2_control_per_request_s": {"n": len(a2_w), "median": round(statistics.median(a2_w), 3),
                                     "values": [round(x, 3) for x in a2_w]},
        "C_steady_per_request_s": {"n": len(c_w), "median": round(statistics.median(c_w), 3),
                                   "min": round(min(c_w), 3), "max": round(max(c_w), 3),
                                   "values": [round(x, 3) for x in c_w]},
        "A_vs_C_steady_median_ratio": round(statistics.median(a_w) / statistics.median(c_w), 4),
        "A_vs_C_steady_spread_note": "ratio of medians; single host, n=%d vs %d" % (len(a_w), len(c_w)),
        "A_batch_walls_s": [b["batch_wall_s"] for b in results["arms"]["A"]["batches"]],
        "C_batch_walls_s": [b["batch_wall_s"] for b in results["arms"]["C"]["batches"]],
        "A_images_per_min_steady": round(len(a_w) * 60 / sum(a_w), 2),
        "C_images_per_min_steady": round(len(c_w) * 60 / sum(c_w), 2),
        "C_overlap_total_s": round(sum(r.get("next_overlap_s", 0) for b in results["arms"]["C"]["batches"]
                                       for r in b["requests"]), 3),
        "C_cpu_encode_s": [r.get("cpu_encode_s") for b in results["arms"]["C"]["batches"] for r in b["requests"]],
        "A_vram_peak_mb": max(r["vram_peak_mb"] for b in results["arms"]["A"]["batches"] for r in b["requests"]),
        "C_vram_peak_mb": max(r["vram_peak_mb"] for b in results["arms"]["C"]["batches"] for r in b["requests"]),
    }
    results["summary"] = summary
    (outdir / "pipeline_warm.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(summary, indent=1))
    print(f"WROTE {outdir / 'pipeline_warm.json'}")


def wf2pid(wf):
    return None


if __name__ == "__main__":
    main()
