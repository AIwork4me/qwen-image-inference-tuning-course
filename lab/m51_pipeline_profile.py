"""P2 pipeline-orchestration overhead profiler + event-driven client prototype.

Research question (course-v2 P2): the M5.1 release-closure audit measured warm
per-request service latency 1.081x (A 13.031 s vs C 12.060 s) but actual
four-request batch throughput only 1.029x (4.42 -> 4.55 img/min). Where,
exactly, does the precomputed arm's ~0.54 s/request extra orchestration time
go, and is any of it client-controllable without touching model, precision,
tokenizer, conditioning, seed, latent, DiT, VAE, scheduler, steps or quality?

Design (single W7900 host, same frozen prompts/seed as the M5.1 campaign):
- Session A (warm): GPU-native TE, 4 requests, PRODUCTION completion mode
  (history polling every 1.0 s, exactly lab.m51_e2e.run_one semantics).
- Session C (warm): persistent OV BF16 worker + precomputed-cond workflows;
  interleaved batches alternating completion mode:
      C[poll], C[ws], C[poll], C[ws]
  where "poll" is the production mode and "ws" is the prototype: completion is
  detected from the ComfyUI websocket `executing {node: None}` event (already
  consumed by RunMonitor), then ONE /history GET, then image fetch. Nothing
  else changes — identical workflows, worker, seeds, and image bytes
  (verified by sha256 across modes).
- Per request we record a fine-grained same-process monotonic timeline:
  post -> first-node executing -> node durations -> ws-done -> history
  visible -> image fetched -> file written -> next post.
- Cross-process worker intervals (CPU encode) use wall-clock epochs from the
  worker reply, interpreted (not differenced) against the client timeline,
  following the repo's clock-semantics convention.

Pure helper: attribute_request() is CPU-testable (tests/test_course_quality.py).

Run (physical host):
  .venv-lab/bin/python -m lab.m51_pipeline_profile --run runs/<new-run-id>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != REPO / "lab"]
sys.path.insert(0, str(REPO))

# NOTE: lab.m51_pipeline and lab.m51_e2e pull heavy deps (numpy/safetensors/
# torch) at import time; they are imported lazily inside the batch/main
# functions so that this module (and its pure helper attribute_request)
# stays importable on CPU-only CI.


def attribute_request(rec: dict) -> dict:
    """Derive client-orchestration phases from one recorded request timeline.

    Pure function over the dict written by run_one_profiled(); unit-testable
    without a GPU. Missing fields yield None (never invented numbers).
    """
    def d(a: str, b: str):
        if rec.get(a) is not None and rec.get(b) is not None:
            return round(rec[b] - rec[a], 4)
        return None

    return {
        "post_http_s": d("t_pre_post", "t_post"),
        "queue_to_first_node_s": d("t_post", "t_first_exec"),
        "server_exec_s": d("t_first_exec", "t_ws_done"),
        "poll_wait_after_done_s": d("t_ws_done", "t_hist"),  # poll quantization
        "fetch_image_s": d("t_hist", "t_fetched"),
        "write_file_s": d("t_fetched", "t_written"),
        "client_tail_s": d("t_written", "t_wall_end"),
        "wall_s": d("t_pre_post", "t_wall_end"),
        "history_polls": rec.get("history_polls"),
        "mode": rec.get("mode"),
    }


def run_one_profiled(server, tag: str, wf: dict, raw_dir: Path, mode: str = "poll",
                     timeout: int = 1800) -> dict:
    """Production-equivalent request path with a fine-grained timeline.

    mode "poll": wait_for_result semantics (sleep 1.0 between /history polls).
    mode "ws":   wait on the RunMonitor websocket completion event, then one
                 /history GET (retry briefly if history lags the event).
    The GPU workload, workflow, monitor, image fetch and file write are
    identical between modes — completion detection is the single variable.
    """
    import requests

    from lab import comfyui_client
    mon = comfyui_client.RunMonitor(server.base_url)
    mon.start()
    rec: dict = {"tag": tag, "mode": mode}
    t0 = time.perf_counter()
    rec["t_pre_post"] = t0
    pid = comfyui_client.post_prompt(server.base_url, wf, client_id=mon.client_id)
    rec["t_post"] = time.perf_counter()
    rec["prompt_id"] = pid
    rec["history_polls"] = 0
    if mode == "poll":
        deadline = time.monotonic() + timeout
        hist = None
        while time.monotonic() < deadline:
            rec["history_polls"] += 1
            h = requests.get(f"{server.base_url}/history/{pid}", timeout=30).json()
            entry = h.get(pid)
            if entry:
                status = entry.get("status", {})
                if status.get("status_str") == "error":
                    raise RuntimeError(f"comfyui execution error: {status.get('messages')}")
                hist = entry
                break
            time.sleep(1.0)
        if hist is None:
            raise TimeoutError(f"no result for {pid} within {timeout}s")
    elif mode == "ws":
        deadline = time.monotonic() + timeout
        done_t = None
        while time.monotonic() < deadline:
            for ev in mon.events:
                if (ev["type"] == "executing" and ev["data"].get("node") is None
                        and ev["data"].get("prompt_id") == pid):
                    done_t = ev["t"]
                    break
            if done_t is not None:
                break
            time.sleep(0.01)
        if done_t is None:
            raise TimeoutError(f"no ws completion for {pid} within {timeout}s")
        hist = None
        for _ in range(100):  # history normally appears immediately
            rec["history_polls"] += 1
            h = requests.get(f"{server.base_url}/history/{pid}", timeout=30).json()
            entry = h.get(pid)
            if entry:
                status = entry.get("status", {})
                if status.get("status_str") == "error":
                    raise RuntimeError(f"comfyui execution error: {status.get('messages')}")
                hist = entry
                break
            time.sleep(0.05)
        if hist is None:
            raise TimeoutError(f"history missing after ws completion for {pid}")
    else:
        raise ValueError(f"unknown mode {mode!r}")
    rec["t_hist"] = time.perf_counter()
    data = comfyui_client.fetch_image(server.base_url, hist)
    rec["t_fetched"] = time.perf_counter()
    p = raw_dir / (tag.replace("|", "__") + f"__{mode}.png")
    p.write_bytes(data)
    rec["t_written"] = time.perf_counter()
    rec["t_wall_end"] = rec["t_written"]
    mon.stop()
    s = mon.summarize(wf)
    # ground-truth server events (already timestamped by the monitor)
    for ev in mon.events:
        if ev["type"] == "executing" and ev["data"].get("prompt_id") == pid:
            if ev["data"].get("node") is not None and "t_first_exec" not in rec:
                rec["t_first_exec"] = ev["t"]
            if ev["data"].get("node") is None:
                rec["t_ws_done"] = ev["t"]
    rec["image"] = str(p)
    rec["image_sha256"] = hashlib.sha256(data).hexdigest()
    rec["image_bytes"] = len(data)
    rec["summary"] = s
    rec["attrib"] = attribute_request(rec)
    return rec


def run_one_fast(mon, server, tag: str, wf: dict, raw_dir: Path, ev0: int,
                 timeout: int = 1800) -> dict:
    """Optimized request path (single variable vs production: the RunMonitor
    is PERSISTENT for the whole batch instead of torn down per request).

    Completion detection uses the monitor's own websocket events (sliced from
    ev0 so per-request node timing stays correct); no new connection, no
    per-request thread join. Image fetch and file write are unchanged. The
    GPU workload and workflow bytes are identical to the poll mode.
    """
    import requests

    from lab import comfyui_client
    rec: dict = {"tag": tag, "mode": "fast"}
    rec["t_pre_post"] = time.perf_counter()
    pid = comfyui_client.post_prompt(server.base_url, wf, client_id=mon.client_id)
    rec["t_post"] = time.perf_counter()
    rec["prompt_id"] = pid
    rec["history_polls"] = 0
    deadline = time.monotonic() + timeout
    hist = None
    while time.monotonic() < deadline:
        for ev in mon.events[ev0:]:
            if ev["type"] == "executing" and ev["data"].get("prompt_id") == pid:
                if ev["data"].get("node") is None:
                    rec["t_ws_done"] = ev["t"]
                    break
        if "t_ws_done" in rec:
            break
        time.sleep(0.01)
    if "t_ws_done" not in rec:
        raise TimeoutError(f"no ws completion for {pid} within {timeout}s")
    for _ in range(100):
        rec["history_polls"] += 1
        h = requests.get(f"{server.base_url}/history/{pid}", timeout=30).json()
        entry = h.get(pid)
        if entry:
            status = entry.get("status", {})
            if status.get("status_str") == "error":
                raise RuntimeError(f"comfyui execution error: {status.get('messages')}")
            hist = entry
            break
        time.sleep(0.05)
    if hist is None:
        raise TimeoutError(f"history missing after ws completion for {pid}")
    rec["t_hist"] = time.perf_counter()
    data = comfyui_client.fetch_image(server.base_url, hist)
    rec["t_fetched"] = time.perf_counter()
    p = raw_dir / (tag.replace("|", "__") + "__fast.png")
    p.write_bytes(data)
    rec["t_written"] = time.perf_counter()
    rec["t_wall_end"] = rec["t_written"]
    for ev in mon.events[ev0:]:
        if ev["type"] == "executing" and ev["data"].get("prompt_id") == pid:
            if ev["data"].get("node") is not None and "t_first_exec" not in rec:
                rec["t_first_exec"] = ev["t"]
            if ev["data"].get("node") is None:
                rec["t_ws_done"] = ev["t"]
    rec["image"] = str(p)
    rec["image_sha256"] = hashlib.sha256(data).hexdigest()
    rec["image_bytes"] = len(data)
    m = comfyui_client.RunMonitor.__new__(comfyui_client.RunMonitor)
    m.events = mon.events[ev0:]
    m.vram_peak_mb = mon.vram_peak_mb
    m.vram_residency_mb = mon.vram_residency_mb
    m.monitor_error = mon.monitor_error
    m.device_name = mon.device_name
    rec["summary"] = m.summarize(wf)
    rec["attrib"] = attribute_request(rec)
    return rec


def batch_fast(srv, batch_id: str, outdir: Path, worker, cond_dir: Path) -> dict:
    """Optimized 4-request precomputed batch: one persistent RunMonitor,
    existing next-conditioning prefetch, per-request cond-file SHA256."""
    from lab import comfyui_client, m51_e2e as E
    from lab.m51_pipeline import PROMPTS, SEED
    from concurrent.futures import ThreadPoolExecutor

    mon = comfyui_client.RunMonitor(srv.base_url)
    mon.start()
    pool = ThreadPoolExecutor(max_workers=1)
    reqs = []
    t_all = time.perf_counter()
    try:
        fut = pool.submit(worker.request, {"id": PROMPTS[0][0], "text": PROMPTS[0][1],
                                           "out": str(cond_dir / f"{PROMPTS[0][0]}.safetensors")})
        for idx, (rid, text) in enumerate(PROMPTS):
            t_loop = time.perf_counter()
            cur = fut.result()
            if idx + 1 < len(PROMPTS):
                nxt_rid, nxt_text = PROMPTS[idx + 1]
                fut = pool.submit(worker.request, {"id": nxt_rid, "text": nxt_text,
                                                   "out": str(cond_dir / f"{nxt_rid}.safetensors")})
            else:
                fut = None
            import hashlib as _h
            cond_sha = _h.sha256(Path(cur["out"]).read_bytes()).hexdigest()
            ev0 = len(mon.events)
            t_wf0 = time.perf_counter()
            wf = E.wf_precomputed(str(cond_dir / f"{rid}.safetensors"),
                                  str(cond_dir / "negative.safetensors"), SEED, 1.0)
            t_wf1 = time.perf_counter()
            rec = run_one_fast(mon, srv, f"{batch_id}|{rid}", wf, outdir / "raw", ev0)
            rec["wf_build_s"] = round(t_wf1 - t_wf0, 4)
            rec["loop_head_s"] = round(t_loop - (reqs[-1]["t_wall_end"] if reqs else t_all), 4)
            rec["cpu_encode_s"] = round(cur["encode_s"], 4)
            rec["cond_sha256"] = cond_sha
            reqs.append(rec)
    finally:
        mon.stop()
        pool.shutdown(wait=False)
    return {"batch": batch_id, "mode": "fast", "batch_wall_s": round(time.perf_counter() - t_all, 3),
            "requests": reqs, "note": "persistent monitor; cond prefetch as in production arm C"}


def batch_generic(srv, batch_id: str, mode: str, outdir: Path, results: dict,
                  worker=None, cond_dir: Path | None = None):
    """One 4-request batch. Native (worker=None) or precomputed arm."""
    # Local lazy imports (CPU-only CI import safety): this helper references
    # PROMPTS/SEED and E directly, so they MUST be bound here — regression-tested
    # in tests/test_course_quality.py after the 1d797ad refactor silently
    # unbound them (NameError at first GPU use).
    from lab import m51_e2e as E
    from lab.m51_pipeline import PROMPTS, SEED

    reqs = []
    pool_workers = 1 if worker is not None else 0
    from concurrent.futures import ThreadPoolExecutor
    pool = ThreadPoolExecutor(max_workers=pool_workers) if pool_workers else None
    fut = None
    t_all = time.perf_counter()
    if worker is not None:
        fut = pool.submit(worker.request, {"id": PROMPTS[0][0], "text": PROMPTS[0][1],
                                           "out": str(cond_dir / f"{PROMPTS[0][0]}.safetensors")})
    for idx, (rid, text) in enumerate(PROMPTS):
        t_loop = time.perf_counter()
        if worker is not None:
            cur = fut.result()  # ready (encoded under previous gen)
            if idx + 1 < len(PROMPTS):
                nxt_rid, nxt_text = PROMPTS[idx + 1]
                fut = pool.submit(worker.request, {"id": nxt_rid, "text": nxt_text,
                                                   "out": str(cond_dir / f"{nxt_rid}.safetensors")})
            else:
                fut = None
        t_wf0 = time.perf_counter()
        if worker is None:
            wf = E.wf_native(text, "", SEED, 1.0)
        else:
            wf = E.wf_precomputed(str(cond_dir / f"{rid}.safetensors"),
                                  str(cond_dir / "negative.safetensors"), SEED, 1.0)
        t_wf1 = time.perf_counter()
        rec = run_one_profiled(srv, f"{batch_id}|{rid}", wf, outdir / "raw", mode=mode)
        rec["wf_build_s"] = round(t_wf1 - t_wf0, 4)
        rec["loop_head_s"] = round(t_loop - (reqs[-1]["t_wall_end"] if reqs else t_all), 4) \
            if worker is None else round(t_loop - (reqs[-1]["t_wall_end"] if reqs else t_all), 4)
        if worker is not None:
            rec["cpu_encode_s"] = round(cur["encode_s"], 4)
            rec["cpu_encode_wall"] = {"t_start": cur["t_start"], "t_end": cur["t_end"]}
        reqs.append(rec)
    if pool is not None:
        pool.shutdown(wait=False)
    return {"batch": batch_id, "mode": mode, "batch_wall_s": round(time.perf_counter() - t_all, 3),
            "requests": reqs}


def summarize_attribution(results: dict) -> dict:
    """Compare poll vs ws modes within arm C and attribute overheads."""
    def attribs(arm, mode):
        return [r["attrib"] for b in results["arms"][arm]["batches"] if b["mode"] == mode
                for r in b["requests"]]

    def med(rows, key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return round(statistics.median(vals), 4) if vals else None

    out = {}
    for mode in ("poll", "ws", "fast"):
        rows = attribs("C", mode)
        if not rows:
            continue
        out[f"C_{mode}"] = {
            "n": len(rows),
            "wall_s_median": med(rows, "wall_s"),
            "poll_wait_after_done_s_median": med(rows, "poll_wait_after_done_s"),
            "fetch_image_s_median": med(rows, "fetch_image_s"),
            "write_file_s_median": med(rows, "write_file_s"),
            "queue_to_first_node_s_median": med(rows, "queue_to_first_node_s"),
            "history_polls_median": med(rows, "history_polls"),
        }
    a_rows = [r["attrib"] for b in results["arms"]["A"]["batches"] for r in b["requests"]]
    out["A_poll"] = {"n": len(a_rows), "wall_s_median": med(a_rows, "wall_s"),
                     "poll_wait_after_done_s_median": med(a_rows, "poll_wait_after_done_s"),
                     "queue_to_first_node_s_median": med(a_rows, "queue_to_first_node_s")}
    bw = {b["batch"]: b["batch_wall_s"] for b in results["arms"]["C"]["batches"]}
    for mode in ("poll", "ws", "fast"):
        walls = [v for k, v in bw.items() if k.endswith(mode)]
        if walls:
            out.setdefault("C_batch_wall_s", {})[mode] = {
                "values": walls, "median": round(statistics.median(walls), 3)}
    if "C_batch_wall_s" in out and "poll" in out["C_batch_wall_s"]:
        base = out["C_batch_wall_s"]["poll"]["median"]
        for mode in ("ws", "fast"):
            if mode in out["C_batch_wall_s"]:
                out["C_batch_wall_s"][f"{mode}_vs_poll_ratio"] = round(
                    base / out["C_batch_wall_s"][mode]["median"], 4)
    # monitor-teardown gap attribution (loop_head ~ per-request RunMonitor stop cost)
    for mode in ("poll", "ws", "fast"):
        lh = [r.get("loop_head_s") for b in results["arms"]["C"]["batches"] if b["mode"] == mode
              for r in b["requests"][1:] if r.get("loop_head_s") is not None]
        if lh:
            out.setdefault("C_loop_head_s", {})[mode] = {
                "n": len(lh), "median": round(statistics.median(lh), 4)}
    hashes = {}
    for b in results["arms"]["C"]["batches"]:
        for r in b["requests"]:
            hashes.setdefault(r["tag"].split("|")[1], {})[b["mode"]] = r["image_sha256"]
    out["image_hash_identical_across_modes"] = {
        k: (len(set(v.values())) == 1) for k, v in hashes.items()}
    out["image_hash_note"] = ("ROCm kernels are not bit-deterministic run-to-run; "
                              "identical conditioning + seed => perceptually identical images "
                              "(course noise-floor claim), hashes MAY differ. Correctness "
                              "contract here = conditioning SHA256 + seed + workflow identity.")
    conds = {}
    for b in results["arms"]["C"]["batches"]:
        for r in b["requests"]:
            if "cond_sha256" in r:
                conds.setdefault(r["tag"].split("|")[1], set()).add(r["cond_sha256"])
    if conds:
        out["cond_sha256_stable_within_request"] = {
            k: (len(v) == 1) for k, v in conds.items()}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--port", type=int, default=8203)
    ap.add_argument("--interleave", default="poll,ws,poll,ws",
                    help="comma list over {poll,ws,fast} for the C session batches")
    args = ap.parse_args()
    modes = tuple(m.strip() for m in args.interleave.split(",") if m.strip())
    run = Path(args.run)
    outdir = run / "pipeline_profile"
    (outdir / "raw").mkdir(parents=True, exist_ok=True)
    from lab import m51_e2e as E
    from lab.m51_pipeline import PROMPTS, SEED, WorkerClient, artifact_root, server_rss
    from lab.server import ComfyServer
    E.install_node()

    results = {"run": str(run), "design": {
        "arms": "A warm native (poll); C warm precomputed with interleaved poll/ws batches",
        "single_variable": "completion detection: 1.0s history polling vs ws event + one GET",
        "frozen": "model, precision, tokenizer, conditioning tensors, seed, latent, DiT, VAE, "
                  "scheduler, 8 steps, cfg, prompts — all unchanged; image bytes must match",
        "prompts": [{"id": i, "text": t, "seed": SEED} for i, t in PROMPTS]},
               "arms": {}}

    # ---- Session A (native, production poll mode) ----
    srv = ComfyServer(port=args.port, log_path=outdir / "server_A.log")
    srv.start()
    try:
        E.run_one(srv, "warmup__A", E.wf_native("warmup image, unique text", "", 43, 1.0),
                  outdir / "raw")
        results["arms"]["A"] = {"batches": []}
        row = batch_generic(srv, "A1_poll", "poll", outdir, results)
        row["server_rss_mb"] = server_rss(srv.proc.pid)
        results["arms"]["A"]["batches"].append(row)
        print(json.dumps({"A": row["batch_wall_s"],
                          "reqs": [r["attrib"]["wall_s"] for r in row["requests"]]}), flush=True)
    finally:
        srv.stop()

    # ---- Session C (precomputed; poll/ws interleaved) ----
    art = artifact_root()
    worker = WorkerClient([str(REPO / ".venv-ov/bin/python"), "-u", str(REPO / "lab/ov_worker.py"),
                           "--model-dir", str(art / "bf16"), "--xml", "language_model_bf16.xml",
                           "--precision", "bf16", "--threads", "16"], str(REPO))
    srv = ComfyServer(port=args.port, log_path=outdir / "server_C.log")
    srv.start()
    try:
        wready = worker.start()
        cond_dir = (outdir / "cond" / "C").resolve()
        cond_dir.mkdir(parents=True, exist_ok=True)
        worker.request({"id": "neg", "text": "", "out": str(cond_dir / "negative.safetensors")})
        worker.request({"id": "warmup", "text": "warmup image, unique text",
                        "out": str(cond_dir / "warmup.safetensors")})
        E.run_one(srv, "warmup__C", E.wf_precomputed(str(cond_dir / "warmup.safetensors"),
                                                     str(cond_dir / "negative.safetensors"), 43, 1.0),
                  outdir / "raw")
        results["arms"]["C"] = {"worker_load_s": wready.get("load_s"), "batches": []}
        for i, mode in enumerate(modes):
            if mode == "fast":
                row = batch_fast(srv, f"C{i + 1}_fast", outdir, worker, cond_dir)
            else:
                row = batch_generic(srv, f"C{i + 1}_{mode}", mode, outdir, results,
                                    worker=worker, cond_dir=cond_dir)
            row["server_rss_mb"] = server_rss(srv.proc.pid)
            row["worker_rss_mb"] = worker.rss_mb()
            results["arms"]["C"]["batches"].append(row)
            print(json.dumps({"batch": row["batch"], "wall": row["batch_wall_s"],
                              "reqs": [r["attrib"]["wall_s"] for r in row["requests"]],
                              "loop_head": [r.get("loop_head_s") for r in row["requests"]],
                              "poll_wait": [r["attrib"]["poll_wait_after_done_s"]
                                            for r in row["requests"]]}), flush=True)
    finally:
        worker.stop()
        srv.stop()

    results["attribution"] = summarize_attribution(results)
    (outdir / "pipeline_profile.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results["attribution"], indent=1))
    print(f"WROTE {outdir / 'pipeline_profile.json'}")


if __name__ == "__main__":
    main()
