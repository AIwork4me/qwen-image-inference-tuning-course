"""Four-request heterogeneous pipeline: GPU-TE sequential vs CPU-TE overlap.

Arms:
  A  GPU TE + GPU denoise, sequential
  B  persistent PyTorch-CPU TE overlapped with GPU denoise
  C  persistent OpenVINO BF16 CPU TE overlapped with GPU denoise
  D  persistent OpenVINO INT8 CPU TE overlapped with GPU denoise

Overlap is asserted from measured wall-clock intervals (time.time() in both
processes), not from utilization graphs.
"""
from __future__ import annotations

import argparse, json, subprocess, sys, threading, time, traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]


def artifact_root() -> Path:
    """Artifact-root resolution shared with lab.m51_run (P1-B):
    $M51_OV_ARTIFACTS, else <repo>/runs/m51-ov-artifacts."""
    import os
    return Path(os.environ.get("M51_OV_ARTIFACTS", REPO / "runs" / "m51-ov-artifacts"))
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != REPO / "lab"]
sys.path.insert(0, str(REPO))

PROMPTS = [
    ("req1", "夕阳下的江南水乡小镇，石桥横跨河道，乌篷船缓缓驶过，白墙黛瓦的民居倒映在金色的河面上，远处炊烟袅袅，超高清摄影质感"),
    ("req2", "A lone hiker standing on a mountain ridge at dawn, a sea of clouds below, first light hitting the peaks, ultra-detailed landscape photography"),
    ("req3", "电影海报设计，主标题文字「星际远航」，副标题文字「2026年10月30日 全球上映」，画面是深空中的巨型飞船与绚丽星云，标题文字清晰锐利、排版居中"),
    ("req4", "清晨的竹林小径，薄雾弥漫，阳光透过竹叶洒下斑驳光影，一条小溪蜿蜒流过，超高清摄影质感"),
]
SEED = 42


class WorkerClient:
    def __init__(self, cmd, cwd, env=None, ready_timeout=300):
        self.cmd, self.cwd, self.env = cmd, cwd, env
        self.proc = None; self.ready = None; self._lock = threading.Lock()
    def start(self):
        self.proc = subprocess.Popen(self.cmd, cwd=self.cwd, env=self.env, stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
        with ThreadPoolExecutor(1) as ex:
            fut = ex.submit(self.proc.stdout.readline)
            line = fut.result(timeout=300)
        try:
            self.ready = json.loads(line)
        except Exception:
            err = ""
            try: err = self.proc.stderr.read()
            except Exception: pass
            raise RuntimeError(f"worker failed to start; first stdout={line!r}; stderr={err[-1500:]}")
        if not self.ready.get("ready"):
            raise RuntimeError(f"worker failed to start: {line}")
        return self.ready
    def request(self, payload, timeout=1200):
        with self._lock:
            self.proc.stdin.write(json.dumps(payload) + "\n"); self.proc.stdin.flush()
            with ThreadPoolExecutor(1) as ex:
                line = ex.submit(self.proc.stdout.readline).result(timeout=timeout)
        rec = json.loads(line)
        if not rec.get("ok"):
            raise RuntimeError(f"worker error: {rec.get('error')}\n{rec.get('traceback','')}")
        return rec
    def rss_mb(self):
        try:
            s = Path(f"/proc/{self.proc.pid}/status").read_text()
            return round(int(s.split("VmRSS:")[1].split()[0]) / 1024, 1)
        except Exception:
            return None
    def stop(self):
        if self.proc and self.proc.poll() is None:
            try: self.proc.stdin.write('{"cmd":"quit"}\n'); self.proc.stdin.flush()
            except Exception: pass
            try: self.proc.wait(timeout=15)
            except Exception: self.proc.kill()


def server_rss(pid):
    try:
        s = Path(f"/proc/{pid}/status").read_text()
        return round(int(s.split("VmRSS:")[1].split()[0]) / 1024, 1)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--port", type=int, default=8199)
    ap.add_argument("--arms", default="A,B,C,D")
    args = ap.parse_args(); run = Path(args.run); outdir = run / "pipeline"; outdir.mkdir(parents=True, exist_ok=True)
    from lab.server import ComfyServer
    from lab import m51_e2e as E
    E.install_node()
    results = {"run": str(run), "prompts": [{"id": i, "text": t, "seed": SEED} for i, t in PROMPTS],
               "arms": {}, "definitions": {
                   "A": "GPU native TE + GPU denoise, sequential",
                   "B": "persistent PyTorch CPU BF16-storage/FP32-manual-cast TE overlapped with GPU denoise",
                   "C": "persistent OpenVINO BF16 CPU TE overlapped with GPU denoise",
                   "D": "persistent OpenVINO INT8 CPU TE overlapped with GPU denoise"}}

    def start_server(tag):
        srv = ComfyServer(port=args.port, log_path=outdir / f"server_{tag}.log"); srv.start(); return srv

    def warm(srv):
        t0 = time.time(); E.run_one(srv, "warmup__pipeline", E.wf_native("warmup image, unique text", "", 43, 1.0), outdir / "raw")
        return time.time() - t0

    def run_A():
        (outdir / "raw").mkdir(parents=True, exist_ok=True)
        srv = start_server("A")
        try:
            rss0 = server_rss(srv.proc.pid)
            warmup = warm(srv)
            reqs = []; t_all = time.time()
            for i, text in PROMPTS:
                tag = f"A|{i}|cfg1"
                r = E.run_one(srv, tag, E.wf_native(text, "", SEED, 1.0), outdir / "raw")
                reqs.append({"req": i, "wall_s": r["wall_s"], "text_encode_s": r["text_encode_s"],
                             "ksampler_s": r["ksampler_s"], "vae_s": r["vae_decode_s"],
                             "vram_peak_mb": r["vram_peak_mb"],
                             "vram_residency_mb": r["vram_residency_mb"], "t_end": time.time()})
            total = time.time() - t_all
            return {"server_warmup_s": round(warmup, 3), "requests": reqs, "total_s": round(total, 3),
                    "images_per_min": round(4 * 60 / total, 2), "server_rss_after_mb": server_rss(srv.proc.pid),
                    "server_rss_before_mb": rss0,
                    "vram_residency_mb": reqs[-1].get("vram_residency_mb"),
                    "vram_peak_mb": max(r["vram_peak_mb"] for r in reqs)}
        finally:
            srv.stop()

    def run_pipeline(name, worker_cmd, worker_cwd, worker_env=None):
        srv = start_server(name); worker = WorkerClient(worker_cmd, worker_cwd, worker_env)
        try:
            wready = worker.start()
            warmup = warm(srv)
            raw = outdir / "raw"; raw.mkdir(exist_ok=True)
            cond_dir = (outdir / "cond" / name).resolve(); cond_dir.mkdir(parents=True, exist_ok=True)
            neg = worker.request({"id": "neg", "text": "", "out": str(cond_dir / "negative.safetensors")})
            # pipeline fill: first prompt
            fill0 = time.time()
            c0 = worker.request({"id": "req1", "text": PROMPTS[0][1], "out": str(cond_dir / "req1.safetensors")})
            fill_s = time.time() - fill0
            reqs = []; pool = ThreadPoolExecutor(max_workers=1); fut = None
            t_all = time.time()
            for idx, (rid, text) in enumerate(PROMPTS):
                cond_path = str(cond_dir / f"{rid}.safetensors")
                wf = E.wf_precomputed(cond_path, str(cond_dir / "negative.safetensors"), SEED, 1.0)
                t_gen0 = time.time()
                pid = __import__("lab.comfyui_client", fromlist=["post_prompt"]).post_prompt(srv.base_url, wf)
                if idx + 1 < len(PROMPTS):
                    nid, ntext = PROMPTS[idx + 1]
                    fut = pool.submit(worker.request, {"id": nid, "text": ntext,
                                                       "out": str(cond_dir / f"{nid}.safetensors")})
                hist = __import__("lab.comfyui_client", fromlist=["wait_for_result"]).wait_for_result(srv.base_url, pid)
                gen_end = time.time()
                nxt = fut.result() if fut is not None else None
                fut = None
                rec = {"req": rid, "gen_start": t_gen0, "gen_end": gen_end,
                       "gen_wall_s": round(gen_end - t_gen0, 3)}
                if nxt is not None:
                    rec.update({"cpu_encode_start": nxt["t_start"], "cpu_encode_end": nxt["t_end"],
                                "cpu_encode_s": round(nxt["encode_s"], 4),
                                "overlap_s": round(max(0.0, min(gen_end, nxt["t_end"]) - max(t_gen0, nxt["t_start"])), 3),
                                "wait_after_gen_s": round(max(0.0, nxt["t_end"] - gen_end), 3)})
                reqs.append(rec)
            total = time.time() - t_all
            overlaps = [r.get("overlap_s", 0) for r in reqs]
            return {"worker_ready": wready, "server_warmup_s": round(warmup, 3), "fill_s": round(fill_s, 3),
                    "requests": reqs, "total_s": round(total + fill_s, 3),
                    "steady_total_s": round(total, 3),
                    "images_per_min": round(4 * 60 / (total + fill_s), 2),
                    "overlap_s_total": round(sum(overlaps), 3), "overlap_proven": sum(overlaps) > 1.0,
                    "worker_rss_after_mb": worker.rss_mb(), "server_rss_after_mb": server_rss(srv.proc.pid),
                    "worker_load_s": wready.get("load_s"), "worker_meta": wready.get("meta")}
        finally:
            worker.stop(); srv.stop()

    existing = outdir / "pipeline_ab.json"
    if existing.exists():
        try:
            old = json.loads(existing.read_text())
            results["arms"] = old.get("arms", {})
        except Exception:
            pass
    arms = args.arms.split(",")
    if "A" in arms:
        print("=== Arm A ===", flush=True); results["arms"]["A"] = run_A(); print(json.dumps(results["arms"]["A"], indent=1), flush=True)
    if "B" in arms:
        print("=== Arm B ===", flush=True)
        env = dict(__import__("os").environ)
        results["arms"]["B"] = run_pipeline("B", [str(REPO/".venv-lab/bin/python"), "-u", str(REPO/"lab/pt_worker.py"), "--threads", "16"], str(REPO), env)
        print(json.dumps(results["arms"]["B"], indent=1), flush=True)
    if "C" in arms:
        print("=== Arm C ===", flush=True)
        env = dict(__import__("os").environ)
        art = artifact_root()
        results["arms"]["C"] = run_pipeline("C", [str(REPO/".venv-ov/bin/python"), "-u", str(REPO/"lab/ov_worker.py"), "--model-dir", str(art/"bf16"), "--xml", "language_model_bf16.xml", "--precision", "bf16", "--threads", "16"], str(REPO), env)
        print(json.dumps(results["arms"]["C"], indent=1), flush=True)
    if "D" in arms:
        print("=== Arm D ===", flush=True)
        env = dict(__import__("os").environ)
        art = artifact_root()
        results["arms"]["D"] = run_pipeline("D", [str(REPO/".venv-ov/bin/python"), "-u", str(REPO/"lab/ov_worker.py"), "--model-dir", str(art/"int8"), "--xml", "language_model_int8.xml", "--precision", "f32", "--threads", "16", "--dynamic-quant-group", "32"], str(REPO), env)
        print(json.dumps(results["arms"]["D"], indent=1), flush=True)
    (outdir / "pipeline_ab.json").write_text(json.dumps(results, indent=2))
    print(f"WROTE {outdir/'pipeline_ab.json'}")

if __name__ == "__main__":
    main()
