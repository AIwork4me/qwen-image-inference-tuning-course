"""Phase-2 profiling: pipeline-level (Layer A) + GPU kernel (Layer B, rocprofv3).

Layer B traces the REAL ComfyUI server process under a bounded rocprofv3
window (-P start:duration:1). Tooling lessons baked in (both learned the
hard way on this stack):
- shutdown MUST be SIGINT to the rocprofv3 wrapper (it finalizes and
  writes CSVs; SIGTERM anywhere leaves only unconverted .dat buffers);
- CSVs are <pid>_<kind>.csv under a hostname subdir.

Traced runs are for COMPOSITION analysis only — interception adds ~1%
overhead, so traced wall times are never performance claims.
"""

from __future__ import annotations

import csv
import json
import re
import statistics
import subprocess
import time
from datetime import datetime
from pathlib import Path

from . import lab_env

VENV_PY = lab_env.REPO / ".venv-lab" / "bin" / "python"


def _client_driver_source() -> str:
    """Profile client: cold (model loads) -> warmup -> sleep-aligned traced gen."""
    repo = lab_env.REPO
    return f'''
import json, sys, time, urllib.request

base = sys.argv[1]
out = {{"events": []}}

def post(path, body=None):
    req = urllib.request.Request(base + path,
                                 data=json.dumps(body).encode() if body else None,
                                 headers={{"Content-Type": "application/json"}})
    return json.loads(urllib.request.urlopen(req, timeout=60).read())

def wait(prompt_id, timeout=1800):
    t0 = time.time()
    while time.time() - t0 < timeout:
        h = json.loads(urllib.request.urlopen(f"{{base}}/history/{{prompt_id}}", timeout=30).read())
        if prompt_id in h:
            return h[prompt_id]
        time.sleep(0.5)
    raise TimeoutError(prompt_id)

def gen(prompt, seed, steps=40, tag=""):
    wf = {{
        "unet": {{"class_type": "UNETLoader",
                 "inputs": {{"unet_name": "qwen_image_2.1_bf16.safetensors", "weight_dtype": "default"}}}},
        "clip": {{"class_type": "CLIPLoader",
                 "inputs": {{"clip_name": "qwen3vl_8b_bf16.safetensors", "type": "qwen_image"}}}},
        "vae": {{"class_type": "VAELoader", "inputs": {{"vae_name": "qwen_image_2.1_vae_bf16.safetensors"}}}},
        "pos": {{"class_type": "CLIPTextEncode", "inputs": {{"text": prompt, "clip": ["clip", 0]}}}},
        "neg": {{"class_type": "CLIPTextEncode", "inputs": {{"text": "", "clip": ["clip", 0]}}}},
        "latent": {{"class_type": "EmptyLatentImage",
                   "inputs": {{"width": 1024, "height": 1024, "batch_size": 1}}}},
        "ks": {{"class_type": "KSampler",
               "inputs": {{"seed": seed, "steps": steps, "cfg": 1.0,
                          "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0,
                          "model": ["unet", 0], "positive": ["pos", 0],
                          "negative": ["neg", 0], "latent_image": ["latent", 0]}}}},
        "dec": {{"class_type": "VAEDecode", "inputs": {{"samples": ["ks", 0], "vae": ["vae", 0]}}}},
        "save": {{"class_type": "SaveImage", "inputs": {{"filename_prefix": "qir_prof", "images": ["dec", 0]}}}},
    }}
    t0 = time.perf_counter()
    pid = post("/prompt", {{"prompt": wf}})["prompt_id"]
    wait(pid)
    dt = time.perf_counter() - t0
    out["events"].append({{"tag": tag, "seed": seed, "steps": steps,
                          "wall_s": round(dt, 3)}})
    print(f"{{tag}}: {{dt:.2f}}s", flush=True)

P1 = "夕阳下的江南水乡小镇，石桥横跨河道，乌篷船缓缓驶过，白墙黛瓦的民居倒映在金色的河面上，远处炊烟袅袅，超高清摄影质感"

mode = sys.argv[3]
WS = float(sys.argv[4])
BOOT = float(sys.argv[5])
t_client0 = time.perf_counter()
gen(P1, 1000042, 40, "cold")
gen(P1, 2000042, 40, "warmup")
target = t_client0 + (WS - BOOT - 1.0)
delay = target - time.perf_counter()
if delay > 0:
    print(f"sleeping {{delay:.1f}}s to align with collection window", flush=True)
    time.sleep(delay)
if mode == "dit":
    gen(P1, 42, 40, "traced-warm")
else:  # "te": fresh prompt so the text encoder runs inside the window
    gen(P1 + " .", 43, 40, "traced-fresh")
json.dump(out, open(sys.argv[2], "w"), indent=1)
'''


def start_server_under_rocprof(out_dir: Path, log: Path, window_start: float,
                               window_len: float, port: int) -> subprocess.Popen:
    cmd = [
        "rocprofv3", "--runtime-trace", "--output-format", "csv",
        "-d", str(out_dir),
        "-P", f"{window_start}:{window_len}:1",
        "--",
        str(VENV_PY), str(lab_env.REPO / "lab" / "comfyui_launch.py"),
        "--listen", "127.0.0.1", "--port", str(port),
        "--extra-model-paths-config", str(lab_env.REPO / "extra_model_paths.yaml"),
    ]
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "w") as lf:
        return subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT,
                                cwd=str(lab_env.COMFY_DIR), start_new_session=True)


def wait_healthy(port: int, timeout: float = 600) -> float:
    import urllib.request

    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout:
        try:
            s = json.loads(urllib.request.urlopen(
                f"http://127.0.0.1:{port}/system_stats", timeout=5).read())
            if any("AMD" in d.get("name", "") for d in s.get("devices", [])):
                return time.perf_counter() - t0
        except Exception:  # noqa: BLE001
            pass
        time.sleep(1.0)
    raise TimeoutError("server under profiler not healthy")


def stop_server_only(rocprof_proc: subprocess.Popen, port: int) -> None:
    """SIGINT the rocprofv3 WRAPPER — verified to finalize and write the CSVs."""
    try:
        rocprof_proc.send_signal(subprocess.signal.SIGINT)
    except ProcessLookupError:
        pass
    try:
        rocprof_proc.wait(timeout=300)
    except subprocess.TimeoutExpired:
        try:
            import os
            import signal

            os.killpg(os.getpgid(rocprof_proc.pid), signal.SIGINT)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            rocprof_proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            rocprof_proc.kill()
            rocprof_proc.wait(timeout=30)
    time.sleep(2.0)


def _classify_kernel(name: str) -> str:
    n = name.lower()
    if ("fmha" in n or "flash" in n or "mha" in n or "attention_fwd" in n
            or "sdpa" in n or "aotriton" in n or "sol_attn" in n or "ck_attn" in n
            or "attn_fwd" in n or "attn_" in n):
        return "attention"
    if ("gemm" in n or "matmul" in n or "s16816" in n or "mfma" in n
            or "xgemm" in n or "igemm" in n or "gemv" in n or "tuned_gemm" in n
            or "hipblaslt" in n or "composable" in n or "wmma" in n
            or n.startswith("cijk_") or "ailk" in n or "alik" in n):
        return "gemm"
    if ("layer_norm" in n or "rms_norm" in n or "group_norm" in n or "rmsnorm" in n
            or "adaln" in n):
        return "norm"
    if ("elementwise" in n or "vectorized" in n or "unrolled" in n
            or "silu" in n or "gelu" in n or "activation" in n or "cat" in n
            or "copy_" in n or "fill_" in n or "add_" in n or "mul" in n
            or "rotate" in n or "rope" in n or "embedding" in n
            or "index" in n or "scatter" in n or "gather" in n or "softplus" in n):
        return "elementwise/activation"
    if "memcpy" in n or "copy" in n:
        return "copy"
    return "other"


def analyze_kernel_trace(kernel_csv: Path, hip_csv: Path | None) -> dict:
    rows = []
    with open(kernel_csv, newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            try:
                rows.append({
                    "name": row["Kernel_Name"],
                    "start": int(row["Start_Timestamp"]),
                    "end": int(row["End_Timestamp"]),
                })
            except (KeyError, ValueError):
                continue
    if not rows:
        return {"error": "no kernel rows"}
    t0 = min(r["start"] for r in rows)
    t1 = max(r["end"] for r in rows)
    span_ns = t1 - t0

    by_class: dict[str, dict] = {}
    for r in rows:
        c = _classify_kernel(r["name"])
        d = r["end"] - r["start"]
        e = by_class.setdefault(c, {"count": 0, "total_ns": 0})
        e["count"] += 1
        e["total_ns"] += d

    # GPU busy time = union of kernel intervals (single GPU)
    intervals = sorted((r["start"], r["end"]) for r in rows)
    busy_ns = 0
    cs, ce = intervals[0]
    for s, e in intervals[1:]:
        if s > ce:
            busy_ns += ce - cs
            cs, ce = s, e
        else:
            ce = max(ce, e)
    busy_ns += ce - cs

    top = {}
    name_totals: dict[str, list[int]] = {}
    for r in rows:
        name_totals.setdefault(r["name"][:160], []).append(r["end"] - r["start"])
    for name, durs in sorted(name_totals.items(), key=lambda kv: -sum(kv[1]))[:15]:
        top[name] = {
            "count": len(durs),
            "total_ms": round(sum(durs) / 1e6, 1),
            "mean_us": round(statistics.mean(durs) / 1e3, 1),
            "class": _classify_kernel(name),
        }

    out = {
        "n_kernels": len(rows),
        "trace_span_s": round(span_ns / 1e9, 3),
        "gpu_busy_s": round(busy_ns / 1e9, 3),
        "gpu_idle_s": round((span_ns - busy_ns) / 1e9, 3),
        "gpu_busy_pct": round(100 * busy_ns / span_ns, 1) if span_ns else None,
        "by_class": {
            c: {"count": v["count"], "total_s": round(v["total_ns"] / 1e9, 4),
                "share_of_busy_pct": round(100 * v["total_ns"] / busy_ns, 1)}
            for c, v in sorted(by_class.items(), key=lambda kv: -kv[1]["total_ns"])
        },
        "top_kernels": top,
    }
    if hip_csv and hip_csv.exists():
        copies = {"H2D": 0, "D2H": 0, "other": 0}
        api_counts: dict[str, int] = {}
        with open(hip_csv, newline="", encoding="utf-8", errors="replace") as f:
            for row in csv.DictReader(f):
                fn = (row.get("Function") or row.get("Function_Name") or "")
                api_counts[fn] = api_counts.get(fn, 0) + 1
                if "Memcpy" in fn:
                    if "HtoD" in fn:
                        copies["H2D"] += 1
                    elif "DtoH" in fn:
                        copies["D2H"] += 1
                    else:
                        copies["other"] += 1
        out["hip_api"] = {
            "top": dict(sorted(api_counts.items(), key=lambda kv: -kv[1])[:12]),
            "memcpy": copies,
            "sync_calls": sum(n for k, n in api_counts.items() if "Synchronize" in k),
        }
    return out


def run(layers: str = "pipeline,kernel", date: str | None = None,
        port: int = 8199) -> int:
    date = date or datetime.now().strftime("%Y%m%d-%H%M%S")
    out_root = lab_env.STUDENT_RUNS_ROOT / f"{date}-profile"
    out_root.mkdir(parents=True, exist_ok=True)
    statuses = {}

    if "pipeline" in layers:
        recs = []
        # The student's OWN baseline (runs/) is preferred and never pooled
        # with the frozen reference campaign (mixing hosts would corrupt
        # the medians — same policy as `lab report` scopes).
        own = ([d for d in sorted(lab_env.STUDENT_RUNS_ROOT.glob("*-B0"))
                if "smoke" not in d.name]
               if lab_env.STUDENT_RUNS_ROOT.is_dir() else [])
        if not own and lab_env.RESULTS_ROOT.is_dir():
            own = [d for d in sorted(lab_env.RESULTS_ROOT.glob("*-B0"))
                   if "smoke" not in d.name]
        for run_dir in own:
            for f in run_dir.glob("*.json"):
                if f.name != "server.log":
                    try:
                        recs.append(json.loads(f.read_text(encoding="utf-8")))
                    except (OSError, ValueError):
                        pass
        if recs:
            warm = [r for r in recs if r["run_kind"] == "measure"]
            fresh = [r for r in recs if r["run_kind"] == "fresh_prompt"]
            cold = [r for r in recs if r["run_kind"] == "cold"]

            def med(rs, getter):
                vals = [getter(r) for r in rs if getter(r) is not None]
                return round(statistics.median(vals), 3) if vals else None

            pipeline = {
                "n_warm": len(warm),
                "T_total_warm_s": med(warm, lambda r: r["runs"][0]["total_s"]),
                "T_dit_denoise_s": med(warm, lambda r: r["ksampler_s"]),
                "T_vae_decode_s": med(warm, lambda r: r["runs"][0]["vae_decode_s"]),
                "T_text_encoder_cached_s": med(warm, lambda r: r["runs"][0]["text_encode_s"]),
                "T_text_encoder_fresh_s": med(fresh, lambda r: r["runs"][0]["text_encode_s"]),
                "T_other_warm_s": None,
                "T_total_fresh_s": med(fresh, lambda r: r["runs"][0]["total_s"]),
                "T_total_cold_s": cold[0]["runs"][0]["total_s"] if cold else None,
                "step_median_s": med(warm, lambda r: r.get("steady_step_median_s")),
                "method": {
                    "T_total_warm_s": "measured (median of warm run records)",
                    "T_dit_denoise_s": "measured (KSampler node timing, websocket events)",
                    "T_vae_decode_s": "measured (VAEDecode node timing)",
                    "T_text_encoder_cached_s": "measured (warm runs: TE node-cache hit)",
                    "T_text_encoder_fresh_s": "measured (fresh-prompt arm)",
                    "T_other_warm_s": "DERIVED by subtraction "
                                      "(T_total - T_dit - T_vae - T_te), never measured",
                    "T_total_cold_s": "measured (single cold run)",
                },
                "note": ("T_other = T_total - T_dit - T_vae - T_te; warm runs have "
                         "TE cached (node-cache hit), fresh runs pay TE"),
            }
            pipeline["T_other_warm_s"] = round(
                pipeline["T_total_warm_s"] - pipeline["T_dit_denoise_s"]
                - pipeline["T_vae_decode_s"] - (pipeline["T_text_encoder_cached_s"] or 0), 3)
            (out_root / "pipeline_breakdown.json").write_text(
                json.dumps(pipeline, indent=2), encoding="utf-8")
            statuses["pipeline"] = "PASS"
            print("pipeline:", json.dumps(pipeline, indent=1))
        else:
            statuses["pipeline"] = "BLOCKED (no B0 records — run baseline first)"
            print(statuses["pipeline"])

    if "kernel" in layers:
        driver = out_root / "_profile_client.py"
        driver.write_text(_client_driver_source(), encoding="utf-8")
        WS, WL = 230.0, 90.0
        for mode in ("dit", "te"):
            kdir = out_root / f"rocprof-{mode}"
            kdir.mkdir(exist_ok=True)
            log = out_root / f"server-{mode}.log"
            prof = start_server_under_rocprof(kdir, log, WS, WL, port)
            try:
                boot = wait_healthy(port)
                print(f"[{mode}] server healthy in {boot:.1f}s "
                      f"(collection window {WS}s+{WL}s)", flush=True)
                drv_out = out_root / f"client-{mode}.json"
                r = subprocess.run(
                    [str(VENV_PY), "-I", str(driver), f"http://127.0.0.1:{port}",
                     str(drv_out), mode, f"{WS:.1f}", f"{boot:.1f}"],
                    capture_output=True, text=True, timeout=1500)
                if r.returncode != 0:
                    raise RuntimeError(f"driver failed: {r.stdout[-1000:]} {r.stderr[-1000:]}")
                (kdir / "client_events.json").write_text(
                    drv_out.read_text(encoding="utf-8"), encoding="utf-8")
            finally:
                stop_server_only(prof, port)
            # CSVs are <pid>_<kind>.csv under a hostname subdir (iterdir, not
            # rglob — rglob proved unreliable on this filesystem)
            kcsv = hcsv = None
            for sub in kdir.iterdir():
                if sub.is_dir():
                    for f in sub.iterdir():
                        if f.name.endswith("_kernel_trace.csv"):
                            kcsv = f
                        elif f.name.endswith("_hip_api_trace.csv"):
                            hcsv = f
            if kcsv is None:
                statuses[f"kernel-{mode}"] = "FAIL (no kernel_trace.csv)"
                continue
            analysis = analyze_kernel_trace(kcsv, hcsv)
            analysis["mode"] = mode
            analysis["note"] = ("traced run — interception overhead included; "
                                "composition only, never a performance claim")
            (kdir / "analysis.json").write_text(json.dumps(analysis, indent=2),
                                                encoding="utf-8")
            statuses[f"kernel-{mode}"] = "PASS"
            print(f"[{mode}] kernels={analysis.get('n_kernels')} "
                  f"busy={analysis.get('gpu_busy_s')}s "
                  f"({analysis.get('gpu_busy_pct')}%) "
                  f"classes={ {k: v['total_s'] for k, v in analysis.get('by_class', {}).items()} }",
                  flush=True)
            port += 1

    status = {"statuses": statuses,
              "output": str(out_root),
              "verdict": "PASS" if all(v == "PASS" for v in statuses.values()) else "PARTIAL"}
    (out_root / "STATUS.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps(status))
    return 0 if status["verdict"] == "PASS" else 1
