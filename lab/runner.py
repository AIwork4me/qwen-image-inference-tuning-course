"""Experiment runner: one manifest end-to-end with the frozen protocol.

Fresh server per experiment (restart-identical discipline) -> health gate ->
cold run -> per-prompt warmup + 3 measured runs (+ fresh-prompt arm when
enabled) -> schema-valid records + PNGs -> attention evidence from the
server log -> clean teardown.
"""

from __future__ import annotations

import hashlib
import json
import statistics
import threading
import time
from datetime import datetime
from pathlib import Path

from . import comfyui_client, lab_env, manifests
from .core import PROMPTS_PATH, plan_runs, steady_step_rate, validate_record
from .server import ComfyServer


class RunMonitor(comfyui_client.RunMonitor):
    """RunMonitor + device-level rocm-smi sampling.

    /system_stats gives the torch view (same instrument as the parent
    project's records); rocm-smi adds the device view (captures non-torch
    allocations) and GPU utilization. Both are periodic samples — recorded
    peaks are LOWER BOUNDS on transients, never exact peaks.
    """

    def __init__(self, base_url: str):
        super().__init__(base_url)
        self.device_vram_peak_mb: float = 0.0
        self.gpu_util_samples: list[float] = []

    def start(self) -> None:
        self._stop.clear()
        self._threads = [
            threading.Thread(target=self._ws_loop, daemon=True),
            threading.Thread(target=self._vram_loop, daemon=True),
            threading.Thread(target=self._rocm_loop, daemon=True),
        ]
        for t in self._threads:
            t.start()

    def _rocm_loop(self) -> None:
        import subprocess

        def csv_rows(args: list[str]) -> list[list[str]]:
            r = subprocess.run(args, capture_output=True, text=True, timeout=5)
            lines = [l for l in r.stdout.splitlines() if l.strip()]
            return [l.split(",") for l in lines if "," in l]

        while not self._stop.is_set():
            try:
                rows = csv_rows(["rocm-smi", "-u", "--csv", "--loglevel", "error"])
                # header: device,GPU use (%)
                for row in rows[1:]:
                    if len(row) >= 2:
                        try:
                            self.gpu_util_samples.append(float(row[1]))
                        except ValueError:
                            pass
                rows = csv_rows(["rocm-smi", "--showmeminfo", "vram", "--csv",
                                 "--loglevel", "error"])
                # Column layouts differ across ROCm releases (3 columns on
                # 7.2.1: device,total,used; 4 on the reference host) — locate
                # the used-bytes column by its HEADER instead of position.
                if rows:
                    header = [h.strip() for h in rows[0]]
                    try:
                        used_idx = header.index("VRAM Total Used Memory (B)")
                    except ValueError:
                        used_idx = len(header) - 1
                    for row in rows[1:]:
                        if len(row) > used_idx and row[used_idx].strip().isdigit():
                            self.device_vram_peak_mb = max(
                                self.device_vram_peak_mb,
                                int(row[used_idx]) / 2**20)
            except Exception:  # noqa: BLE001
                pass
            self._stop.wait(1.0)


def _host_mem() -> dict:
    out = {}
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                k, v = line.split(":", 1)
                out[k] = int(v.strip().split()[0]) / 2**20  # MiB
    except OSError:
        pass
    return {
        "host_mem_available_mib": round(out.get("MemAvailable", 0), 1),
        "host_swap_used_mib": round(out.get("SwapTotal", 0) - out.get("SwapFree", 0), 1),
    }


def _execute(base_url: str, wf: dict) -> tuple[float, float, dict, dict]:
    mon = RunMonitor(base_url)
    mon.start()
    try:
        t0 = time.perf_counter()
        prompt_id = comfyui_client.post_prompt(base_url, wf, client_id=mon.client_id)
        history = comfyui_client.wait_for_result(base_url, prompt_id)
        total_s = time.perf_counter() - t0
    finally:
        mon.stop()
    ts = {
        msg[0]: msg[1]["timestamp"]
        for msg in history.get("status", {}).get("messages", [])
        if len(msg) == 2 and isinstance(msg[1], dict) and "timestamp" in msg[1]
    }
    if "execution_start" in ts and "execution_success" in ts:
        execution_time_s = (ts["execution_success"] - ts["execution_start"]) / 1000.0
    else:
        execution_time_s = total_s
    summary = mon.summarize(wf)
    summary["vram_device_peak_mb"] = round(mon.device_vram_peak_mb, 1)
    if mon.gpu_util_samples:
        summary["gpu_util_pct_p50"] = round(statistics.median(mon.gpu_util_samples), 1)
        summary["gpu_util_pct_max"] = round(max(mon.gpu_util_samples), 1)
    return total_s, execution_time_s, summary, history


def build_record(manifest: dict, spec: dict, *, total_s: float, execution_time_s: float,
                 summary: dict, env_ref: str, run_id: str, server_boot_s: float | None,
                 attention_evidence: dict) -> dict:
    run_kind = spec["kind"]
    steady = steady_step_rate(summary["step_times_s"])
    record = {
        "experiment": manifest["id"],
        "phase": {"baseline": 1, "less-computation": 3, "faster-computation": 4,
                  "combined": 6, "heterogeneous": 5, "throughput": 5,
                  "final": 6}.get(manifest["category"], 0),
        "framework": "comfyui",
        "precision": manifest["precision"],
        "width": spec["width"],
        "height": spec["height"],
        "steps": manifest["steps"],
        "seed": spec["seed"],
        "prompt_id": spec["prompt_id"],
        "protocol_version": "course-lab-v1 (protocol v1.3 timing semantics)",
        "runs": [{
            "kind": "cold" if run_kind == "cold" else (
                "fresh" if run_kind == "fresh_prompt" else "warm"),
            "total_s": round(total_s, 3),
            "text_encode_s": summary["text_encode_s"],
            "denoise_s": round(execution_time_s, 3),
            "vae_decode_s": summary["vae_decode_s"],
            "step_times_s": summary["step_times_s"],
        }],
        "vram_peak_mb": summary["vram_peak_mb"],
        "env_ref": env_ref,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "run_kind": run_kind,
        "attention_backend": manifest["attention_backend"],
        "execution_time_s": round(execution_time_s, 3),
        "sampler": manifest["sampler"],
        "scheduler": manifest["scheduler"],
        "cfg": manifest["cfg"],
        "ksampler_s": summary["ksampler_s"],
        "node_timings_s": summary["node_timings_s"],
        "cached_nodes": summary["cached_nodes"],
        "vram_residency_mb": summary["vram_residency_mb"],
        "unet": manifest["unet"],
        "clip": manifest["clip"],
        "vae": manifest["vae"],
        "weight_dtype": manifest["weight_dtype"],
        "tiled_vae": manifest["tiled_vae"],
        "session_id": run_id,
        "restart_idx": 0,
        # ---- lab extensions ----
        "lab": "qwen-image-inference-tuning-course",
        "run_id": run_id,
        "category": manifest["category"],
        "steady_step_median_s": round(steady, 3) if steady else None,
        "server_boot_s": round(server_boot_s, 2) if server_boot_s else None,
        "attention_evidence": attention_evidence,
        "vram_device_peak_mb": summary.get("vram_device_peak_mb"),
        "gpu_util_pct_p50": summary.get("gpu_util_pct_p50"),
        "gpu_util_pct_max": summary.get("gpu_util_pct_max"),
        "host_mem": _host_mem(),
    }
    if summary.get("monitor_error"):
        record["monitor_error"] = summary["monitor_error"]
    return record


def _required_weight_files(manifest: dict) -> list[str]:
    return [f"diffusion_models/{manifest['unet']}",
            f"text_encoders/{manifest['clip']}",
            f"vae/{manifest['vae']}"]


def _precheck_weights(manifest: dict, *, verify: bool = False) -> None:
    """Fail fast (before any server boot) when the platform cannot supply
    the models this experiment needs. Staged-by-design: only the files the
    manifest references are required, so a student can smoke-test BEST
    before obtaining every variant."""
    from . import weights as weights_mod

    missing, corrupt = [], []
    for rel in _required_weight_files(manifest):
        path = weights_mod.WEIGHTS_ROOT / rel
        if not path.exists():
            missing.append(rel)
        elif path.stat().st_size != weights_mod.WEIGHTS[rel][0]:
            corrupt.append(rel)
        elif verify:
            r = weights_mod.verify_weight(rel)
            if not r["ok"]:
                corrupt.append(rel)
    if missing or corrupt:
        base = weights_mod.DEFAULT_MIRROR_BASE
        lines = [f"experiment {manifest['id']} needs these pinned model files:"]
        for rel in _required_weight_files(manifest):
            size_gb = weights_mod.WEIGHTS[rel][0] / 1e9
            flag = "MISSING" if rel in missing else (
                "CORRUPT (size mismatch)" if rel in corrupt else "present")
            lines.append(f"  [{flag}] {rel} ({size_gb:.2f} GB)")
        lines.append(f"download (resumable, 16-way ranged):")
        for rel in missing + corrupt:
            lines.append(f"  python -I lab/download_weights.py "
                         f"{base}/resolve/main/{rel} weights/comfyui-split/{rel} 16")
        raise SystemExit("\n".join(lines))


def run_experiment(experiment_id: str, *, date: str | None = None, port: int = 8188,
                   only_prompt: str | None = None, warmup: int | None = None,
                   measured: int | None = None, smoke: bool = False) -> Path:
    manifest = manifests.load_manifest(experiment_id)
    _precheck_weights(manifest)
    date = date or datetime.now().strftime("%Y%m%d-%H%M%S")
    run_id = f"{date}-{manifest['id']}" + ("-smoke" if smoke else "")
    results_dir = lab_env.STUDENT_RUNS_ROOT / run_id
    if results_dir.exists() and any(results_dir.glob("*.json")):
        raise RuntimeError(f"results dir already has records: {results_dir} "
                           "(evidence is never overwritten; pick a new --date)")
    results_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = lab_env.STUDENT_RUNS_ROOT / "raw" / run_id

    env_ref = lab_env.write_snapshot()

    server = ComfyServer(
        port=port,
        log_path=results_dir / "server.log",
        server_flags=manifest["server_flags"],
        env_vars=manifest["env_vars"],
        sdp_mode=manifest["sdp_mode"],
    )
    print(f"[{manifest['id']}] starting fresh server (flags={manifest['server_flags']}, "
          f"sdp={manifest['sdp_mode']}) ...", flush=True)
    server.start()
    try:
        lab_env.check_server_linux(server.base_url)
        prompts = json.loads(PROMPTS_PATH.read_text(encoding="utf-8"))
        prompt_text = {p["id"]: (p["prompt"], p["negative_prompt"]) for p in prompts["prompts"]}
        if only_prompt:
            prompt_text = {only_prompt: prompt_text[only_prompt]}
        res = manifest["resolution"]
        specs = plan_runs(list(prompt_text), [(res, res)], prompts["seeds"],
                          warmup=warmup if warmup is not None else (0 if smoke else 1),
                          measured=measured if measured is not None else
                          (1 if smoke else manifest["measured"]))
        if manifest.get("fresh_prompt_arm") and not smoke:
            for prompt_id in prompt_text:
                specs.append({"kind": "fresh_prompt", "prompt_id": prompt_id,
                              "width": res, "height": res,
                              "seed": prompts["seeds"][0], "run_idx": 0})

        def wf_for(prompt_id, seed, text_override=None):
            text, negative = prompt_text[prompt_id]
            if text_override is not None:
                text = text_override
            if manifest.get("scheduler") == "manual_sigmas":
                from .workflows import build_workflow_manual_sigmas

                return build_workflow_manual_sigmas(
                    unet=manifest["unet"], clip=manifest["clip"], vae=manifest["vae"],
                    prompt=text, negative=negative, width=res, height=res,
                    seed=seed, sigmas=manifest["sigmas"],
                    sampler=manifest["sampler"], cfg=manifest["cfg"])
            return comfyui_client.build_workflow(
                unet=manifest["unet"], clip=manifest["clip"], vae=manifest["vae"],
                prompt=text, negative=negative, width=res, height=res,
                steps=manifest["steps"], seed=seed,
                sampler=manifest["sampler"], scheduler=manifest["scheduler"],
                cfg=manifest["cfg"], weight_dtype=manifest["weight_dtype"],
                tiled_vae=manifest["tiled_vae"])

        first_prompt = next(iter(prompt_text))
        comfyui_client.preflight_workflow(server.base_url, wf_for(first_prompt, seed=0))

        for spec in specs:
            text_override = None
            if spec["kind"] == "fresh_prompt":
                text_override = prompt_text[spec["prompt_id"]][0] + " ."
            wf = wf_for(spec["prompt_id"], spec["seed"], text_override=text_override)
            total_s, execution_time_s, summary, history = _execute(server.base_url, wf)
            png = comfyui_client.fetch_image(server.base_url, history)
            tag = spec["kind"] if spec["kind"] in ("cold", "warmup", "fresh_prompt") \
                else f"r{spec['run_idx']}"
            out_png = raw_dir / f"{spec['prompt_id']}-{spec['seed']}-{tag}.png"
            out_png.parent.mkdir(parents=True, exist_ok=True)
            out_png.write_bytes(png)

            record = build_record(
                manifest, spec, total_s=total_s, execution_time_s=execution_time_s,
                summary=summary, env_ref=env_ref, run_id=run_id,
                server_boot_s=server.boot_s if spec["kind"] == "cold" else None,
                attention_evidence=server.scrape_attention_evidence(
                    manifest["attention_log_patterns"]))
            record["image_sha256"] = hashlib.sha256(png).hexdigest()
            record["image_path"] = str(out_png.relative_to(lab_env.REPO))
            validate_record(record)
            out_json = results_dir / f"{spec['prompt_id']}-{spec['seed']}-{tag}.json"
            out_json.write_text(json.dumps(record, indent=2), encoding="utf-8")
            r0 = record["runs"][0]
            print(f"  {spec['kind']:13s} {spec['prompt_id']} seed={spec['seed']} "
                  f"total={r0['total_s']}s ks={record['ksampler_s']}s "
                  f"te={r0['text_encode_s']}s vae={r0['vae_decode_s']}s "
                  f"steps={len(r0['step_times_s'])} "
                  f"vrampeak={record['vram_peak_mb']}MB dev={record['vram_device_peak_mb']}MB",
                  flush=True)
    finally:
        server.stop()
    print(f"[{manifest['id']}] results: {results_dir}", flush=True)
    return results_dir
