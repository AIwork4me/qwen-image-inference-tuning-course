"""Minimal ComfyUI HTTP API client for benchmarking (Linux/AMD native).

Pure functions (build_workflow, RunMonitor.summarize) are importable without
requests; HTTP helpers import requests lazily. Workflow nodes are the
standard ComfyUI API-format set; sampler/scheduler/cfg defaults follow the
official Qwen-Image-2.1 template (euler / simple / cfg 1, raw prompts).

Linux-native notes (vs the parent project's Windows variant):
- the VRAM sampler accepts any AMD CUDA device — Linux reports the W7900 as
  "cuda:0 AMD Radeon Graphics : native", which a "7900" name-grep would miss;
- check_server() gates on an AMD CUDA device (anti CPU-fallback).

Derived from AIwork4me/qwen-image-rocm bench/comfyui_client.py (MIT).
"""

import json
import threading
import time
import uuid


def build_workflow(
    *,
    unet: str,
    clip: str,
    vae: str,
    prompt: str,
    negative: str,
    width: int,
    height: int,
    steps: int,
    seed: int,
    sampler: str,
    scheduler: str,
    cfg: float = 1.0,
    weight_dtype: str = "default",
    tiled_vae: bool = False,
) -> dict:
    def nid(prefix: str) -> str:
        return f"{prefix}_{uuid.uuid4().hex[:8]}"

    unet_id, clip_id, vae_id = nid("unet"), nid("clip"), nid("vae")
    pos_id, neg_id = nid("pos"), nid("neg")
    latent_id, ks_id, dec_id, save_id = (
        nid("latent"), nid("ks"), nid("dec"), nid("save"),
    )
    return {
        unet_id: {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": unet, "weight_dtype": weight_dtype},
        },
        clip_id: {
            "class_type": "CLIPLoader",
            "inputs": {"clip_name": clip, "type": "qwen_image"},
        },
        vae_id: {
            "class_type": "VAELoader",
            "inputs": {"vae_name": vae},
        },
        pos_id: {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": prompt, "clip": [clip_id, 0]},
        },
        neg_id: {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": negative, "clip": [clip_id, 0]},
        },
        latent_id: {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": width, "height": height, "batch_size": 1},
        },
        ks_id: {
            "class_type": "KSampler",
            "inputs": {
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": sampler,
                "scheduler": scheduler,
                "denoise": 1.0,
                "model": [unet_id, 0],
                "positive": [pos_id, 0],
                "negative": [neg_id, 0],
                "latent_image": [latent_id, 0],
            },
        },
        dec_id: (
            {
                "class_type": "VAEDecodeTiled",
                "inputs": {
                    "samples": [ks_id, 0], "vae": [vae_id, 0],
                    "tile_size": 512, "overlap": 64,
                    "temporal_size": 64, "temporal_overlap": 8,
                },
            }
            if tiled_vae
            else {
                "class_type": "VAEDecode",
                "inputs": {"samples": [ks_id, 0], "vae": [vae_id, 0]},
            }
        ),
        save_id: {
            "class_type": "SaveImage",
            "inputs": {"filename_prefix": "qir_bench", "images": [dec_id, 0]},
        },
    }


class RunMonitor:
    """Per-node / per-step timing + VRAM sampling for one prompt execution.

    ComfyUI emits over /ws: 'executing' {node, prompt_id} (node=None marks
    the end of the whole prompt), 'progress' once per KSampler step (fired
    at step end), 'execution_cached' {nodes}, plus start/success/error.

    Node duration = time from its 'executing' event to the next 'executing'
    event (nodes run sequentially). Step k duration = progress_k - progress_{k-1};
    step 1 = first progress - KSampler 'executing' start.
    """

    def __init__(self, base_url: str, client_id: str | None = None):
        self.base_url = base_url.rstrip("/")
        self.client_id = client_id or uuid.uuid4().hex
        self.events: list[dict] = []  # {"t": perf_counter, "type": str, "data": dict}
        self.vram_peak_mb: float = 0.0
        self.vram_residency_mb: float | None = None
        self.monitor_error: str | None = None
        self.device_name: str | None = None
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []

    def start(self) -> None:
        self._stop.clear()
        self._threads = [
            threading.Thread(target=self._ws_loop, daemon=True),
            threading.Thread(target=self._vram_loop, daemon=True),
        ]
        for t in self._threads:
            t.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        for t in self._threads:
            t.join(timeout)

    # ---- threads -------------------------------------------------------
    def _ws_loop(self) -> None:
        try:
            import websocket  # websocket-client
        except ImportError:
            self.monitor_error = "websocket-client not installed"
            return
        url = (
            self.base_url.replace("http://", "ws://")
            + f"/ws?clientId={self.client_id}"  # NB: camelCase — the server
            # routes prompt events ONLY to the socket registered under this id
        )
        try:
            ws = websocket.create_connection(url, timeout=5)
        except Exception as exc:  # noqa: BLE001 - degrade gracefully
            self.monitor_error = f"ws connect failed: {exc}"
            return
        try:
            ws.settimeout(1.0)
            while not self._stop.is_set():
                try:
                    op, data = ws.recv_data()
                except websocket.WebSocketTimeoutException:
                    continue
                if op != 1:  # 1 = text frame; 2 = binary preview images
                    continue
                msg = json.loads(data.decode("utf-8"))
                self.events.append(
                    {"t": time.perf_counter(), "type": msg.get("type"),
                     "data": msg.get("data") or {}}
                )
        except Exception as exc:  # noqa: BLE001
            if not self._stop.is_set():
                self.monitor_error = f"ws error: {exc}"
        finally:
            try:
                ws.close()
            except Exception:  # noqa: BLE001
                pass

    def _vram_loop(self) -> None:
        import requests

        while not self._stop.is_set():
            try:
                stats = requests.get(f"{self.base_url}/system_stats", timeout=5).json()
                for d in stats.get("devices", []):
                    if "AMD" in (d.get("name") or ""):  # Linux-native gate
                        self.device_name = d.get("name")
                        used = (d.get("vram_total", 0) - d.get("vram_free", 0)) / 2**20
                        self.vram_peak_mb = max(self.vram_peak_mb, used)
                        self.vram_residency_mb = round(used, 1)
            except Exception:  # noqa: BLE001 - transient poll failures are fine
                pass
            self._stop.wait(0.5)

    # ---- summarization (pure; unit-testable with injected events) -------
    def summarize(self, workflow: dict) -> dict:
        classes = {nid: node["class_type"] for nid, node in workflow.items()}
        node_durations: dict[str, float] = {}
        cached_nodes: list[str] = []
        step_times: list[float] = []
        open_node: str | None = None
        open_t: float | None = None
        last_progress_t: float | None = None
        last_t: float | None = None
        for ev in self.events:
            typ, data, t = ev["type"], ev["data"], ev["t"]
            last_t = t
            if typ == "executing":
                node = data.get("node")
                if open_node is not None and open_t is not None:
                    node_durations[open_node] = t - open_t
                if node is not None:
                    open_node, open_t, last_progress_t = node, t, None
                else:
                    open_node = None
            elif typ == "execution_cached":
                for n in data.get("nodes", []):
                    if n not in cached_nodes:
                        cached_nodes.append(n)
            elif typ == "progress":
                if last_progress_t is None:
                    if open_t is not None:
                        step_times.append(t - open_t)  # step 1: node start -> 1st callback
                else:
                    step_times.append(t - last_progress_t)
                last_progress_t = t
        if open_node is not None and open_t is not None and last_t is not None:
            node_durations[open_node] = last_t - open_t  # unterminated node (error path)

        def by_class(class_names: tuple[str, ...]) -> float:
            return sum(d for n, d in node_durations.items() if classes.get(n) in class_names)

        return {
            "text_encode_s": round(by_class(("CLIPTextEncode",)), 3),
            "vae_decode_s": round(by_class(("VAEDecode", "VAEDecodeTiled")), 3),
            "ksampler_s": round(by_class(("KSampler", "SamplerCustomAdvanced")), 3),
            "step_times_s": [round(s, 3) for s in step_times],
            "node_timings_s": {
                f"{classes.get(n, n)}#{n}": round(d, 3) for n, d in node_durations.items()
            },
            "cached_nodes": [f"{classes.get(n, n)}#{n}" for n in cached_nodes if n in classes],
            "vram_peak_mb": round(self.vram_peak_mb, 1),
            "vram_residency_mb": self.vram_residency_mb,
            "monitor_error": self.monitor_error,
        }


def _get(url: str, timeout: int = 10):
    import requests

    return requests.get(url, timeout=timeout)


def check_server(base_url: str) -> None:
    """Abort unless the server reports an AMD CUDA device (anti CPU-fallback)."""
    stats = _get(f"{base_url}/system_stats").json()
    devices = stats.get("devices", [])
    if not any("AMD" in (d.get("name") or "") for d in devices):
        raise RuntimeError(
            f"no AMD device in /system_stats: {devices} — refusing to benchmark"
        )


def preflight_workflow(base_url: str, workflow: dict) -> None:
    """Verify every node class exists and model names are in server enums."""
    for node_id, node in workflow.items():
        cls = node["class_type"]
        info = _get(f"{base_url}/object_info/{cls}", timeout=15)
        if info.status_code != 200:
            raise RuntimeError(f"node class '{cls}' (id {node_id}) not on server")
        spec = info.json().get(cls, {})
        required = spec.get("input", {}).get("required", {})
        for field, spec_values in required.items():
            if isinstance(spec_values, list) and spec_values and isinstance(spec_values[0], list):
                allowed = spec_values[0]
                value = node["inputs"].get(field)
                if value is not None and value not in allowed:
                    raise RuntimeError(
                        f"node {node_id} ({cls}): '{field}' value '{value}' not in "
                        f"server enum {allowed[:5]}..."
                    )


def post_prompt(base_url: str, workflow: dict, client_id: str | None = None) -> str:
    import requests

    body = {"prompt": workflow}
    if client_id:
        body["client_id"] = client_id
    r = requests.post(f"{base_url}/prompt", json=body, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(
            f"comfyui rejected workflow (HTTP {r.status_code}): {r.text[:500]}"
        )
    body = r.json()
    if "error" in body:
        raise RuntimeError(f"comfyui rejected workflow: {body['error']}")
    return body["prompt_id"]


def wait_for_result(base_url: str, prompt_id: str, timeout_s: int = 1800) -> dict:
    import requests

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        hist = requests.get(f"{base_url}/history/{prompt_id}", timeout=30).json()
        entry = hist.get(prompt_id)
        if entry:
            status = entry.get("status", {})
            if status.get("status_str") == "error":
                msgs = status.get("messages", [])
                raise RuntimeError(f"comfyui execution error: {msgs}")
            return entry
        time.sleep(1.0)
    raise TimeoutError(f"no result for {prompt_id} within {timeout_s}s")


def fetch_image(base_url: str, history: dict) -> bytes:
    import requests

    for node_out in history.get("outputs", {}).values():
        for img in node_out.get("images", []):
            r = requests.get(
                f"{base_url}/view",
                params={
                    "filename": img["filename"],
                    "subfolder": img.get("subfolder", ""),
                    "type": img.get("type", "output"),
                },
                timeout=60,
            )
            r.raise_for_status()
            return r.content
    raise RuntimeError("no image in history outputs")
