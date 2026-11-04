"""ComfyUI server lifecycle: detached launch, health gate, log evidence,
memory free between experiments, teardown with a port-free check."""

from __future__ import annotations

import os
import re
import signal
import subprocess
import time
from pathlib import Path

from . import lab_env

REPO = lab_env.REPO
COMFY_DIR = lab_env.COMFY_DIR
VENV_PY = REPO / ".venv-lab" / "bin" / "python"

DEFAULT_ATTENTION_PATTERNS = [
    r"Using pytorch attention",
    r"Using Comfy Kitchen attention",
    r"Comfy Kitchen attention is unavailable",
    r"Using .* attention",
    r"attention",
    r"quadratic",
    r"disabling",
    r"Set: torch\.backends\.cudnn",
    r"AMD arch",
    r"ROCm version",
]


class ComfyServer:
    def __init__(self, port: int = 8188, log_path: Path | None = None,
                 server_flags: list[str] | None = None, env_vars: dict | None = None,
                 sdp_mode: str = "none"):
        self.port = port
        self.base_url = f"http://127.0.0.1:{port}"
        self.log_path = log_path or (lab_env.RESULTS_ROOT / "server.log")
        self.server_flags = server_flags or []
        self.env_vars = env_vars or {}
        self.sdp_mode = sdp_mode  # "none" = normal functional attention
        self.proc: subprocess.Popen | None = None
        self.boot_s: float | None = None
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def start(self, health_timeout_s: int = 600) -> "ComfyServer":
        env = dict(os.environ)
        env.pop("QIR_SDP", None)
        if self.sdp_mode in ("math", "math-hard"):
            env["QIR_SDP"] = self.sdp_mode
        env.update(self.env_vars)
        cmd = [
            str(VENV_PY), str(REPO / "lab" / "comfyui_launch.py"),
            "--listen", "127.0.0.1", "--port", str(self.port),
            "--extra-model-paths-config", str(REPO / "extra_model_paths.yaml"),
            *self.server_flags,
        ]
        with open(self.log_path, "w") as logf:
            self.proc = subprocess.Popen(
                cmd, stdout=logf, stderr=subprocess.STDOUT, cwd=COMFY_DIR,
                env=env, start_new_session=True,
            )
        t0 = time.perf_counter()
        import requests

        deadline = t0 + health_timeout_s
        last_err = None
        while time.perf_counter() < deadline:
            if self.proc.poll() is not None:
                tail = self._log_tail(30)
                raise RuntimeError(
                    f"ComfyUI died during boot (rc={self.proc.returncode}):\n{tail}")
            try:
                lab_env.check_server_linux(self.base_url)
                self.boot_s = time.perf_counter() - t0
                return self
            except Exception as exc:  # noqa: BLE001
                last_err = exc
            time.sleep(1.0)
        raise TimeoutError(f"server not healthy within {health_timeout_s}s: {last_err}")

    def _log_tail(self, n: int) -> str:
        try:
            return "\n".join(self.log_path.read_text(errors="replace").splitlines()[-n:])
        except OSError:
            return "<no log>"

    def scrape_attention_evidence(self, patterns: list[str] | None = None) -> dict:
        """Which attention path did the server ACTUALLY take? (log evidence)"""
        patterns = patterns or DEFAULT_ATTENTION_PATTERNS
        try:
            text = self.log_path.read_text(errors="replace")
        except OSError:
            return {"error": "no server log"}
        hits = {}
        for pat in patterns:
            found = re.findall(f"^.*{pat}.*$", text, re.M | re.I)
            if found:
                hits[pat] = found[-3:]
        return hits

    def free_memory(self) -> None:
        import requests

        try:
            requests.post(f"{self.base_url}/free", json={"unload_models": True},
                          timeout=60)
        except Exception:  # noqa: BLE001
            pass

    def stop(self, timeout_s: int = 30) -> None:
        if self.proc is None:
            return
        try:
            os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            self.proc.terminate()
        try:
            self.proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                self.proc.kill()
            self.proc.wait(timeout=10)
        time.sleep(1.0)
        free = subprocess.run(
            ["bash", "-lc", f"ss -tln | grep -c ':{self.port} ' || true"],
            capture_output=True, text=True)
        if free.stdout.strip() not in ("0", ""):
            raise RuntimeError(f"port {self.port} still bound after server stop")

    def __enter__(self) -> "ComfyServer":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()
