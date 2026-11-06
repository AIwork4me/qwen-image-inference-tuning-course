"""GPU-first notebook runtime: three declared execution modes for the labs.

    REFERENCE       offline CPU-only analysis of the frozen committed evidence
                    (GitHub CPU CI profile; never launches GPU work)
    GPU_CLASSROOM   bounded physical W7900 exercises (preferred student mode)
    GPU_RESEARCH    full-protocol physical A/B campaigns (staff/advanced)

Design contract (course PR #4, GPU-first upgrade):
- The student edits exactly one variable in the first cell:
      COURSE_MODE = "GPU_CLASSROOM"
  and calls ``rt.begin(COURSE_MODE, notebook="01")``.  Nothing else in the
  notebook source changes between modes, languages, or CI.
- ``QIR_COURSE_MODE`` overrides the requested mode (CI pins REFERENCE;
  the qualification harness pins GPU_CLASSROOM / GPU_RESEARCH).  An unknown
  mode is an error, never a silent fallback.
- GPU work always runs through the existing ``python -m lab`` CLI in a
  subprocess (fresh server per experiment, append-only records under
  ``runs/``, schema-validated, evidence/ untouched), guarded by:
    * hardware admission (real AMD gfx1100-class GPU, VRAM headroom,
      pinned weights for the requested experiment),
    * a filesystem lock so only ONE GPU workload runs at a time,
    * per-profile budgets (prompts/warmup/measured/timeout).
- A smoke/classroom execution proves the pipeline WORKS and gives a bounded
  physical comparison; it is labelled SMOKE_ONLY / CLASSROOM_BOUNDED and is
  never presented as a full-protocol performance claim.
"""
from __future__ import annotations

import json
import os
import re
import secrets
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from . import lab_env
from . import student_provenance as prov

MODES = ("REFERENCE", "GPU_CLASSROOM", "GPU_RESEARCH")
_MODE_RE = re.compile(r"^[A-Z_]+$")

# Bounded classroom profiles per experiment: (only_prompt, warmup, measured, timeout_s)
# Full research protocol comes from the manifest (3 prompts x 3 seeds x measured=3).
CLASSROOM_PROFILES: dict[str, dict[str, Any]] = {
    "B0":  {"only_prompt": "p1-zh-scene", "warmup": 1, "measured": 2, "timeout_s": 1500},
    "L1":  {"only_prompt": "p1-zh-scene", "warmup": 1, "measured": 2, "timeout_s": 900},
    "F1":  {"only_prompt": "p1-zh-scene", "warmup": 1, "measured": 2, "timeout_s": 1500},
    "F2":  {"only_prompt": "p1-zh-scene", "warmup": 1, "measured": 2, "timeout_s": 1500},
    "BEST": {"only_prompt": "p1-zh-scene", "warmup": 1, "measured": 2, "timeout_s": 900},
}
RESEARCH_TIMEOUT_S = 9000  # full protocol incl. fresh-prompt arm and server boot

GPU_BUSY_USED_MB_LIMIT = 8192  # refuse to start when the device already holds >8 GB


class CourseModeError(RuntimeError):
    """Raised for an invalid/unsafe mode request (never a silent fallback)."""


class AdmissionError(RuntimeError):
    """GPU admission failed; message is student-readable."""


def resolve_mode(requested: str | None = None) -> str:
    """Normalize the mode. Precedence: env override > requested > REFERENCE."""
    env = os.environ.get("QIR_COURSE_MODE")
    mode = (env or requested or "REFERENCE").strip()
    if mode not in MODES:
        raise CourseModeError(
            f"Unknown COURSE_MODE {mode!r}. Valid modes: {', '.join(MODES)}. "
            "Set COURSE_MODE = 'GPU_CLASSROOM' in the first cell, or "
            "QIR_COURSE_MODE=REFERENCE for CPU-only analysis.")
    return mode


# ----------------------------------------------------------------------------
# run-record loading (shared by every mode; pure file parsing, no torch)
# ----------------------------------------------------------------------------

@dataclass
class RunBundle:
    """Parsed records of one physical run directory (runs/<run_id>/)."""
    run_id: str
    root: Path
    records: list[dict] = field(default_factory=list)

    @property
    def experiment(self) -> str:
        return self.records[0]["experiment"] if self.records else self.run_id.rstrip("-smoke").split("-")[-1]

    def _runs(self, kind: str) -> list[dict]:
        out = []
        for rec in self.records:
            if rec.get("run_kind") != "measure":
                continue
            for r in rec.get("runs", []):
                if r["kind"] == kind:
                    out.append((rec, r))
        return out

    def warm(self) -> list[dict]:
        return [r for _, r in self._runs("warm")]

    def warm_totals(self) -> list[float]:
        return [float(r["total_s"]) for r in self.warm()]

    def cold_total(self) -> float | None:
        c = self._runs("cold")
        return float(c[0][1]["total_s"]) if c else None

    def is_smoke(self) -> bool:
        return self.run_id.endswith("-smoke")

    def stats(self) -> dict:
        vals = self.warm_totals()
        if not vals:
            raise ValueError(f"no warm measured records in {self.run_id}")
        med = statistics.median(vals)
        spread = (max(vals) - min(vals)) if len(vals) > 1 else 0.0
        n_prompts = len({rec.get("prompt_id") for rec in self.records
                         if rec.get("run_kind") == "measure"})
        if self.is_smoke():
            population = "smoke (bounded single-shot)"
        elif n_prompts >= 3 and len(vals) >= 9:
            population = "formal protocol (multi-prompt, multi-seed)"
        else:
            population = f"classroom-bounded (single-prompt, n={len(vals)})"
        return {
            "run_id": self.run_id,
            "experiment": self.experiment,
            "population": population,
            "n_warm": len(vals),
            "median_s": round(med, 3),
            "mean_s": round(statistics.mean(vals), 3),
            "min_s": round(min(vals), 3),
            "max_s": round(max(vals), 3),
            "range_s": round(spread, 3),
            "steps": self.records[0].get("steps") if self.records else None,
            "precision": self.records[0].get("precision") if self.records else None,
            "attention_backend": (self.records[0].get("attention_backend") if self.records else None),
            "unet": self.records[0].get("unet") if self.records else None,
            "vram_peak_mb": max((r.get("vram_peak_mb") or 0) for r in self.records) if self.records else None,
            "vram_device_peak_mb": max((r.get("vram_device_peak_mb") or 0) for r in self.records) if self.records else None,
            "per_prompt": _per_prompt(self.records),
            "warm_totals": [round(v, 3) for v in vals],
        }

    def step_stats(self) -> dict:
        steps = [len(r["step_times_s"]) for rec in self.records
                 for r in rec.get("runs", []) if r.get("kind") == "warm"]
        med_step = [rec.get("steady_step_median_s") for rec in self.records
                    if rec.get("steady_step_median_s")]
        out = {"step_counts": sorted(set(steps))}
        if med_step:
            out["steady_step_median_s"] = round(statistics.median(med_step), 3)
        return out


def _per_prompt(records: list[dict]) -> dict[str, list[float]]:
    by: dict[str, list[float]] = {}
    for rec in records:
        if rec.get("run_kind") != "measure":
            continue
        for r in rec.get("runs", []):
            if r["kind"] == "warm":
                by.setdefault(rec.get("prompt_id", "?"), []).append(round(float(r["total_s"]), 3))
    return by


def load_run(run_id: str, root: Path | None = None) -> RunBundle:
    root = (root or lab_env.REPO).resolve()
    d = root / "runs" / run_id
    if not d.is_dir():
        raise FileNotFoundError(
            f"run directory not found: {d}. Physical experiments land under "
            "runs/<run-id>/ (list them with rt.list_runs())")
    recs = []
    for f in sorted(d.glob("*.json")):
        if f.name == "server.log":
            continue
        try:
            recs.append(json.loads(f.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
    if not recs:
        raise ValueError(f"no valid records in {d}")
    return RunBundle(run_id=run_id, root=root, records=recs)


def list_runs(experiment: str | None = None, root: Path | None = None) -> list[str]:
    root = (root or lab_env.REPO).resolve()
    pat = f"*-{experiment}" if experiment else "*"
    return sorted(p.name for p in (root / "runs").glob(pat) if p.is_dir())


# ----------------------------------------------------------------------------
# A/B comparison (protocol-honest)
# ----------------------------------------------------------------------------

def ab_compare(base: RunBundle, cand: RunBundle, *, label: str | None = None) -> dict:
    """Compare two physical run bundles.

    The returned classification distinguishes bounded exercises from formal
    protocol comparisons; a speedup from smoke populations is labelled as
    such and never presented as a formal performance claim.
    """
    bs, cs = base.stats(), cand.stats()
    paired = []
    for pid, vals in bs["per_prompt"].items():
        cvals = cs["per_prompt"].get(pid)
        if cvals:
            paired.append({"prompt_id": pid,
                           "base_median_s": round(statistics.median(vals), 3),
                           "cand_median_s": round(statistics.median(cvals), 3),
                           "paired_ratio": round(statistics.median(vals) /
                                                 statistics.median(cvals), 3)})
    speedup = round(bs["median_s"] / cs["median_s"], 3) if cs["median_s"] else None
    formal = (not base.is_smoke() and not cand.is_smoke()
              and bs["n_warm"] >= 9 and cs["n_warm"] >= 9)
    classification = ("FORMAL_PROTOCOL_MEASURED" if formal else
                      "BOUNDED_PHYSICAL_COMPARISON")
    out = {
        "classification": classification,
        "label": label or f"{bs['experiment']} vs {cs['experiment']}",
        "baseline": bs, "candidate": cs,
        "speedup_median": speedup,
        "latency_delta_s": round(cs["median_s"] - bs["median_s"], 3),
        "vram_delta_mb": round((cs.get("vram_device_peak_mb") or cs.get("vram_peak_mb") or 0)
                               - (bs.get("vram_device_peak_mb") or bs.get("vram_peak_mb") or 0), 1),
        "paired_per_prompt": paired,
        "caveats": [
            "speedup = baseline median / candidate median of the populations above",
            "smoke/classroom populations are bounded; do not report them as "
            "full-protocol statistics",
        ],
    }
    return out


def plot_ab(comparison: dict, *, metric: str = "latency") -> Any:
    """Latency or VRAM bar chart FROM MEASURED RECORDS (guarded import)."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - kernel envs without mpl
        print(f"plot skipped (matplotlib unavailable: {exc})")
        return None
    b, c = comparison["baseline"], comparison["candidate"]
    if metric == "vram":
        vals = [b.get("vram_device_peak_mb") or b.get("vram_peak_mb") or 0,
                c.get("vram_device_peak_mb") or c.get("vram_peak_mb") or 0]
        ylab = "device VRAM peak (MB)"
    else:
        vals = [b["median_s"], c["median_s"]]
        ylab = "warm median latency (s)"
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    names = [f"{b['experiment']}\n(n={b['n_warm']})", f"{c['experiment']}\n(n={c['n_warm']})"]
    bars = ax.bar(names, vals, color=["#4477aa", "#ee6677"])
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{v:.1f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel(ylab)
    ax.set_title(f"{comparison['label']}  [{comparison['classification']}]",
                 fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------
# admission
# ----------------------------------------------------------------------------

@dataclass
class Admission:
    status: str  # PASS | PASS_WITH_WARNINGS | BLOCKED
    mode: str
    detail: dict
    warnings: list[str]

    def ok(self) -> bool:
        return self.status != "BLOCKED"


def admit_hardware(mode: str) -> Admission:
    """Physical W7900 admission. Never imports torch in REFERENCE mode."""
    detail: dict[str, Any] = {}
    warnings: list[str] = []
    if mode == "REFERENCE":
        return Admission("PASS", mode, {"note": "offline reference mode; no GPU required"}, [])
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        raise AdmissionError(
            f"GPU mode {mode} requested but PyTorch is unavailable in this "
            f"kernel ({exc}). Use COURSE_MODE='REFERENCE' on CPU-only hosts, "
            "or launch the notebook with the course .venv-lab kernel "
            "(docs/RADEON_CLOUD_QUICKSTART.md).") from exc
    if not torch.cuda.is_available():
        raise AdmissionError(
            "torch.cuda.is_available() is False: no usable ROCm/HIP device. "
            "On Radeon Cloud re-run the admission cell after the W7900 is "
            "attached; on other hosts use COURSE_MODE='REFERENCE'.")
    props = torch.cuda.get_device_properties(0)
    name = torch.cuda.get_device_name(0)
    arch = getattr(props, "gcnArchName", "") or ""
    free_b, total_b = torch.cuda.mem_get_info()
    used_b = total_b - free_b
    detail.update({
        "device_name": name,
        "gfx_arch": arch.split(":")[0],
        "vram_total_mib": round(total_b / 2**20),
        "vram_used_mib": round(used_b / 2**20),
        "torch": torch.__version__, "hip": torch.version.hip,
        "device_count": torch.cuda.device_count(),
    })
    blocking: list[str] = []
    if "AMD" not in name:
        blocking.append(f"device is not AMD: {name!r}")
    if not str(arch).startswith("gfx1100"):
        warnings.append(
            f"gfx arch {arch or '?'} is not gfx1100 (W7900-class): results are "
            "a valid independent measurement but NOT comparable to the "
            "committed reference campaign")
    if torch.cuda.device_count() != 1:
        blocking.append(f"expected exactly 1 GPU, torch sees {torch.cuda.device_count()}")
    if used_b / 2**20 > GPU_BUSY_USED_MB_LIMIT:
        blocking.append(
            f"GPU already holds {used_b/2**30:.1f} GiB (another workload is "
            "running). Finish it first — one GPU workload at a time.")
    if blocking:
        raise AdmissionError("GPU admission BLOCKED: " + "; ".join(blocking))
    status = "PASS_WITH_WARNINGS" if warnings else "PASS"
    return Admission(status, mode, detail, warnings)


# ----------------------------------------------------------------------------
# session
# ----------------------------------------------------------------------------

def _campaign_dir() -> Path | None:
    v = os.environ.get("QIR_NOTEBOOK_CAMPAIGN")
    return (lab_env.REPO / v).resolve() if v else None


def _sessions_root() -> Path:
    c = _campaign_dir()
    return (c / "sessions") if c else (lab_env.REPO / "runs" / "notebook-sessions")


def _gpu_lock_path() -> Path:
    return lab_env.STUDENT_RUNS_ROOT / ".notebook_gpu.lock"


def _acquire_gpu_lock(timeout_s: float = 600.0) -> int:
    """Serialize GPU workloads (one at a time across kernels/notebooks)."""
    lock = _gpu_lock_path()
    lock.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if lock.exists():
            try:
                other = int(lock.read_text().strip() or 0)
                os.kill(other, 0)
                time.sleep(5)
                continue
            except (ProcessLookupError, ValueError, PermissionError):
                pass  # stale lock from a dead process
        lock.write_text(str(os.getpid()))
        try:
            got = int(lock.read_text().strip() or 0)
        except ValueError:
            got = -1
        if got == os.getpid():
            return os.getpid()
        time.sleep(2)
    raise AdmissionError(
        f"another GPU workload has held {_gpu_lock_path()} for >{timeout_s:.0f}s. "
        "Wait for it to finish (one GPU workload at a time), then re-run the cell.")


def _release_gpu_lock() -> None:
    lock = _gpu_lock_path()
    try:
        if lock.read_text().strip() == str(os.getpid()):
            lock.unlink()
    except (OSError, ValueError):
        pass


def _diagnose(exc: Exception, cmd: list[str], log_hint: str) -> str:
    """Map the most common physical failures to student-readable guidance."""
    tail = ""
    if isinstance(exc, subprocess.TimeoutExpired):
        return (f"TIMEOUT while running {' '.join(cmd[:6])}... . The bounded "
                "budget was exceeded. Safe recovery: re-run this cell (a new "
                "run tag is appended; nothing is overwritten). If it keeps "
                "timing out, check `rocm-smi` for a stuck workload and "
                "docs/TROUBLESHOOTING.md.")
    out = (getattr(exc, "stdout", b"") or b"")
    err = (getattr(exc, "stderr", b"") or b"")
    if isinstance(out, bytes):
        out = out.decode("utf-8", "replace")
    if isinstance(err, bytes):
        err = err.decode("utf-8", "replace")
    tail = (out + err)[-1500:]
    if "needs these pinned model files" in tail or "MISSING" in tail:
        return ("Required model weights are missing. Stage them with the "
                "pinned downloader, e.g.\n"
                "  bash scripts/cloud/start.sh --all-models\n"
                "or the per-file command printed by the lab CLI. Then re-run "
                f"this cell.\n--- CLI output tail ---\n{tail}")
    if "results dir already has records" in tail:
        return ("Run tag already used (records are append-only). Re-run the "
                "cell; the runtime generates a fresh unique tag each time.")
    if "Address already in use" in tail or "port" in tail.lower():
        return ("The ComfyUI port is busy. Another server may be running; "
                "wait a minute and re-run, or check `ss -tlnp | grep 8188`. "
                f"--- CLI output tail ---\n{tail}")
    if "no AMD device" in tail:
        return ("The ComfyUI server did not expose an AMD GPU (CPU-fallback "
                "guard). Check `rocm-smi` and re-run after the GPU is free.\n"
                f"--- CLI output tail ---\n{tail}")
    return (f"experiment CLI failed (exit {getattr(exc, 'returncode', '?')}). "
            f"Server log: {log_hint}\n--- CLI output tail ---\n{tail}")


class Session:
    """One notebook's declared session: mode, admission, executions, records.

    Every session owns a unique ``session_id`` created before any GPU work.
    Physical runs executed here are bound to it via PROVENANCE.json; runs
    loaded from disk are classified (CURRENT_SESSION_PHYSICAL / ... /
    COMMITTED_PHYSICAL_REFERENCE) and can never silently masquerade as the
    student's own new measurement."""

    def __init__(self, mode: str, notebook: str):
        self.mode = mode
        self.notebook = notebook
        self.session_id = prov.new_session_id()
        self.source = prov.source_identity()
        self.started_at = datetime.now().astimezone().isoformat(timespec="seconds")
        self.admission: Admission | None = None
        self.executions: list[dict] = []
        self.consumed: list[dict] = []
        self.gulock_pid: int | None = None
        self.gpulock_pid: int | None = None

    # -- provenance ---------------------------------------------------------
    def _fresh_tag(self, suffix: str) -> str:
        """Minute-resolution default tag, de-conflicted with a short random
        suffix so a re-run never collides with append-only records."""
        base = datetime.now().strftime("%H%M%S")
        tag = f"{base}-nbq"
        while any((lab_env.STUDENT_RUNS_ROOT / f"{tag}{s}").exists()
                  for s in ((f"-{suffix}",) if suffix else ("",))):
            tag = f"{base}-{secrets.token_hex(2)}-nbq"
        return tag

    def _bind(self, run_id: str, kind: str) -> None:
        try:
            prov.write_binding(run_id, self.session_id, kind=kind,
                               extra={"mode": self.mode, "notebook": self.notebook})
        except prov.ProvenanceError as exc:  # pragma: no cover - defensive
            print(f"[PROVENANCE] WARNING: could not bind {run_id}: {exc}")

    def origin_of(self, run_id: str) -> str:
        """Stamp text for a run this session executed or consumed."""
        for e in self.executions:
            if e.get("run_id") == run_id:
                return prov.stamp_text("CURRENT_SESSION_PHYSICAL")
        for c in self.consumed:
            if c.get("run_id") == run_id and c.get("category"):
                return prov.stamp_text(c["category"])
        return prov.stamp_text(classify_run(run_id, session_id=self.session_id)["category"])

    # -- banner -----------------------------------------------------------
    def banner(self) -> str:
        a = self.admission
        if self.mode == "REFERENCE":
            mid = "MODE=REFERENCE | frozen committed evidence | no GPU work launched"
        elif a is None:
            mid = f"MODE={self.mode} | admission not yet run"
        else:
            d = a.detail
            mid = (f"MODE={self.mode} | GPU={d.get('device_name')} "
                   f"{d.get('gfx_arch')} | VRAM {d.get('vram_used_mib')}/"
                   f"{d.get('vram_total_mib')} MiB | admission={a.status}")
        line = "=" * 74
        return f"{line}\n {mid}\n{line}"

    def status(self) -> None:
        print(self.banner())
        if self.admission and self.admission.warnings:
            for w in self.admission.warnings:
                print(f" WARNING: {w}")
        if self.mode == "REFERENCE":
            print(" Historical numbers are references to investigate, NOT your "
                  "measurements. No physical performance claim is made here.")

    # -- admission --------------------------------------------------------
    def admit(self) -> Admission:
        self.admission = admit_hardware(self.mode)
        self.status()
        return self.admission

    # -- experiment execution ----------------------------------------------
    def run(self, experiment: str, *, profile: str = "classroom",
            tag: str | None = None) -> RunBundle:
        """Execute a physical experiment through the pinned lab CLI.

        profile='classroom': bounded (1 prompt, warmup+2 measured) — a
        teaching exercise, not a formal benchmark.
        profile='research':  the manifest's full protocol (GPU_RESEARCH only).
        profile='smoke':     the repo's classic single-shot smoke.
        """
        if self.mode == "REFERENCE":
            raise CourseModeError(
                "Physical execution is disabled in REFERENCE mode. Set "
                "COURSE_MODE='GPU_CLASSROOM' (bounded) or 'GPU_RESEARCH' "
                "(full protocol) in the first cell, then Restart & Run All.")
        if self.admission is None:
            self.admit()
        assert self.admission is not None
        if not self.admission.ok():
            raise AdmissionError("admission BLOCKED; fix the reported issue first")
        if profile == "research" and self.mode != "GPU_RESEARCH":
            raise CourseModeError(
                "profile='research' (full multi-prompt multi-seed protocol) is "
                "restricted to GPU_RESEARCH mode; classroom runs stay bounded")
        if experiment in CLASSROOM_PROFILES and profile == "classroom":
            prof = CLASSROOM_PROFILES[experiment]
            args = ["--only-prompt", prof["only_prompt"],
                    "--warmup", str(prof["warmup"]),
                    "--measured", str(prof["measured"])]
            timeout = prof["timeout_s"]
            kind = "classroom-bounded"
        elif profile == "smoke":
            args = ["--smoke"]
            timeout = 1200
            kind = "smoke"
        elif profile == "research":
            args = []
            timeout = RESEARCH_TIMEOUT_S
            kind = "research-full-protocol"
        else:
            raise CourseModeError(
                f"no bounded classroom profile for experiment {experiment!r}; "
                "use profile='smoke' or 'research'")
        date = tag or self._fresh_tag(experiment)
        run_id = f"{date}-{experiment}" + (("-smoke") if kind == "smoke" else "")
        cmd = [sys.executable, "-m", "lab", "run", "--experiment", experiment,
               "--date", date, *args]
        log_path = lab_env.STUDENT_RUNS_ROOT / run_id / "server.log"
        print(f"[{kind}] physical W7900 execution:\n  {' '.join(cmd)}")
        self.gpulock_pid = _acquire_gpu_lock()
        t0 = time.time()
        try:
            r = subprocess.run(cmd, cwd=str(lab_env.REPO), timeout=timeout,
                               capture_output=True, text=True)
        except subprocess.TimeoutExpired as exc:
            raise AdmissionError(_diagnose(exc, cmd, str(log_path))) from exc
        finally:
            _release_gpu_lock()
            self.gpulock_pid = None
        if r.returncode != 0:
            fake = subprocess.CalledProcessError(r.returncode, cmd, r.stdout, r.stderr)
            raise AdmissionError(_diagnose(fake, cmd, str(log_path)))
        bundle = load_run(run_id)
        self._bind(run_id, kind=kind)
        st = bundle.stats()
        print(f"[OK] run_id={run_id} | n_warm={st['n_warm']} | "
              f"median={st['median_s']}s | steps={st['steps']} | "
              f"vram_device_peak={st['vram_device_peak_mb']}MB")
        print(f"     records: {lab_env.REPO / 'runs' / run_id}")
        print(f"[PROVENANCE] {prov.label(run_id, 'CURRENT_SESSION_PHYSICAL')}")
        self.executions.append({
            "run_id": run_id, "experiment": experiment, "kind": kind,
            "cmd": cmd, "wall_s": round(time.time() - t0, 1),
            "stats": st, "started": datetime.now().astimezone().isoformat(timespec="seconds"),
            "session_id": self.session_id,
            "run_origin": "CURRENT_SESSION_PHYSICAL",
            "source_head": self.source.get("head"),
            "source_dirty": self.source.get("dirty"),
        })
        return bundle

    # -- consuming existing physical evidence --------------------------------
    def consume(self, run_id: str, *, why: str = "",
                expect: str | tuple[str, ...] | None = None) -> RunBundle:
        """Load an existing physical record instead of executing a new one.

        The record's provenance category is RESOLVED AND VALIDATED here (fail
        closed): an unbound, untracked, non-campaign run cannot be consumed,
        and a committed instructor run is always labelled
        COMMITTED_PHYSICAL_REFERENCE — never as a new measurement by this
        student. ``expect`` pins the category explicitly."""
        if expect is not None:
            info = prov.require_category(run_id, expect, session_id=self.session_id)
        else:
            info = prov.classify_run(run_id, session_id=self.session_id)
        bundle = load_run(run_id)
        st = bundle.stats()
        cat = info["category"]
        print(f"[CONSUMED PHYSICAL RECORD] run_id={run_id} "
              f"({st['population']}, n_warm={st['n_warm']}) {why}")
        print(f"[PROVENANCE] {prov.label(run_id, cat)}")
        self.consumed.append({"run_id": run_id, "experiment": st["experiment"],
                              "why": why, "stats": st, "category": cat})
        return bundle

    # -- module-specific bounded physical helpers -----------------------------
    def _run_cli(self, cmd: list[str], *, timeout: float, kind: str,
                 lock: bool = True, bind: str | None = None) -> subprocess.CompletedProcess:
        """Shared bounded subprocess runner (GPU lock + diagnosis).

        ``bind`` names the run directory (under runs/) that this CLI creates;
        after success it receives a PROVENANCE.json binding to this session."""
        if self.mode == "REFERENCE":
            raise CourseModeError(f"{kind} is physical GPU work; disabled in REFERENCE mode")
        if self.admission is None:
            self.admit()
        assert self.admission is not None and self.admission.ok()
        print(f"[{kind}] {' '.join(cmd)}")
        if lock:
            self.gpulock_pid = _acquire_gpu_lock()
        t0 = time.time()
        try:
            r = subprocess.run(cmd, cwd=str(lab_env.REPO), timeout=timeout,
                               capture_output=True, text=True)
        except subprocess.TimeoutExpired as exc:
            raise AdmissionError(_diagnose(exc, cmd, "runs/<latest>/server.log")) from exc
        finally:
            if lock:
                _release_gpu_lock()
                self.gpulock_pid = None
        if r.returncode != 0:
            fake = subprocess.CalledProcessError(r.returncode, cmd, r.stdout, r.stderr)
            raise AdmissionError(_diagnose(fake, cmd, "runs/<latest>/server.log"))
        out = (r.stdout or "")[-4000:]
        if out.strip():
            print(out.strip()[-1500:])
        entry = {
            "kind": kind, "cmd": cmd, "wall_s": round(time.time() - t0, 1),
            "started": datetime.now().astimezone().isoformat(timespec="seconds"),
            "session_id": self.session_id,
            "run_origin": "CURRENT_SESSION_PHYSICAL" if bind else None,
            "source_head": self.source.get("head"),
            "source_dirty": self.source.get("dirty"),
        }
        if bind:
            self._bind(bind, kind=kind)
            entry["run_id"] = bind
            print(f"[PROVENANCE] {prov.label(bind, 'CURRENT_SESSION_PHYSICAL')}")
        self.executions.append(entry)
        return r

    def run_profile(self, *, layers: str = "pipeline", tag: str | None = None) -> Path:
        """`lab profile`. layers='pipeline' is fast (record analysis);
        layers='pipeline,kernel' adds two ~5-min rocprofv3 windows."""
        date = tag or self._fresh_tag("profile")
        timeout = 1500 if layers == "pipeline" else 2400
        self._run_cli([sys.executable, "-m", "lab", "profile", "--layers", layers,
                       "--date", date], timeout=timeout, kind=f"profile:{layers}",
                      bind=f"{date}-profile")
        return lab_env.STUDENT_RUNS_ROOT / f"{date}-profile"

    def run_quality(self, experiment: str, *, reference: str | None = None,
                    date: str | None = None) -> int:
        """Quality gate for one experiment vs its declared reference baseline."""
        cmd = [sys.executable, "-m", "lab", "quality", "--experiment", experiment]
        if reference:
            cmd += ["--reference", reference]
        if date:
            cmd += ["--date", date]
        self._run_cli(cmd, timeout=900, kind=f"quality:{experiment}")
        return 0

    def run_m51_baseline(self, device: str, *, repeats: int = 1,
                         prompt_ids: str = "p1-zh-scene",
                         threads: int = 8) -> Path:
        """M5.0/M5.1 Part A: isolated text-encoder probe (cuda|cpu) per device.

        m51_baseline.py takes --out as a DIRECTORY and writes
        probe_<device>.json + cond_<device>/ inside it.
        """
        run_id = f"{self._fresh_tag('m51-te-' + device)}-m51-te-{device}"
        out = lab_env.STUDENT_RUNS_ROOT / run_id
        cmd = [sys.executable, "-u", "lab/m51_baseline.py", "--device", device,
               "--out", str(out), "--repeats", str(repeats),
               "--prompt-ids", prompt_ids, "--threads", str(threads)]
        to = 900 if device == "cuda" else 2400
        self._run_cli(cmd, timeout=to, kind=f"m51-te-{device}",
                      lock=(device == "cuda"), bind=run_id)
        return out

    def run_h3(self, experiment: str = "H3_OV_BF16", *,
               tag: str | None = None) -> Path:
        """M5.1 Part A: the OpenVINO CPU-TE experiment (H3) on this host.

        Runs the full pinned lab CLI (`python -m lab run --experiment H3_*`),
        which performs byte-level artifact admission, the conservative RAM
        gate, and the token/conditioning contract before any measurement.
        The OV BF16 arm is cheap (load ~seconds, encode sub-second/prompt),
        so the classroom uses the same protocol as the reference campaign."""
        if experiment not in ("H3_OV_FP32", "H3_OV_BF16", "H3_OV_INT8"):
            raise CourseModeError(f"not an H3 OpenVINO experiment: {experiment!r}")
        date = tag or self._fresh_tag(experiment)
        run_id = f"{date}-{experiment}"
        self._run_cli([sys.executable, "-m", "lab", "run", "--experiment", experiment,
                       "--date", date], timeout=2400, kind=f"h3:{experiment}",
                      lock=False, bind=run_id)
        return lab_env.STUDENT_RUNS_ROOT / run_id

    def run_m51_pipeline(self, *, arms: str = "A,C", tag: str | None = None) -> Path:
        """M5.1 Part B: four-request heterogeneous pipeline (A=GPU TE seq,
        C=OpenVINO CPU TE overlap). Requires staged OV artifacts.

        The CLI takes the run as a repo-relative path (`--run runs/<tag>`),
        matching the documented campaign invocation."""
        run = tag or (self._fresh_tag("m51pipe") + "-m51pipe")
        self._run_cli([sys.executable, "-u", "lab/m51_pipeline.py",
                       "--run", f"runs/{run}", "--arms", arms],
                      timeout=3600, kind=f"m51-pipeline:{arms}", bind=run)
        return lab_env.STUDENT_RUNS_ROOT / run

    def run_fairab(self, *, pairs: int = 1, blocks: int = 1,
                   tag: str | None = None) -> Path:
        """M5.1 Part C: fair-client factorial (A/C x original/fast)."""
        run = tag or (self._fresh_tag("fairab") + "-fairab")
        self._run_cli([sys.executable, "-u", "lab/m51_pipeline_fair_ab.py",
                       "--run", f"runs/{run}", "--pairs", str(pairs),
                       "--blocks", str(blocks)], timeout=5400,
                      kind="m51-fairab", bind=run)
        return lab_env.STUDENT_RUNS_ROOT / run

    def note_consumed(self, run_dir_or_id: str, *, why: str = "",
                      expect: str | tuple[str, ...] | None = None) -> None:
        """Register a physical record loaded by notebook code outside
        session.run/consume (e.g. module-specific JSON), keeping provenance
        visible and the session log complete. The category is resolved the
        same fail-closed way as consume()."""
        rid = str(run_dir_or_id).rstrip("/").split("/")[-1]
        if expect is not None:
            info = prov.require_category(rid, expect, session_id=self.session_id)
            cat = info["category"]
        else:
            try:
                cat = prov.classify_run(rid, session_id=self.session_id)["category"]
            except prov.ProvenanceError:
                # module-specific records (e.g. frozen files) classify by path
                cat = prov.classify_path(str(run_dir_or_id))["category"]
                if cat not in prov.PROVENANCE_CATEGORIES:
                    raise
        self.consumed.append({"run_id": str(run_dir_or_id), "why": why,
                              "category": cat})
        print(f"[CONSUMED PHYSICAL RECORD] run_id={run_dir_or_id} {why}")
        print(f"[PROVENANCE] {prov.label(rid, cat)}")

    def latest_campaign_run(self, experiment: str, *, min_formal: bool = True) -> str | None:
        """Newest non-smoke run of `experiment` from the current qualification
        campaign (declared via QIR_NOTEBOOK_CAMPAIGN), else newest overall."""
        cands = [r for r in list_runs(experiment) if "smoke" not in r]
        if not cands:
            return None
        camp = _campaign_dir()
        if camp and camp.is_dir():
            wanted = {p.name for p in camp.glob(f"*-{experiment}")}
            camp_runs = [r for r in cands if r in wanted]
            # runs referenced by the campaign manifest take precedence
            ref = _campaign_runs_referenced()
            camp_ref = [r for r in camp_runs if r in ref]
            if camp_ref:
                return camp_ref[-1]
            if camp_runs:
                return camp_runs[-1]
        return cands[-1]

    # -- closing -------------------------------------------------------------
    def finish(self, summary: dict) -> Path:
        root = _sessions_root()
        root.mkdir(parents=True, exist_ok=True)
        name = (f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{self.notebook}-"
                f"{self.mode}-{os.getpid()}.json")
        path = root / name
        if path.exists():
            raise FileExistsError(f"session record already exists: {path}")
        payload = {
            "notebook": self.notebook, "mode": self.mode,
            "session_id": self.session_id,
            "source": self.source,
            "started_at": self.started_at,
            "finished_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "admission": {"status": self.admission.status,
                          "detail": self.admission.detail,
                          "warnings": self.admission.warnings}
            if self.admission else None,
            "executions": self.executions, "consumed": self.consumed,
            "summary": summary,
            "kernel": sys.executable,
        }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
        print(f"session record: {path}")
        return path


def _campaign_runs_referenced() -> set[str]:
    c = _campaign_dir()
    out: set[str] = set()
    if not c or not c.is_dir():
        return out
    mf = c / "gpu_ab_summary.json"
    if mf.is_file():
        try:
            d = json.loads(mf.read_text(encoding="utf-8"))
            for row in d.get("comparisons", []):
                for k in ("baseline_run_id", "candidate_run_id"):
                    if row.get(k):
                        out.add(row[k])
        except (OSError, ValueError):
            pass
    return out


def begin(mode_requested: str | None = None, *, notebook: str = "?") -> Session:
    """Entry point used by every course notebook (first code cell)."""
    mode = resolve_mode(mode_requested)
    s = Session(mode, notebook)
    if mode != "REFERENCE":
        s.admit()  # raises AdmissionError with student-readable guidance
    else:
        s.admission = admit_hardware(mode)
        s.status()
    return s
