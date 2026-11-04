"""Cloud staging planner — the pure, CPU-testable decision logic behind
``scripts/cloud/start.sh``.

Why this module exists: the first release put incremental disk budgeting and
ComfyUI staging decisions inline in bash, where they were (a) wrong — the
budget ignored already-present files and blocked ``--smoke-only`` on a full
disk — and (b) untestable. Every rule below has a CPU-only regression test
in ``tests/test_cloud_stage.py``; start.sh only renders the plan.

Design rules:
- Budgeting is INCREMENTAL: only files that still need downloading count
  toward the disk requirement. A verified-present file costs nothing.
- ``--smoke-only`` downloads nothing, so its budget requirement is zero.
- An existing but unrecognized ComfyUI tree is NEVER deleted — it is moved
  aside (quarantined) so the user can recover it.
- Fresh ComfyUI installs land in a private staging directory that this
  process owns, are verified (version file + main.py), and are then moved
  into place; a failed staging run cleans up only its own directory.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from lab import weights as weights_mod  # noqa: E402

COMFY_PIN = "b5a432157de6b8884f81104cd02926adf4d597e4"
COMFY_VERSION = "0.43.0"

STAGE1 = (
    "diffusion_models/qwen_image_2.1_turbo_int8_convrot.safetensors",
    "text_encoders/qwen3vl_8b_bf16.safetensors",
    "vae/qwen_image_2.1_vae_bf16.safetensors",
)
ALL = tuple(sorted(weights_mod.WEIGHTS))

# working room on top of the download itself (venv wheels, tarballs,
# parts files of the resumable downloader)
WORKING_MARGIN_GIB = 2.0


def disk_free_bytes(path: Path) -> int:
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize


def _file_state(rel: str, weights_root: Path) -> str:
    """present | truncated | absent (size check only — fast, no hashing;
    the digest verification happens later in start.sh step 4)."""
    path = weights_root / rel
    expected = weights_mod.WEIGHTS[rel][0]
    if not path.exists():
        return "absent"
    if path.stat().st_size != expected:
        return "truncated"
    return "present"


def plan_model_staging(*, weights_root: Path, all_models: bool = False,
                       smoke_only: bool = False,
                       free_bytes: int | None = None) -> dict:
    """Incremental staging plan for the pinned weight set.

    Returns a dict:
      targets      — the rel-paths this run aims to have
      present      — already present with the pinned byte size
      broken       — present but wrong size (must be re-downloaded)
      downloads    — rel-paths that must be fetched (absent + broken)
      need_gib     — incremental bytes the downloads will consume
      free_gib     — bytes available at the weights root (or the injected value)
      verdict      — OK | BLOCKED
      message      — human-readable explanation
    """
    targets = list(ALL if all_models else STAGE1)
    present, broken, downloads = [], [], []
    for rel in targets:
        state = _file_state(rel, weights_root)
        if state == "present":
            present.append(rel)
        elif state == "truncated":
            broken.append(rel)
            downloads.append(rel)
        else:
            downloads.append(rel)

    if smoke_only:
        # --smoke-only performs no downloads at all; a nearly-full disk
        # must not block it (the smoke itself writes only MBs).
        need_bytes = 0
    else:
        need_bytes = sum(weights_mod.WEIGHTS[rel][0] for rel in downloads)

    free = free_bytes if free_bytes is not None else disk_free_bytes(weights_root)
    need_gib = need_bytes / 2**30
    free_gib = free / 2**30
    # working room for the downloader's parts files / tarballs — only
    # relevant when something actually gets downloaded
    margin = WORKING_MARGIN_GIB if (downloads and not smoke_only) else 0.0

    if free < need_bytes + margin * 2**30:
        verdict = "BLOCKED"
        message = (
            f"{free_gib:.1f} GiB free at {weights_root} but this run still "
            f"needs ~{need_gib:.1f} GiB of downloads (+{margin:.0f} GiB "
            "working room). Already-present pinned files were NOT counted. "
            "Free space, attach larger persistent storage, or run the "
            "default staged set (no --all-models).")
    else:
        verdict = "OK"
        message = (f"disk budget: {free_gib:.1f} GiB free >= "
                   f"~{need_gib:.1f} GiB of new downloads "
                   f"(+{margin:.0f} GiB working room; "
                   f"{len(present)} file(s) already present)")

    return {
        "targets": targets, "present": present, "broken": broken,
        "downloads": downloads, "need_gib": round(need_gib, 2),
        "free_gib": round(free_gib, 1), "verdict": verdict,
        "message": message,
    }


def classify_comfyui(comfy_dir: Path, pin: str = COMFY_PIN) -> dict:
    """Decide what to do with an existing ComfyUI directory.

    Actions:
      keep                  — pinned marker + main.py + matching version file
      stage_fresh           — nothing usable exists at the path
      quarantine_then_stage — something exists but is not OUR pinned tree:
                              NEVER deleted; moved aside by start.sh
      refuse                — path exists but is not a directory (start.sh
                              aborts; the user resolves it manually)
    """
    if not comfy_dir.exists():
        return {"action": "stage_fresh",
                "reason": f"no ComfyUI at {comfy_dir}"}
    if not comfy_dir.is_dir():
        return {"action": "refuse",
                "reason": f"{comfy_dir} exists but is not a directory — "
                          "resolve it manually (nothing was touched)"}
    marker = comfy_dir / ".lab_pinned_commit"
    version_file = comfy_dir / "comfyui_version.py"
    main = comfy_dir / "main.py"
    marker_ok = (marker.is_file() and marker.read_text().strip() == pin)
    version_ok = False
    if version_file.is_file():
        m = re.search(r'__version__\s*=\s*"([^"]+)"',
                      version_file.read_text(errors="replace"))
        version_ok = bool(m and m.group(1) == COMFY_VERSION)
    if marker_ok and main.is_file() and version_ok:
        return {"action": "keep",
                "reason": f"pinned tree verified (marker + v{COMFY_VERSION} "
                          "+ main.py)"}
    why = []
    if not marker_ok:
        why.append("pinned-commit marker missing or mismatched")
    if not main.is_file():
        why.append("main.py missing (partial extraction?)")
    if not version_ok:
        why.append(f"comfyui_version.py missing or not v{COMFY_VERSION}")
    return {"action": "quarantine_then_stage",
            "reason": "existing tree is not the lab's pinned ComfyUI ("
                      + "; ".join(why) + ") — it will be moved aside, "
                      "never deleted"}


def verify_staged_tree(tree: Path, pin: str = COMFY_PIN) -> dict:
    """Integrity gate for a freshly staged ComfyUI tree BEFORE it replaces
    anything: main.py present and the version file reports the pinned
    version. The commit identity itself is anchored by the fetch source
    (git checkout of the pin, or the codeload tarball named by the pin)."""
    main = tree / "main.py"
    version_file = tree / "comfyui_version.py"
    if not main.is_file():
        return {"ok": False, "reason": "main.py missing from staged tree"}
    if not version_file.is_file():
        return {"ok": False, "reason": "comfyui_version.py missing"}
    m = re.search(r'__version__\s*=\s*"([^"]+)"',
                  version_file.read_text(errors="replace"))
    if not m or m.group(1) != COMFY_VERSION:
        return {"ok": False,
                "reason": f"staged tree reports version "
                          f"{m.group(1) if m else '<unparseable>'}, "
                          f"expected {COMFY_VERSION}"}
    return {"ok": True, "reason": f"staged tree verified (v{COMFY_VERSION})"}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("plan", help="incremental model staging plan")
    p.add_argument("--repo", default=str(REPO))
    p.add_argument("--all-models", action="store_true")
    p.add_argument("--smoke-only", action="store_true")
    p.add_argument("--free-bytes", type=int, default=None,
                   help="override the measured free space (testing)")
    p.add_argument("--json", action="store_true")

    p = sub.add_parser("comfy-action", help="decision for an existing ComfyUI dir")
    p.add_argument("--repo", default=str(REPO))
    p.add_argument("--pin", default=COMFY_PIN)
    p.add_argument("--action-only", action="store_true",
                   help="print just the action word (for start.sh)")

    p = sub.add_parser("verify-staged", help="integrity gate for a staged tree")
    p.add_argument("tree")

    ns = ap.parse_args(argv)
    repo = Path(ns.repo).resolve() if hasattr(ns, "repo") else REPO
    if ns.cmd == "plan":
        plan = plan_model_staging(
            weights_root=repo / "weights" / "comfyui-split",
            all_models=ns.all_models, smoke_only=ns.smoke_only,
            free_bytes=ns.free_bytes)
        if ns.json:
            print(json.dumps(plan, indent=2))
        else:
            print(f"VERDICT {plan['verdict']}")
            print(f"NEED_GIB {plan['need_gib']}")
            print(f"FREE_GIB {plan['free_gib']}")
            for rel in plan["downloads"]:
                print(f"DOWNLOAD {rel}")
            for rel in plan["broken"]:
                print(f"BROKEN {rel}")
            for rel in plan["present"]:
                print(f"PRESENT {rel}")
            print(f"MESSAGE {plan['message']}")
        return 0 if plan["verdict"] == "OK" else 3
    if ns.cmd == "comfy-action":
        verdict = classify_comfyui(repo / "ComfyUI", ns.pin)
        if ns.action_only:
            print(verdict["action"])
        else:
            print(json.dumps(verdict))
        return 0
    if ns.cmd == "verify-staged":
        verdict = verify_staged_tree(Path(ns.tree))
        print(json.dumps(verdict))
        return 0 if verdict["ok"] else 1
    return 2


if __name__ == "__main__":
    sys.exit(main())
