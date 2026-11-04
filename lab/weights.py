"""Weight integrity: pinned digests for every checkpoint the lab touches.

The trust anchor is THIS in-repo table (never a live endpoint). Pins were
recorded by the parent project at upstream Comfy-Org/Qwen-Image-2.1
revision ``WEIGHTS_REVISION`` and cross-verified against the local working
copies that produced every committed benchmark. Derived from
AIwork4me/qwen-image-rocm bench/quickstart_config.py (MIT).
"""

import hashlib
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
WEIGHTS_ROOT = REPO / "weights" / "comfyui-split"
STAMP_PATH = WEIGHTS_ROOT / ".digest_cache.json"

WEIGHTS_REVISION = "aacf31a6d48e707f352ecba714cee0680da3479b"

OFFICIAL_URL_BASE = "https://huggingface.co/Comfy-Org/Qwen-Image-2.1"
DEFAULT_MIRROR_BASE = "https://hf-mirror.com/Comfy-Org/Qwen-Image-2.1"

# rel path -> (exact byte size, pinned sha256)
WEIGHTS: dict[str, tuple[int, str]] = {
    "diffusion_models/qwen_image_2.1_bf16.safetensors": (
        14_230_280_616,
        "89f4158d066cc33906a199fca85634f766892dd78f49b6698dabf187ac86c4bc"),
    "diffusion_models/qwen_image_2.1_int8_convrot.safetensors": (
        7_256_783_064,
        "cb74113cb03faecd79611b01fd7fd642f0aa60d6f0b95086abee214d75eaa57d"),
    "diffusion_models/qwen_image_2.1_turbo_bf16.safetensors": (
        14_230_280_616,
        "8b74237a7f505773e2c66726fa696444b4c704585c93a347cff9044937f7914a"),
    "diffusion_models/qwen_image_2.1_turbo_int8_convrot.safetensors": (
        7_256_787_829,
        "8144a0182964aa8403ad6dbc011db837304167d235161b9aabaa8532e967314c"),
    "text_encoders/qwen3vl_8b_bf16.safetensors": (
        17_534_334_616,
        "68bdc82bc1b66851162ae656225e7e2068166b603db19bd5d5a3b90eb12669a9"),
    "vae/qwen_image_2.1_vae_bf16.safetensors": (
        675_509_688,
        "bb21f7473051e1ac368515dd3f2e15cd44d7a11748ee8823e1ddca3e4876b7c9"),
}


def weights_state() -> list[dict]:
    """Per-file state: present/absent/truncated (size check only — always
    follow with verify-weight for the digest check before use)."""
    states = []
    for rel in WEIGHTS:
        path = WEIGHTS_ROOT / rel
        expected = WEIGHTS[rel][0]
        if not path.exists():
            state = "absent"
        elif path.stat().st_size != expected:
            state = "truncated"
        else:
            state = "present"
        states.append({"rel": rel, "path": str(path), "expected_bytes": expected,
                       "state": state, "revision": WEIGHTS_REVISION})
    return states


def _load_stamps() -> dict:
    try:
        return json.loads(STAMP_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 23), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_weight(rel: str, update_stamp: bool = True) -> dict:
    """Trusted integrity check: local file sha256 vs the PINNED digest.

    Runs the full hash on every invocation (size-only shortcuts are a
    bypass). A stamp cache (size+mtime keyed) makes repeat checks fast, but
    any file change re-hashes. ``ok=false`` is reported, never raised.
    """
    if rel not in WEIGHTS:
        raise ValueError(f"unknown weight {rel!r}")
    path = WEIGHTS_ROOT / rel
    expected_size, pinned = WEIGHTS[rel]
    if not path.exists():
        return {"rel": rel, "ok": False, "reason": "absent",
                "expected_sha256": pinned}
    st = path.stat()
    if st.st_size != expected_size:
        return {"rel": rel, "ok": False, "reason": "size-mismatch",
                "local_size": st.st_size, "expected_size": expected_size,
                "expected_sha256": pinned}
    stamps = _load_stamps()
    entry = stamps.get(rel)
    if (entry and entry.get("size") == st.st_size
            and entry.get("mtime_ns") == st.st_mtime_ns
            and entry.get("sha256") == pinned):
        return {"rel": rel, "ok": True, "reason": "stamp-cache-hit",
                "sha256": pinned, "anchor": "pinned-digest+cache"}
    local = _sha256(path)
    ok = local == pinned
    if ok and update_stamp:
        stamps[rel] = {"size": st.st_size, "mtime_ns": st.st_mtime_ns,
                       "sha256": pinned, "verified_at": int(time.time())}
        try:
            STAMP_PATH.write_text(json.dumps(stamps, indent=2), encoding="utf-8")
        except OSError:
            pass  # cache is best-effort; verification result stands
    return {"rel": rel, "ok": ok,
            "reason": "hashed" if ok else "digest-mismatch",
            "local_sha256": local, "expected_sha256": pinned,
            "anchor": "pinned-digest"}


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=["weights-state", "verify-weight", "digests"])
    ap.add_argument("--rel", default=None, help="weight path relative to weights root")
    ns = ap.parse_args(sys.argv[1:])
    if ns.command == "weights-state":
        print(json.dumps(weights_state(), indent=2))
    elif ns.command == "digests":
        print(json.dumps({"revision": WEIGHTS_REVISION,
                          "official": OFFICIAL_URL_BASE,
                          "default_mirror": DEFAULT_MIRROR_BASE,
                          "weights": {k: {"size": v[0], "sha256": v[1]}
                                      for k, v in WEIGHTS.items()}}, indent=2))
    else:
        if not ns.rel:
            raise SystemExit("--rel is required for verify-weight")
        print(json.dumps(verify_weight(ns.rel), indent=2))
        sys.exit(0)


if __name__ == "__main__":
    main()
