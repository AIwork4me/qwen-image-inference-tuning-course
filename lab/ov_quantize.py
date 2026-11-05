"""Create the C2 OpenVINO INT8 weight-compressed IR from the clean FP32 export.

NNCF INT8_ASYM, ratio=1.0, group_size=-1 (channel-wise), matching the
validated upstream recipe. The token embedding table stays BF16->FP32 at
runtime, so this is explicitly a transformer-weight INT8 arm, not the more
aggressive upstream int8-embedding variant.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import openvino as ov
import nncf


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 23), b""):
            h.update(b)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--xml", default="language_model_fp32.xml")
    args = ap.parse_args()
    src, out = Path(args.src), Path(args.out); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    model = ov.Core().read_model(str(src / args.xml))
    print(f"read {args.xml} in {time.time()-t0:.1f}s", flush=True)
    t0 = time.time()
    compressed = nncf.compress_weights(
        model, mode=nncf.CompressWeightsMode.INT8_ASYM, ratio=1.0, group_size=-1)
    print(f"compressed in {time.time()-t0:.1f}s", flush=True)
    xml = out / "language_model_int8.xml"
    ov.save_model(compressed, str(xml), compress_to_fp16=False)
    binp = xml.with_suffix(".bin")
    manifest = {
        "source_dir": str(src), "source_xml_sha256": sha256(src / args.xml),
        "source_bin_bytes": (src / args.xml).with_suffix(".bin").stat().st_size,
        "tool": f"nncf {nncf.__version__} compress_weights",
        "mode": "INT8_ASYM", "ratio": 1.0, "group_size": -1,
        "embeddings": "BF16->FP32 runtime gather from pinned checkpoint (not int8)",
        "output_xml_sha256": sha256(xml), "output_bin_bytes": binp.stat().st_size,
        "output_bin_sha256": sha256(binp),
        "output_gib": round(binp.stat().st_size / 2**30, 3),
    }
    (out / "conversion_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
