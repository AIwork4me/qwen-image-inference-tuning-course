"""Compress the FP32 IR to FP16 storage for memory-safe BF16 CPU deployment.

This is NOT a BF16-stored IR: weights are stored FP16 and the CPU executes with
INFERENCE_PRECISION_HINT=bf16. The distinction is recorded in the manifest and
in the M5.1 documents. Validated against the FP32-IR+BF16-hint arm.
"""
from __future__ import annotations
import argparse, hashlib, json, time
from pathlib import Path
import openvino as ov

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--src",required=True); ap.add_argument("--out",required=True)
    ap.add_argument("--xml",default="language_model_fp32.xml"); args=ap.parse_args()
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    t=time.time(); m=ov.Core().read_model(str(Path(args.src)/args.xml)); print("read",round(time.time()-t,2),flush=True)
    t=time.time(); ov.save_model(m,str(out/"language_model_bf16.xml"),compress_to_fp16=True); print("saved",round(time.time()-t,2),flush=True)
    b=out/"language_model_bf16.bin"; x=out/"language_model_bf16.xml"
    def sha(f):
        h=hashlib.sha256()
        with open(f,"rb") as fh:
            for blk in iter(lambda: fh.read(1<<23), b""): h.update(blk)
        return h.hexdigest()
    (out/"conversion_manifest.json").write_text(json.dumps({
        "source": str(Path(args.src)/args.xml), "source_xml_sha256": sha(Path(args.src)/args.xml),
        "tool":"ov.save_model compress_to_fp16=True", "storage":"FP16", "runtime_precision":"bf16",
        "output_xml_sha256": sha(x), "output_bin_sha256": sha(b),
        "bin_bytes": b.stat().st_size,
        "bin_gib": round(b.stat().st_size/2**30,3)},indent=2))
    print("DONE", b.stat().st_size/2**30, "GiB")

if __name__=="__main__": main()
