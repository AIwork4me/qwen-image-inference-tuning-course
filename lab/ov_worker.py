"""Persistent OpenVINO CPU text-encoder worker (JSON-lines IPC, safetensors out)."""
from __future__ import annotations
import argparse, json, os, sys, time, traceback
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import save_file

from ov_encoder import OVTextEncoder  # ov_encoder removes lab/ from sys.path itself

def atomic_save(path: str, cond: np.ndarray):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + f".tmp{os.getpid()}")
    save_file({"cond": torch.from_numpy(cond[None] if cond.ndim == 2 else cond).contiguous().float()}, str(tmp))
    os.replace(tmp, p)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True); ap.add_argument("--xml", default="language_model_bf16.xml")
    ap.add_argument("--precision", default="bf16"); ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--dynamic-quant-group", type=int, default=None)
    args = ap.parse_args()
    t0 = time.time()
    enc = OVTextEncoder(args.model_dir, precision=args.precision, threads=args.threads,
                        model_xml=args.xml, dynamic_quant_group=args.dynamic_quant_group)
    print(json.dumps({"ready": True, "load_s": round(time.time() - t0, 3), "meta": enc.meta()}), flush=True)
    for line in sys.stdin:
        line = line.strip()
        if not line: continue
        try:
            req = json.loads(line)
            if req.get("cmd") == "quit":
                break
            t_start = time.time()
            o = enc.encode(req["text"])
            atomic_save(req["out"], o["hidden"])
            t_end = time.time()
            print(json.dumps({"id": req.get("id"), "ok": True, "t_start": t_start, "t_end": t_end,
                              "encode_s": o["encode_s"], "n_tokens_full": o["n_tokens_full"],
                              "n_tokens_kept": o["n_tokens_kept"], "out": req["out"]}), flush=True)
        except Exception as e:
            print(json.dumps({"id": (req.get("id") if 'req' in dir() else None), "ok": False,
                              "error": repr(e), "traceback": traceback.format_exc()[-2000:]}), flush=True)

if __name__ == "__main__": main()
