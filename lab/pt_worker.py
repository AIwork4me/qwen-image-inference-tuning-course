"""Persistent PyTorch CPU text-encoder worker (the M5.0 C0 path)."""
from __future__ import annotations
import argparse, json, os, sys, time, traceback
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import save_file

REPO = Path(__file__).resolve().parents[1]
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != REPO / "lab"]
sys.path.insert(0, str(REPO / "ComfyUI"))

def atomic_save(path: str, cond):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + f".tmp{os.getpid()}")
    save_file({"cond": cond.detach().to("cpu", torch.float32).contiguous()}, str(tmp))
    os.replace(tmp, p)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--threads", type=int, default=16)
    args = ap.parse_args(); torch.set_num_threads(args.threads)
    import comfy.sd
    t0 = time.time()
    clip = comfy.sd.load_clip(
        ckpt_paths=[str(REPO/"weights/comfyui-split/text_encoders/qwen3vl_8b_bf16.safetensors")],
        embedding_directory=str(REPO/"ComfyUI/models/embeddings"),
        clip_type=comfy.sd.CLIPType.QWEN_IMAGE,
        model_options={"load_device": torch.device("cpu"), "offload_device": torch.device("cpu")})
    print(json.dumps({"ready": True, "load_s": round(time.time()-t0, 3), "threads": args.threads}), flush=True)
    for line in sys.stdin:
        line = line.strip()
        if not line: continue
        req = None
        try:
            req = json.loads(line)
            if req.get("cmd") == "quit": break
            t_start = time.time()
            toks = clip.tokenize(req["text"])
            out = clip.encode_from_tokens_scheduled(toks)
            cond = out[0][0] if isinstance(out, list) else out
            t_end = time.time()
            atomic_save(req["out"], cond)
            print(json.dumps({"id": req.get("id"), "ok": True, "t_start": t_start, "t_end": t_end,
                              "encode_s": t_end-t_start, "shape": list(cond.shape),
                              "out": req["out"]}), flush=True)
        except Exception as e:
            print(json.dumps({"id": (req or {}).get("id"), "ok": False, "error": repr(e),
                              "traceback": traceback.format_exc()[-2000:]}), flush=True)

if __name__ == "__main__": main()
