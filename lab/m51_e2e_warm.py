"""Measure steady GPU-native E2E latency in a fresh server (one warm-up)."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from .server import ComfyServer
from . import m51_e2e as E

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--run",required=True); ap.add_argument("--port",type=int,default=8199)
    args=ap.parse_args(); run=Path(args.run); raw=run/"e2e/raw"; raw.mkdir(parents=True,exist_ok=True)
    E.install_node(); res={"arm":"G_native","warmup_s":None,"runs":{}}
    srv=ComfyServer(port=args.port, log_path=run/"e2e/server_warm.log"); srv.start()
    try:
        s=E.run_one(srv,"warmup__G_native",E.wf_native(E.PROMPTS["p2-en-scene"]["prompt"],"",42,1.0),raw)
        res["warmup_s"]=s["wall_s"]
        for pid,pr in E.PROMPTS.items():
            tag=f"G_native|{pid}|cfg1"
            r=E.run_one(srv,tag,E.wf_native(pr["prompt"],pr["negative_prompt"],42,1.0),raw)
            res["runs"][tag]={k:r[k] for k in ("wall_s","text_encode_s","ksampler_s","vae_decode_s","vram_peak_mb","vram_residency_mb","image","image_sha256")}
            print(tag,r["wall_s"],r["text_encode_s"],r["ksampler_s"],r["vram_peak_mb"],flush=True)
    finally:
        srv.stop()
    (run/"e2e/warm_gpu_native.json").write_text(json.dumps(res,indent=2))

if __name__=="__main__": main()
