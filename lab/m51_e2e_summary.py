"""Aggregate E2E images/timings/memory into a reviewable summary + grids."""
from __future__ import annotations
import argparse, json, statistics
from pathlib import Path
from PIL import Image, ImageDraw

ARMS = ["G_native", "C0_native", "G_precomputed", "C1a_OV_FP32", "C1b_OV_BF16", "C2_OV_INT8", "C0_precomputed"]

def med(xs): return round(statistics.median(xs), 3) if xs else None

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--run",required=True); args=ap.parse_args()
    run=Path(args.run); e=run/"e2e"; d=json.loads((e/"e2e_ab.json").read_text())
    warm=json.loads((e/"warm_gpu_native.json").read_text()) if (e/"warm_gpu_native.json").exists() else {"runs":{}}
    out={"run":str(run),"human_verdict":"PENDING_HUMAN_REVIEW","arms":{},"quality":d.get("quality",{}),"lpips_available":d.get("lpips_available")}
    for arm,runs in d["arms"].items():
        cfg1=[r for t,r in runs.items() if t.endswith("|cfg1")]
        rec={"n_cfg1":len(cfg1),
             "wall_median_s":med([r["wall_s"] for r in cfg1]),
             "text_encode_median_s":med([r["text_encode_s"] for r in cfg1]),
             "ksampler_median_s":med([r["ksampler_s"] for r in cfg1]),
             "vae_median_s":med([r["vae_decode_s"] for r in cfg1]),
             "vram_peak_median_mb":med([r["vram_peak_mb"] for r in cfg1]),
             "vram_residency_median_mb":med([r["vram_residency_mb"] for r in cfg1 if r.get("vram_residency_mb")]),
             "images_per_min":round(60/ (med([r["wall_s"] for r in cfg1]) or 1),2)}
        out["arms"][arm]=rec
    # warm native session (steady, same pixel output; PNG metadata differs)
    wrun=warm.get("runs",{})
    if wrun:
        out["arms"]["G_native_warm_session"]={"n_cfg1":len(wrun),"wall_median_s":med([r["wall_s"] for r in wrun.values()]),
            "text_encode_median_s":med([r["text_encode_s"] for r in wrun.values()]),
            "ksampler_median_s":med([r["ksampler_s"] for r in wrun.values()]),
            "vae_median_s":med([r["vae_decode_s"] for r in wrun.values()]),
            "vram_peak_median_mb":med([r["vram_peak_mb"] for r in wrun.values()]),
            "vram_residency_median_mb":med([r.get("vram_residency_mb") for r in wrun.values() if r.get("vram_residency_mb")]),
            "images_per_min":round(60/(med([r["wall_s"] for r in wrun.values()]) or 1),2)}
    # quality medians/worst vs G_native
    q=d.get("quality",{}); by={}
    for k,v in q.items():
        arm=k.split("|")[0]; by.setdefault(arm,[]).append(v)
    out["quality_summary"]={arm:{"n":len(v),"psnr_db_median":round(statistics.median([x["psnr_db"] for x in v]),3),
        "psnr_db_worst":round(min([x["psnr_db"] for x in v]),3),
        "ssim_median":round(statistics.median([x["ssim"] for x in v]),5),"ssim_worst":round(min([x["ssim"] for x in v]),5),
        "lpips_median":round(statistics.median([x["lpips_alex"] for x in v]),5) if v and "lpips_alex" in v[0] else None,
        "lpips_worst":round(max([x["lpips_alex"] for x in v]),5) if v and "lpips_alex" in v[0] else None} for arm,v in by.items()}
    # memory saving statement: full-pipeline peak, not just residency
    g=out["arms"].get("G_native_warm_session") or out["arms"].get("G_native")
    c=out["arms"].get("C1b_OV_BF16")
    if g and c and g.get("vram_peak_median_mb") and c.get("vram_peak_median_mb"):
        out["vram_full_pipeline_peak_saved_mb"]=round(g["vram_peak_median_mb"]-c["vram_peak_median_mb"],1)
        out["vram_residency_saved_mb"]=round((g.get("vram_residency_median_mb") or 0)-(c.get("vram_residency_median_mb") or 0),1)
    (e/"e2e_summary.json").write_text(json.dumps(out,indent=2))
    print(json.dumps(out,indent=2))
    # grids
    grid_dir=e/"grids"; grid_dir.mkdir(exist_ok=True)
    def find(arm,pid,cfg):
        r=d["arms"].get(arm,{}).get(f"{arm}|{pid}|{cfg}")
        return Path(r["image"]) if r else None
    for pid in ["p1-zh-scene","p2-en-scene","p3-poster-text"]:
        imgs=[]
        for arm in ["G_native","C0_native","C1a_OV_FP32","C1b_OV_BF16","C2_OV_INT8"]:
            p=find(arm,pid,"cfg1")
            if p: imgs.append((arm,p))
        if len(imgs)<2: continue
        W=512; cols=len(imgs); canvas=Image.new("RGB",(W*cols,W+28),"white"); dr=ImageDraw.Draw(canvas)
        for i,(name,p) in enumerate(imgs):
            canvas.paste(Image.open(p).convert("RGB").resize((W,W)),(i*W,28))
            dr.text((i*W+8,8),name,fill="black")
        canvas.save(grid_dir/f"{pid}_cfg1.jpg",quality=92)
    # Chinese typography native-resolution crop from p3
    p3=[(a,find(a,"p3-poster-text","cfg1")) for a in ["G_native","C0_native","C0_precomputed","C1a_OV_FP32","C1b_OV_BF16","C2_OV_INT8"]]
    p3=[(a,p) for a,p in p3 if p]
    if p3:
        box=(64,96,960,832); W=box[2]-box[0]; H=box[3]-box[1]
        canvas=Image.new("RGB",(W*len(p3),H+28),"white"); dr=ImageDraw.Draw(canvas)
        for i,(name,p) in enumerate(p3):
            canvas.paste(Image.open(p).convert("RGB").crop(box),(i*W,28)); dr.text((i*W+8,8),name,fill="black")
        (e/"typography").mkdir(exist_ok=True)
        canvas.save(e/"typography/p3_poster_zh_native_crop.jpg",quality=95)

if __name__=="__main__": main()
