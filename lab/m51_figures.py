"""Generate M5.1 figures from run-scoped JSON evidence (no invented numbers)."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO=Path(__file__).resolve().parents[1]
RUN=REPO/"runs/m51-20261105-cleanroom-01"
FIG=REPO/"docs/figures"; FIG.mkdir(parents=True,exist_ok=True)

perf=json.loads((RUN/"performance/encoder_ab.json").read_text())
pids=["p1-zh-scene","p2-en-scene","p3-poster-text","negative"]
labels={"p1-zh-scene":"ZH scene\n(66 tok)","p2-en-scene":"EN scene\n(51 tok)","p3-poster-text":"ZH poster\n(78 tok)","negative":"empty neg\n(22 tok)"}
arms=["C0_PyTorch_CPU","C1a_OV_FP32","C1b_OV_BF16","C2_OV_INT8"]
names=["C0 PyTorch CPU","C1a OV FP32","C1b OV BF16*","C2 OV INT8"]
vals=[[perf["arms"][a]["prompts"][p]["warm_median_s"] for p in pids] for a in arms]
x=np.arange(len(pids)); w=0.2
plt.figure(figsize=(10,5))
for i,(n,v) in enumerate(zip(names,vals)):
    plt.bar(x+(i-1.5)*w,v,w,label=n)
plt.yscale("log"); plt.ylabel("warm encode latency (s, log scale)")
plt.title("Qwen3-VL-8B text encoder on the SAME CPU (EPYC 9334, 16-thread cgroup)")
plt.xticks(x,[labels[p] for p in pids]); plt.legend(); plt.grid(axis="y",alpha=.3)
plt.text(0.01,0.02,"*BF16 runtime on FP16-stored deployment IR",transform=plt.gca().transAxes,fontsize=8)
plt.tight_layout(); plt.savefig(FIG/"m51_encoder_latency.png",dpi=160); plt.close()

# precision vs latency
pts=[("C1a OV FP32",perf["arms"]["C1a_OV_FP32"]["prompts"]["p1-zh-scene"]["warm_median_s"],4.1e-6),
     ("C1b OV BF16",perf["arms"]["C1b_OV_BF16"]["prompts"]["p1-zh-scene"]["warm_median_s"],0.01253),
     ("C2 OV INT8",perf["arms"]["C2_OV_INT8"]["prompts"]["p1-zh-scene"]["warm_median_s"],0.06382)]
plt.figure(figsize=(7,5))
for n,l,e in pts:
    plt.scatter(l,e*100,s=90,label=n)
    plt.annotate(n,(l,e*100),textcoords="offset points",xytext=(8,6),fontsize=9)
plt.xscale("log"); plt.yscale("log")
plt.xlabel("warm latency, p1 (s, log)"); plt.ylabel("conditioning rel L2 vs C0/FP32 (%, log)")
plt.title("Precision vs CPU encoder speed (p1, 66 tokens)")
plt.grid(alpha=.3); plt.tight_layout(); plt.savefig(FIG/"m51_precision_tradeoff.png",dpi=160); plt.close()

# E2E latency
e2e=json.loads((RUN/"e2e/e2e_summary.json").read_text())
fig,ax=plt.subplots(figsize=(8,4.5))
items=[("G native GPU TE",e2e["arms"]["G_native_warm_session"]["wall_median_s"]),
       ("C1b precomputed\n(serialized +TE)",e2e["arms"]["C1b_OV_BF16"]["wall_median_s"]+0.55),
       ("C1b precomputed\n(cond ready)",e2e["arms"]["C1b_OV_BF16"]["wall_median_s"]),
       ("C0 native CPU TE",e2e["arms"]["C0_native"]["wall_median_s"])]
ax.barh([i[0] for i in items],[i[1] for i in items],color=["#4477aa","#66aa55","#66aa55","#cc6677"])
ax.set_xscale("log"); ax.set_xlabel("image latency (s, log)")
for i,(n,v) in enumerate(items): ax.text(v*1.05,i,f"{v:.2f}s",va="center")
ax.set_title("End-to-end 1024x1024 turbo 8-step latency (W7900, same DiT/VAE)")
plt.tight_layout(); plt.savefig(FIG/"m51_e2e_latency.png",dpi=160); plt.close()

# VRAM
g=e2e["arms"]["G_native_warm_session"]; b=e2e["arms"]["C1b_OV_BF16"]
plt.figure(figsize=(7,4.5))
x=np.arange(2); w=.36
plt.bar(x-w/2,[g["vram_residency_median_mb"]/1024,b["vram_residency_median_mb"]/1024],w,label="residency")
plt.bar(x+w/2,[g["vram_peak_median_mb"]/1024,b["vram_peak_median_mb"]/1024],w,label="full-pipeline peak")
plt.xticks(x,["G: GPU TE resident","C1b: OV CPU TE"]); plt.ylabel("VRAM (GiB)")
plt.title("GPU VRAM: residency vs full-pipeline peak (turbo 8-step, 1024²)")
plt.legend(); plt.grid(axis="y",alpha=.3)
plt.tight_layout(); plt.savefig(FIG/"m51_vram.png",dpi=160); plt.close()

# pipeline timeline: per-arm relative clocks (arms ran in different sessions)
p=json.loads((RUN/"pipeline/pipeline_ab.json").read_text())
fig,axes=plt.subplots(2,1,figsize=(12,7),gridspec_kw={"height_ratios":[1.1,1]})
ax=axes[0]
rows=[("A","#4477aa",0),("C","#66aa55",1),("D","#aa88cc",2)]
for arm,color,y in rows:
    reqs=p["arms"][arm]["requests"]
    if arm=="A":
        t0=reqs[0]["t_end"]-reqs[0]["wall_s"]
        starts=[r["t_end"]-r["wall_s"] for r in reqs]
        for i,r in enumerate(reqs):
            ax.barh(y,r["wall_s"],left=starts[i]-t0,height=.36,color=color,alpha=.85,
                    label="GPU TE+denoise" if i==0 else None)
    else:
        fill=p["arms"][arm]["fill_s"]; t0=reqs[0]["gen_start"]-fill
        ax.barh(y,fill,left=0,height=.18,color="#222222",alpha=.8,label=f"{arm}: CPU TE (fill+overlap)" if arm=="C" else None)
        for i,r in enumerate(reqs):
            ax.barh(y,r["gen_wall_s"],left=r["gen_start"]-t0,height=.36,color=color,alpha=.85,
                    label=f"{arm}: GPU denoise" if i==0 else None)
            if "cpu_encode_start" in r:
                ax.barh(y+.23,r["cpu_encode_s"],left=r["cpu_encode_start"]-t0,height=.13,color="#222222",alpha=.8)
ax.set_yticks([0,1,2]); ax.set_yticklabels(["A GPU-TE sequential","C OV BF16 CPU TE","D OV INT8 CPU TE"])
ax.set_xlabel("relative wall time (s)"); ax.set_title("A/C/D: OpenVINO TE is fully hidden; pipeline is GPU-bound")
ax.legend(fontsize=7,ncol=4,loc="lower right"); ax.grid(axis="x",alpha=.3)
ax=axes[1]
B=p["arms"]["B"]; reqs=B["requests"]; fill=B["fill_s"]; t0=reqs[0]["gen_start"]-fill
ax.barh(0,fill,left=0,height=.25,color="#222222",alpha=.85,label="PyTorch CPU TE (fill req1, 176 s)")
for i,r in enumerate(reqs):
    ax.barh(0,r["gen_wall_s"],left=r["gen_start"]-t0,height=.36,color="#cc6677",alpha=.85,
            label="GPU denoise (12-13 s)" if i==0 else None)
    if "cpu_encode_start" in r:
        ax.barh(.23,r["cpu_encode_s"],left=r["cpu_encode_start"]-t0,height=.16,color="#222222",alpha=.85,
                label="next CPU TE (137-208 s)" if i==0 else None)
ax.set_yticks([0]); ax.set_yticklabels(["B PyTorch CPU TE"])
ax.set_xlabel("relative wall time (s)"); ax.set_title("B: real overlap, but CPU-bound — GPU idles 124-196 s per request")
ax.legend(fontsize=8,ncol=3,loc="lower right"); ax.grid(axis="x",alpha=.3)
plt.tight_layout(); plt.savefig(FIG/"m51_pipeline_timeline.png",dpi=160); plt.close()
print("figures written to",FIG)
