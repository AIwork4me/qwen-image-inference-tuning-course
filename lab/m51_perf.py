"""Build the same-machine CPU encoder A/B artifact from probe JSONs."""
import argparse, json, statistics
from pathlib import Path

def load(p): return json.loads(Path(p).read_text())

def summarize(rows):
    warm = [r["encode_s"] for r in rows[1:]]
    return {"cold_s": rows[0]["encode_s"], "warm_samples_s": warm,
            "warm_median_s": round(statistics.median(warm), 4) if warm else None,
            "warm_min_s": min(warm) if warm else None, "warm_max_s": max(warm) if warm else None,
            "n_warm": len(warm), "shape": rows[0]["shape"],
            "n_tokens_full": rows[0]["n_tokens_full"], "ids": rows[0].get("ids", rows[0].get("ids_clip"))}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--run",required=True); args=ap.parse_args()
    run=Path(args.run)
    arms={
        "G_PyTorch_GPU": load(run/"baseline/probe_cuda.json"),
        "C0_PyTorch_CPU": load(run/"baseline/probe_cpu.json"),
        "C1a_OV_FP32": load(run/"ov/run_fp32/probe_C1a_OV_FP32.json"),
        "C1b_OV_BF16": load(run/"ov/run_bf16_compressed/probe_C1b_OV_BF16_COMPRESSED.json"),
        "C1b_OV_BF16_uncompressedIR": load(run/"ov/run_bf16/probe_C1b_OV_BF16.json"),
        "C2_OV_INT8": load(run/"ov/run_int8/probe_C2_OV_INT8.json"),
    }
    out={"run":str(run),"method":{
        "same_cpu":"all CPU arms measured on AMD EPYC 9334 under a 16-CPU cgroup quota",
        "threads":{"C0":16,"C1":16},"repeats":"C0 cold+1 warm per prompt (n=1 warm); C1 cold+2 warm (n=2)",
        "cold_definition":"model load + compile/init + first conditioning tensor",
        "warm_definition":"same-process repeated encode of the same frozen token sequence",
        "primary_ratio":"median warm C0 / median warm C1"}, "arms":{}, "speedup_vs_C0":{}}
    for name,d in arms.items():
        rows={pid:summarize(v) for pid,v in d["prompts"].items()}
        out["arms"][name]={"load_s":d.get("load_s"),"rss_peak_mb":d.get("rss_peak_mb"),
                           "meta":d.get("meta"),"prompts":rows,
                           "cold_to_first_cond_s":round((d.get("load_s") or 0)+rows[list(rows)[0]]["cold_s"],3)}
    c0=out["arms"]["C0_PyTorch_CPU"]["prompts"]
    for name in ["C1a_OV_FP32","C1b_OV_BF16","C1b_OV_BF16_uncompressedIR","C2_OV_INT8"]:
        out["speedup_vs_C0"][name]={pid:round(c0[pid]["warm_median_s"]/out["arms"][name]["prompts"][pid]["warm_median_s"],1) for pid in c0}
    ts=run/"performance/thread_sweep_bf16.json"
    if ts.exists(): out["thread_sweep_bf16"]=load(ts)
    (run/"performance").mkdir(exist_ok=True)
    (run/"performance/encoder_ab.json").write_text(json.dumps(out,indent=2))
    for name in out["speedup_vs_C0"]: print(name,out["speedup_vs_C0"][name])
    for pid in c0: print(pid,"C0",c0[pid]["warm_median_s"],"C1b",out["arms"]["C1b_OV_BF16"]["prompts"][pid]["warm_median_s"],"C2",out["arms"]["C2_OV_INT8"]["prompts"][pid]["warm_median_s"])

if __name__=="__main__": main()
