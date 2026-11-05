"""OpenVINO BF16 thread sweep on the same 16-CPU cgroup quota."""
import argparse, json, time
from pathlib import Path
from ov_encoder import OVTextEncoder, REPO

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--model-dir",required=True); ap.add_argument("--xml",default="language_model_bf16.xml")
    ap.add_argument("--out",required=True); ap.add_argument("--threads",default="4,8,16,32"); args=ap.parse_args()
    prompts={p["id"]:p["prompt"] for p in json.loads((REPO/"lab/prompts.json").read_text())["prompts"]}
    out={"model_dir":args.model_dir,"xml":args.xml,"results":{}}
    for th in [int(x) for x in args.threads.split(",")]:
        enc=OVTextEncoder(args.model_dir,precision="bf16",threads=th,model_xml=args.xml)
        rows={}
        for pid in ["p1-zh-scene","p3-poster-text"]:
            samples=[]
            for _ in range(2):
                o=enc.encode(prompts[pid]); samples.append(round(o["encode_s"],4))
            rows[pid]={"samples":samples,"min":min(samples)}
        out["results"][th]={"compile_s":round(enc.compile_s,3),"prompts":rows}
        print(th,rows,flush=True)
    Path(args.out).parent.mkdir(parents=True,exist_ok=True)
    Path(args.out).write_text(json.dumps(out,indent=2))

if __name__=="__main__": main()
