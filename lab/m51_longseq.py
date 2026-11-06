"""Long-sequence OpenVINO compatibility probe (INT8 hazard + precision trend)."""
from __future__ import annotations

import argparse, json, time
from pathlib import Path
import numpy as np
from ov_encoder import OVTextEncoder, REPO

TEXTS = {
    "zh_long": "一幅超高清电影级海报设计，画面中是一座未来城市的天际线，霓虹灯映照在湿润的街道上，行人撑伞走过，远处的高铁穿行于楼宇之间，天空呈现深蓝色与橙色的渐变，主标题文字清晰锐利，副标题排版居中，构图严谨，细节丰富。" * 3,
    "en_long": "A cinematic ultra-detailed photograph of a mountain lake at sunrise, with snow-capped peaks reflected in still water, pine trees along the shore, soft mist rising from the surface, dramatic clouds, golden light, high dynamic range, sharp focus, professional composition, and rich natural colors. " * 3,
    "zh_xlong": "一幅超高清电影级海报设计，画面中是一座未来城市的天际线，霓虹灯映照在湿润的街道上，行人撑伞走过，远处的高铁穿行于楼宇之间，天空呈现深蓝色与橙色的渐变，主标题文字清晰锐利，副标题排版居中，构图严谨，细节丰富。" * 7,
    "en_xlong": "A cinematic ultra-detailed photograph of a mountain lake at sunrise, with snow-capped peaks reflected in still water, pine trees along the shore, soft mist rising from the surface, dramatic clouds, golden light, high dynamic range, sharp focus, professional composition, and rich natural colors. " * 7,
}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--arm",required=True); ap.add_argument("--model-dir",required=True)
    ap.add_argument("--xml",default="language_model_fp32.xml"); ap.add_argument("--precision",default="f32")
    ap.add_argument("--out",required=True); ap.add_argument("--threads",type=int,default=16)
    args=ap.parse_args(); out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    enc=OVTextEncoder(args.model_dir, precision=args.precision, threads=args.threads, model_xml=args.xml)
    res={"arm":args.arm,"meta":enc.meta(),"cases":{}}
    for name,text in TEXTS.items():
        try:
            o=enc.encode(text)
            (out/f"cond_{args.arm}").mkdir(exist_ok=True)
            np.savez(out/f"cond_{args.arm}/{name}.npz", cond=o["hidden"], ids=np.array(o["ids"]))
            res["cases"][name]={"status":"PASS","n_tokens_full":o["n_tokens_full"],
                "n_tokens_kept":o["n_tokens_kept"],"encode_s":round(o["encode_s"],3),
                "finite":bool(np.isfinite(o["hidden"]).all()),"shape":list(o["hidden"].shape)}
        except Exception as e:
            res["cases"][name]={"status":"FAIL","error":repr(e)[:1000]}
        print(name,res["cases"][name],flush=True)
    (out/f"longseq_{args.arm}.json").write_text(json.dumps(res,indent=2))

if __name__=="__main__": main()
