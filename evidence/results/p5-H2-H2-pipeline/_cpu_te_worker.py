
import sys, time
sys.path.insert(0, "/workspace/qwen-image-rocm/phase2-comfyui/ComfyUI")
import torch

text, out_path = sys.argv[1], sys.argv[2]
import comfy.sd
import comfy.model_management as mm

t0 = time.perf_counter()
clip = comfy.sd.load_clip(
    ckpt_paths=["/workspace/qwen-image-rocm/weights/comfyui-split/text_encoders/qwen3vl_8b_bf16.safetensors"],
    embedding_directory="/workspace/qwen-image-rocm/phase2-comfyui/ComfyUI/models/embeddings",
    clip_type=comfy.sd.CLIPType.QWEN_IMAGE,
    model_options={"load_device": torch.device("cpu"),
                   "offload_device": torch.device("cpu")})
load_s = time.perf_counter() - t0

t0 = time.perf_counter()
tokens = clip.tokenize(text)
tok_s = time.perf_counter() - t0
t0 = time.perf_counter()
cond_out = clip.encode_from_tokens_scheduled(tokens)
cond = cond_out[0][0] if isinstance(cond_out, list) else cond_out
enc_s = time.perf_counter() - t0

torch.save({"cond": cond.cpu(), "device": "cuda"}, out_path)
import os
print(json.dumps if False else "", file=sys.stderr)
import json
meta = {"load_s": round(load_s, 2), "tokenize_s": round(tok_s, 4),
        "encode_s": round(enc_s, 3), "shape": list(cond.shape),
        "dtype": str(cond.dtype), "file_bytes": os.path.getsize(out_path)}
json.dump(meta, open(out_path + ".meta.json", "w"), indent=1)
print(json.dumps(meta))
