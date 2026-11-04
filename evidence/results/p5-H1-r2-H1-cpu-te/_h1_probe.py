
import sys, json, time
sys.path.insert(0, "/workspace/qwen-image-rocm/phase2-comfyui/ComfyUI")
import torch
import comfy.sd

PROMPTS = json.load(open("/workspace/qwen-image-rocm/bench/prompts.json"))
texts = {p["id"]: p["prompt"] for p in PROMPTS["prompts"]}
TE = "/workspace/qwen-image-rocm/weights/comfyui-split/text_encoders/qwen3vl_8b_bf16.safetensors"
EMB = "/workspace/qwen-image-rocm/phase2-comfyui/ComfyUI/models/embeddings"

out = {}
conds = {}
for device in ("cuda", "cpu"):
    tag = "gpu" if device == "cuda" else "cpu"
    try:
        t0 = time.perf_counter()
        clip = comfy.sd.load_clip(
            ckpt_paths=[TE], embedding_directory=EMB,
            clip_type=comfy.sd.CLIPType.QWEN_IMAGE,
            model_options={"load_device": torch.device(device),
                           "offload_device": torch.device(device)})
        load_s = time.perf_counter() - t0
        enc = {}
        for pid, text in texts.items():
            t0 = time.perf_counter()
            tokens = clip.tokenize(text)
            tok_s = time.perf_counter() - t0
            t0 = time.perf_counter()
            cond_out = clip.encode_from_tokens_scheduled(tokens)
            cond = cond_out[0][0] if isinstance(cond_out, list) else cond_out
            if device == "cuda":
                torch.cuda.synchronize()
            enc_s = time.perf_counter() - t0
            enc[pid] = {"tokenize_s": round(tok_s, 4),
                        "encode_s": round(enc_s, 3),
                        "shape": list(cond.shape), "dtype": str(cond.dtype)}
            conds.setdefault(pid, {})[tag] = cond.detach().float().cpu()
        out[tag] = {"load_s": round(load_s, 2), "encode": enc}
        if device == "cuda":
            out[tag]["vram_allocated_mb"] = round(torch.cuda.memory_allocated() / 2**20, 1)
        del clip
        if device == "cuda":
            torch.cuda.empty_cache()
    except Exception as exc:  # noqa: BLE001
        out[tag] = {"error": repr(exc)[:400]}

# numerical agreement gpu vs cpu
agree = {}
for pid, pair in conds.items():
    if "gpu" in pair and "cpu" in pair:
        g, c = pair["gpu"], pair["cpu"]
        if g.shape == c.shape:
            agree[pid] = {
                "max_abs_diff": round((g - c).abs().max().item(), 5),
                "mean_abs_diff": round((g - c).abs().mean().item(), 6),
                "cosine": round(torch.nn.functional.cosine_similarity(
                    g.flatten().unsqueeze(0), c.flatten().unsqueeze(0)).item(), 6),
            }
        else:
            agree[pid] = {"shape_mismatch": [list(g.shape), list(c.shape)]}
out["agreement_gpu_vs_cpu"] = agree
json.dump(out, open(sys.argv[1], "w"), indent=1)
print(json.dumps(out, indent=1))
