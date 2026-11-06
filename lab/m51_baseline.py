"""M5.0 rebaseline probe (PyTorch BF16 CLIP on GPU and CPU).

Run in an isolated process per device so RSS/VRAM measurements are not
contaminated by the other arm. Records token IDs, conditioning tensors,
timings, and memory. Outputs are run-scoped and written atomically.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
# Remove lab/ from sys.path (it contains profile.py which would shadow the
# stdlib profile module and break torch dynamo/cProfile imports when this
# script is invoked directly as `python lab/m51_baseline.py`), then put the
# repo root and the pinned ComfyUI tree on the path.
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != Path(__file__).resolve().parent]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "ComfyUI"))

TE_PATH = REPO / "weights" / "comfyui-split" / "text_encoders" / "qwen3vl_8b_bf16.safetensors"
PROMPTS_PATH = REPO / "lab" / "prompts.json"


def tensor_hash(t) -> str:
    import torch
    a = t.detach().to("cpu", dtype=torch.float32).contiguous()
    return hashlib.sha256(f"{tuple(a.shape)}|float32|".encode() + a.numpy().tobytes()).hexdigest()


def extract_ids(tokens, key="qwen3vl_8b"):
    pairs = tokens[key][0]
    return [int(p[0]) for p in pairs if isinstance(p[0], int)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["cuda", "cpu"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--prompt-ids", default="p1-zh-scene,p2-en-scene,p3-poster-text,negative")
    args = ap.parse_args()
    import torch
    import comfy.sd
    from safetensors.torch import save_file

    if args.device == "cpu":
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(1)
    torch.manual_seed(0)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    cond_dir = out / f"cond_{args.device}"; cond_dir.mkdir(exist_ok=True)
    prompts = {p["id"]: p["prompt"] for p in json.loads(PROMPTS_PATH.read_text())["prompts"]}
    prompts["negative"] = ""
    wanted = args.prompt_ids.split(",")

    t0 = time.perf_counter()
    clip = comfy.sd.load_clip(
        ckpt_paths=[str(TE_PATH)],
        embedding_directory=str(REPO / "ComfyUI" / "models" / "embeddings"),
        clip_type=comfy.sd.CLIPType.QWEN_IMAGE,
        model_options={"load_device": torch.device(args.device),
                       "offload_device": torch.device(args.device)},
    )
    load_s = time.perf_counter() - t0
    if args.device == "cuda":
        torch.cuda.synchronize()

    from comfy.text_encoders.qwen_image21 import QwenImage21Tokenizer
    manual_tok = QwenImage21Tokenizer()

    results = {"device": args.device, "threads": args.threads,
               "torch": torch.__version__, "load_s": round(load_s, 3),
               "repeats_requested": args.repeats, "prompts": {}}

    def run_one(pid: str, rep: int) -> dict:
        text = prompts[pid]
        t0 = time.perf_counter()
        tokens = clip.tokenize(text)
        tok_s = time.perf_counter() - t0
        ids_clip = extract_ids(tokens)
        # Production-policy diagnostic (P0-C): the pinned ComfyUI runtime's
        # QwenImage21Tokenizer defaults to prevent_empty_text=False — that is
        # what clip.tokenize("") executes and what the OpenVINO encoder
        # replicates. The fallback policy (True) inserts the literal token
        # 220 ("empty") for the empty string; it is recorded for explanation
        # only (ids_manual_fallback), never used for conditioning.
        manual = manual_tok.tokenize_with_weights(
            text, llama_template=None, prevent_empty_text=False, thinking=True, keep_vision=True)
        ids_manual = extract_ids(manual)
        fallback = manual_tok.tokenize_with_weights(
            text, llama_template=None, prevent_empty_text=True, thinking=True, keep_vision=True)
        ids_fallback = extract_ids(fallback)
        if args.device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        cond_out = clip.encode_from_tokens_scheduled(tokens)
        cond = cond_out[0][0] if isinstance(cond_out, list) else cond_out
        if args.device == "cuda":
            torch.cuda.synchronize()
        enc_s = time.perf_counter() - t0
        rec = {"rep": rep, "text": text, "tokenize_s": round(tok_s, 6),
               "encode_s": round(enc_s, 6), "shape": list(cond.shape),
               "dtype": str(cond.dtype), "ids_clip": ids_clip,
               "ids_manual": ids_manual, "ids_equal": ids_clip == ids_manual,
               "ids_manual_fallback_preventTrue": ids_fallback,
               "ids_production_equal_fallback": ids_manual == ids_fallback,
               "n_tokens_full": len(ids_clip),
               "n_tokens_kept": int(cond.shape[1]),
               "cond_sha256": tensor_hash(cond),
               "finite": bool(torch.isfinite(cond).all().item())}
        save_file({"cond": cond.detach().to("cpu", dtype=torch.float32).contiguous()},
                  str(cond_dir / f"{pid}-r{rep}.safetensors"))
        if args.device == "cuda":
            rec["vram_allocated_mb"] = round(torch.cuda.memory_allocated() / 2**20, 1)
            rec["vram_reserved_mb"] = round(torch.cuda.memory_reserved() / 2**20, 1)
            rec["vram_peak_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
        return rec

    for pid in wanted:
        results["prompts"][pid] = [run_one(pid, r) for r in range(args.repeats)]
        print(json.dumps({pid: results["prompts"][pid]}, indent=1), flush=True)

    results["rss_peak_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
    results["torch_num_threads"] = torch.get_num_threads()
    results["torch_interop_threads"] = torch.get_num_interop_threads()
    results["omp_num_threads"] = os.environ.get("OMP_NUM_THREADS")
    (out / f"probe_{args.device}.json").write_text(json.dumps(results, indent=2))
    print(f"WROTE {out / f'probe_{args.device}.json'}")


if __name__ == "__main__":
    main()
