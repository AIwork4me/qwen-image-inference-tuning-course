"""Persistent OpenVINO CPU text encoder for the Qwen-Image-2.1 conditioning path.

Contract (verified against ComfyUI's QwenImage21TEModel):
  - tokenization: ComfyUI QwenImage21Tokenizer (T2I system prefix, no think block)
  - embeddings: BF16->FP32 gather from the pinned checkpoint (no IR embed op)
  - graph output: layer-36 residual stream BEFORE final RMSNorm
  - trim at the second <|im_start|>; output [1, kept, 4096] FP32
No logits, no post-norm, no generic chat template.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
_HERE = Path(__file__).resolve().parent
sys.path[:] = [p for p in sys.path
               if Path(p or ".").resolve() != _HERE]
TE_PATH = REPO / "weights" / "comfyui-split" / "text_encoders" / "qwen3vl_8b_bf16.safetensors"
COMFY = REPO / "ComfyUI"
if str(COMFY) not in sys.path:
    sys.path.insert(0, str(COMFY))

IM_START = 151644


class OVTextEncoder:
    def __init__(self, model_dir: str | Path, precision: str = "f32",
                 threads: int = 16, cache_dir: str | Path | None = None,
                 performance_hint: str = "LATENCY", model_xml: str = "language_model_fp32.xml",
                 dynamic_quant_group: int | None = None):
        import openvino as ov
        from comfy.text_encoders.qwen_image21 import QwenImage21Tokenizer
        from safetensors import safe_open

        self.ov = ov
        self.model_dir = str(model_dir)
        self.precision = precision
        self.threads = threads
        self.dynamic_quant_group = dynamic_quant_group
        t0 = time.perf_counter()
        self.tokenizer = QwenImage21Tokenizer()
        self.tokenizer_load_s = time.perf_counter() - t0

        core = ov.Core()
        if cache_dir:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
            core.set_property({"CACHE_DIR": str(cache_dir)})
        self.model_xml = model_xml
        model = core.read_model(str(Path(model_dir) / model_xml))
        self.read_s = time.perf_counter() - t0
        props = {
            "PERFORMANCE_HINT": performance_hint,
            "INFERENCE_NUM_THREADS": threads,
            "NUM_STREAMS": 1,
            "INFERENCE_PRECISION_HINT": precision,
            "EXECUTION_MODE_HINT": "PERFORMANCE",
        }
        if dynamic_quant_group is not None:
            props["DYNAMIC_QUANTIZATION_GROUP_SIZE"] = dynamic_quant_group
        t0 = time.perf_counter()
        self.compiled = core.compile_model(model, "CPU", props)
        self.compile_s = time.perf_counter() - t0
        devices = [str(d) for d in self.compiled.get_property("EXECUTION_DEVICES")]
        if devices != ["CPU"]:
            raise RuntimeError(f"OpenVINO encoder must execute on CPU, got {devices}")
        t0 = time.perf_counter()
        with safe_open(str(TE_PATH), framework="pt", device="cpu") as sf:
            table = sf.get_tensor("model.embed_tokens.weight").float().numpy()
        self.embed_table = table
        self.embed_load_s = time.perf_counter() - t0
        self.input_slots = {}
        for i, inp in enumerate(self.compiled.inputs):
            rank = len(inp.get_partial_shape())
            et = inp.get_element_type().get_type_name()
            names = set(inp.names)
            if rank == 3 and et == "f32":
                self.input_slots[i] = "inputs_embeds"
            elif rank == 2 and et in ("i64", "i32"):
                self.input_slots[i] = "attention_mask"
            elif rank == 3 and et in ("i64", "i32"):
                self.input_slots[i] = "position_ids"
            else:
                raise RuntimeError(f"unrecognized IR input {i}: {names} {et} {inp.get_partial_shape()}")
        if sorted(self.input_slots.values()) != ["attention_mask", "inputs_embeds", "position_ids"]:
            raise RuntimeError(f"unexpected IR inputs: {self.input_slots}")
        self.req = self.compiled.create_infer_request()

    def token_ids(self, text: str) -> list[int]:
        toks = self.tokenizer.tokenize_with_weights(
            text, llama_template=None, prevent_empty_text=False, thinking=True, keep_vision=True)
        return [int(t[0]) for t in toks["qwen3vl_8b"][0] if isinstance(t[0], int)]

    def encode(self, text: str, synchronize=None) -> dict:
        ids = self.token_ids(text)
        n = len(ids)
        t0 = time.perf_counter()
        embeds = self.embed_table[np.asarray(ids, dtype=np.int64)][None, :, :]
        ar = np.arange(n, dtype=np.int64)
        pos = np.stack([ar, ar, ar]).reshape(3, 1, n)
        feed = {
            "inputs_embeds": embeds.astype(np.float32),
            "attention_mask": np.ones((1, n), dtype=np.int64),
            "position_ids": pos,
        }
        by_slot = {i: feed[name] for i, name in self.input_slots.items()}
        if synchronize is not None:
            synchronize()
        t0 = time.perf_counter()
        self.req.reset_state()
        res = self.req.infer(by_slot)
        lat = time.perf_counter() - t0
        hidden = np.asarray(res[list(res.keys())[-1]], dtype=np.float32)
        if hidden.shape != (1, n, 4096):
            raise RuntimeError(f"unexpected output shape {hidden.shape} for {n} tokens")
        starts = [i for i, t in enumerate(ids) if t == IM_START]
        trim = starts[1] if len(starts) > 1 else 0
        kept = hidden[:, trim:, :]
        return {"hidden": kept, "ids": ids, "n_tokens_full": n,
                "n_tokens_kept": int(kept.shape[1]), "encode_s": lat,
                "trim_index": trim}

    def meta(self) -> dict:
        def prop(name):
            try:
                return str(self.compiled.get_property(name))
            except Exception:
                return None
        return {"openvino": self.ov.__version__, "device": "CPU",
                "execution_devices": [str(d) for d in self.compiled.get_property("EXECUTION_DEVICES")],
                "inference_precision": prop("INFERENCE_PRECISION_HINT"),
                "performance_hint": prop("PERFORMANCE_HINT"),
                "execution_mode": prop("EXECUTION_MODE_HINT"),
                "threads": self.threads, "num_streams": 1,
                "read_s": round(self.read_s, 3), "compile_s": round(self.compile_s, 3),
                "embed_table_load_s": round(self.embed_load_s, 3),
                "tokenizer_load_s": round(self.tokenizer_load_s, 3),
                "model_dir": self.model_dir, "model_xml": self.model_xml,
                "dynamic_quant_group": getattr(self, "dynamic_quant_group", None)}


def _main() -> None:
    import argparse
    from safetensors.torch import save_file

    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--precision", choices=["f32", "bf16"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--xml", default="language_model_fp32.xml")
    ap.add_argument("--dynamic-quant-group", type=int, default=None)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--prompt-ids", default="p1-zh-scene,p2-en-scene,p3-poster-text,negative")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    cond_dir = out / "cond"; cond_dir.mkdir(exist_ok=True)
    prompts = {p["id"]: p["prompt"] for p in json.loads((REPO / "lab" / "prompts.json").read_text())["prompts"]}
    prompts["negative"] = ""

    t0 = time.perf_counter()
    enc = OVTextEncoder(args.model_dir, precision=args.precision, threads=args.threads,
                        cache_dir=None, model_xml=args.xml,
                        dynamic_quant_group=args.dynamic_quant_group)
    load_s = time.perf_counter() - t0
    res = {"arm": args.arm, "precision": args.precision, "load_s": round(load_s, 3),
           "meta": enc.meta(), "prompts": {}}
    for pid in args.prompt_ids.split(","):
        rows = []
        for r in range(args.repeats):
            o = enc.encode(prompts[pid])
            cond = o["hidden"][0]
            fn = cond_dir / f"{pid}-r{r}.safetensors"
            save_file({"cond": __import__("torch").from_numpy(cond.copy()).contiguous()}, str(fn))
            rows.append({"rep": r, "encode_s": round(o["encode_s"], 6),
                         "shape": list(o["hidden"].shape), "ids": o["ids"],
                         "n_tokens_full": o["n_tokens_full"],
                         "n_tokens_kept": o["n_tokens_kept"],
                         "trim_index": o["trim_index"],
                         "finite": bool(np.isfinite(o["hidden"]).all()),
                         "cond_sha256": __import__("hashlib").sha256(
                             f"{tuple(cond.shape)}|float32|".encode() + cond.tobytes()).hexdigest(),
                         "cond_path": str(fn)})
        res["prompts"][pid] = rows
        print(json.dumps({pid: rows}, indent=1), flush=True)
    import resource
    res["rss_peak_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
    (out / f"probe_{args.arm}.json").write_text(json.dumps(res, indent=2))
    print(f"WROTE {out / f'probe_{args.arm}.json'}")


if __name__ == "__main__":
    _main()
