"""Clean OpenVINO export of the Qwen3-VL-8B text encoder used by Qwen-Image-2.1.

Source checkpoint: the course's pinned ComfyUI-format qwen3vl_8b_bf16.safetensors
(keys are identical to transformers Qwen3VLForConditionalGeneration). Only the
36-layer language model is exported; token embedding is done at runtime from the
same verified checkpoint (bit-exact dequant of the BF16 table), so token IDs are
identical to ComfyUI's product tokenizer.

Output: layer-36 residual stream BEFORE final RMSNorm, shape [1, seq, 4096].
This is the Qwen-Image conditioning contract (layer_idx=-1,
layer_norm_hidden_state=False), not logits.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import resource
import time
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

REPO = Path(__file__).resolve().parents[1]
TE_PATH = REPO / "weights" / "comfyui-split" / "text_encoders" / "qwen3vl_8b_bf16.safetensors"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 23), b""):
            h.update(b)
    return h.hexdigest()


class TextOnly(torch.nn.Module):
    def __init__(self, lm):
        super().__init__()
        self.lm = lm

    def forward(self, inputs_embeds, attention_mask, position_ids):
        captured = {}

        def norm_hook(module, args, kwargs, output):
            captured["prenorm"] = args[0]
            return args[0]  # neutralize final RMSNorm (official Qwen-Image semantics)

        h = self.lm.norm.register_forward_hook(norm_hook, with_kwargs=True)
        try:
            self.lm(inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                    position_ids=position_ids, output_hidden_states=False,
                    use_cache=False)
        finally:
            h.remove()
        assert "prenorm" in captured
        return captured["prenorm"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-seq", type=int, default=2048)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    cfg_dict = json.loads(Path(args.config).read_text())
    from transformers import Qwen3VLTextConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel
    cfg = Qwen3VLTextConfig.from_dict(cfg_dict["text_config"])
    cfg._attn_implementation = "eager"

    # NOTE: must construct on CPU, not meta/to_empty: RoPE's non-persistent
    # inv_freq buffer is computed in __init__ and is NOT in the checkpoint.
    # to_empty() would leave it uninitialized and corrupt every layer.
    lm = Qwen3VLTextModel(cfg).eval()
    lm.embed_tokens = None  # unused: we always feed inputs_embeds; saves 2.5 GB fp32
    inv = dict(lm.named_buffers()).get("rotary_emb.inv_freq")
    if inv is None or not torch.isfinite(inv).all():
        raise RuntimeError("rotary inv_freq buffer invalid after construction")
    print(f"constructed on CPU in {time.time()-t0:.1f}s; rss={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20:.1f} GiB", flush=True)

    params = dict(lm.named_parameters())
    buffers = dict(lm.named_buffers())
    missing, loaded, covered = [], 0, set()
    t0 = time.time()
    with safe_open(str(TE_PATH), framework="pt", device="cpu") as sf:
        keys = list(sf.keys())
        # ComfyUI's safetensors use the bare transformers text-model names with a
        # leading "model." (model.layers.*, model.norm.*), not "model.language_model.*".
        expected = [k for k in keys if k.startswith("model.")
                    and not k.startswith(("model.embed_tokens.", "model.visual.",
                                          "model.lm_head."))]
        if not expected:
            raise RuntimeError("no text-model tensors matched the checkpoint")
        for k in expected:
            name = k[len("model."):]
            target = params.get(name)
            if target is None:
                target = buffers.get(name)
            if target is None:
                missing.append(name); continue
            val = sf.get_tensor(k).to(torch.float32)
            if tuple(val.shape) != tuple(target.shape):
                raise RuntimeError(f"shape mismatch {k}: {val.shape} vs {target.shape}")
            with torch.no_grad():
                target.copy_(val)
            covered.add(name)
            loaded += 1
    if missing:
        raise RuntimeError(f"unmapped checkpoint keys: {missing[:10]} (n={len(missing)})")
    uncovered = sorted(set(params) - covered)
    if uncovered:
        raise RuntimeError(f"parameters not loaded from checkpoint: {uncovered[:10]} (n={len(uncovered)})")
    if loaded < 390:
        raise RuntimeError(f"implausibly few tensors loaded: {loaded}")
    print(f"loaded {loaded} tensors in {time.time()-t0:.1f}s; rss={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20:.1f} GiB", flush=True)

    # sanity forward on a short sequence
    lm.eval()
    wrapped = TextOnly(lm).eval()
    n = 16
    emb = torch.randn(1, n, cfg.hidden_size)
    att = torch.ones(1, n, dtype=torch.long)
    ar = torch.arange(n, dtype=torch.long)
    pos = torch.stack([ar, ar, ar]).reshape(3, 1, n)
    with torch.no_grad():
        ref = wrapped(emb, att, pos)
    print("eager output", tuple(ref.shape), flush=True)

    # transformers 4.57's vmap-based masking helper is not TorchScript-traceable.
    # Inject the mathematically equivalent full-sequence causal mask.
    import transformers.models.qwen3_vl.modeling_qwen3_vl as _qv

    def _simple_causal_mask(config, input_embeds, attention_mask, cache_position,
                            past_key_values, position_ids):
        b, s = input_embeds.shape[:2]
        q = cache_position if cache_position is not None else torch.arange(s)
        kv = torch.arange(s)
        keep = (kv[None, :] <= q[:, None])[None, None, :, :]
        if attention_mask is not None:
            keep = keep & attention_mask[:, None, None, :].bool()
        out = torch.zeros((b, 1, s, s), dtype=input_embeds.dtype)
        return out.masked_fill(~keep, torch.finfo(input_embeds.dtype).min)

    _qv.create_causal_mask = _simple_causal_mask

    import openvino as ov
    dim = ov.Dimension(1, args.max_seq)
    t0 = time.time()
    with torch.no_grad():
        ov_model = ov.convert_model(
            wrapped, example_input=(emb, att, pos),
            input=[(1, dim, cfg.hidden_size), (1, dim), (3, 1, dim)])
    print(f"converted in {time.time()-t0:.1f}s; rss={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20:.1f} GiB", flush=True)

    xml = out / "language_model_fp32.xml"
    t0 = time.time()
    ov.save_model(ov_model, str(xml), compress_to_fp16=False)
    print(f"saved {xml} ({xml.stat().st_size/1e9:.2f} GB) in {time.time()-t0:.1f}s", flush=True)

    sha = lambda path: __import__("hashlib").sha256(Path(path).read_bytes()).hexdigest()
    binp = xml.with_suffix(".bin")
    manifest = {
        "source_safetensors": str(TE_PATH.relative_to(REPO)),
        "source_sha256": sha256(TE_PATH),
        "source_size_bytes": TE_PATH.stat().st_size,
        "config_source": "Qwen/Qwen-Image-2.1 text_encoder/config.json (HF snapshot), text_config only",
        "architecture": "Qwen3VLTextModel, 36 layers, hidden 4096, pre-final-RMSNorm residual",
        "output": "layer-36 residual before final RMSNorm [1,seq,4096]",
        "openvino": ov.__version__,
        "dtype": "fp32",
        "compress_to_fp16": False,
        "seq_dim": f"dynamic [1,{args.max_seq}]",
        "tensors_loaded": loaded,
        "embed_tokens": "excluded from graph; runtime BF16->FP32 gather from same checkpoint",
        "rss_peak_gib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2),
        "output_xml_sha256": sha(xml), "output_bin_sha256": sha(binp),
        "output_bin_bytes": binp.stat().st_size, "output_gib": round(binp.stat().st_size / 2**30, 3),
        "tool": "openvino.convert_model (PyTorch frontend), eager attention",
    }
    (out / "conversion_manifest.json").write_text(json.dumps(manifest, indent=2))
    print("DONE", json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
