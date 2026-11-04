"""Official-protocol Turbo workflow: ManualSigmas schedule (Comfy-Org template).

The Comfy-Org/Qwen-Image-2.1 official Turbo t2i template samples with
ManualSigmas "1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148,
0.704534, 0.414568, 0" + KSamplerSelect(euler) + CFGGuider(1.0). This
builder reproduces that schedule so the schedule variable is isolated
from L1 (euler/simple/8).
"""

from __future__ import annotations

import uuid


def build_workflow_manual_sigmas(
    *,
    unet: str,
    clip: str,
    vae: str,
    prompt: str,
    negative: str,
    width: int,
    height: int,
    seed: int,
    sigmas: str,
    sampler: str = "euler",
    cfg: float = 1.0,
) -> dict:
    def nid(prefix: str) -> str:
        return f"{prefix}_{uuid.uuid4().hex[:8]}"

    unet_id, clip_id, vae_id = nid("unet"), nid("clip"), nid("vae")
    pos_id, neg_id = nid("pos"), nid("neg")
    latent_id = nid("latent")
    noise_id, guider_id, ks_sel_id, sigmas_id, sampler_id = (
        nid("noise"), nid("guider"), nid("kssel"), nid("sigmas"), nid("sampler"))
    dec_id, save_id = nid("dec"), nid("save")
    return {
        unet_id: {"class_type": "UNETLoader",
                  "inputs": {"unet_name": unet, "weight_dtype": "default"}},
        clip_id: {"class_type": "CLIPLoader",
                  "inputs": {"clip_name": clip, "type": "qwen_image"}},
        vae_id: {"class_type": "VAELoader", "inputs": {"vae_name": vae}},
        pos_id: {"class_type": "CLIPTextEncode",
                 "inputs": {"text": prompt, "clip": [clip_id, 0]}},
        neg_id: {"class_type": "CLIPTextEncode",
                 "inputs": {"text": negative, "clip": [clip_id, 0]}},
        latent_id: {"class_type": "EmptyLatentImage",
                    "inputs": {"width": width, "height": height, "batch_size": 1}},
        noise_id: {"class_type": "RandomNoise",
                   "inputs": {"noise_seed": seed}},
        guider_id: {"class_type": "CFGGuider",
                    "inputs": {"model": [unet_id, 0], "positive": [pos_id, 0],
                               "negative": [neg_id, 0], "cfg": cfg}},
        ks_sel_id: {"class_type": "KSamplerSelect",
                    "inputs": {"sampler_name": sampler}},
        sigmas_id: {"class_type": "ManualSigmas",
                    "inputs": {"sigmas": sigmas}},
        sampler_id: {"class_type": "SamplerCustomAdvanced",
                     "inputs": {"noise": [noise_id, 0], "guider": [guider_id, 0],
                                "sampler": [ks_sel_id, 0], "sigmas": [sigmas_id, 0],
                                "latent_image": [latent_id, 0]}},
        dec_id: {"class_type": "VAEDecode",
                 "inputs": {"samples": [sampler_id, 0], "vae": [vae_id, 0]}},
        save_id: {"class_type": "SaveImage",
                  "inputs": {"filename_prefix": "qir_bench",
                             "images": [dec_id, 0]}},
    }
