"""Lab-owned ComfyUI node: load precomputed conditioning from safetensors.

Safe alternative to torch.load for the M5.1 bridge. Validates the tensor
contract before exposing it to the DiT: key 'cond', shape [1,seq,4096] (or
[seq,4096]), float32, finite. No pickle, no code execution.
"""
import torch
from safetensors.torch import load_file


class LinuxLabCondSafetensors:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"cond_path": ("STRING", {"default": ""})}}

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "load"
    CATEGORY = "tuning_lab"

    def load(self, cond_path):
        blob = load_file(cond_path, device="cpu")
        if "cond" not in blob:
            raise ValueError(f"{cond_path}: missing 'cond' tensor")
        cond = blob["cond"].to(torch.float32)
        if cond.ndim == 2:
            cond = cond.unsqueeze(0)
        if cond.ndim != 3 or cond.shape[0] != 1 or cond.shape[2] != 4096:
            raise ValueError(f"invalid conditioning shape {tuple(cond.shape)}")
        if not torch.isfinite(cond).all():
            raise ValueError("conditioning contains NaN/Inf")
        if torch.cuda.is_available():
            cond = cond.to("cuda")
        return ([[cond, {}]],)


NODE_CLASS_MAPPINGS = {"LinuxLabCondSafetensors": LinuxLabCondSafetensors}
NODE_DISPLAY_NAME_MAPPINGS = {"LinuxLabCondSafetensors": "Tuning Lab Cond Loader (safetensors)"}
