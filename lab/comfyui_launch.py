"""ComfyUI server launcher (applies SDP env before ComfyUI imports torch).

QIR_SDP modes (diagnostic only — the normal Linux path leaves it unset):
- "math"     : disable flash + mem-efficient SDPA before ComfyUI import.
- "math-hard": same, PLUS neuter any later re-enable (needed when ComfyUI
               runs with --use-pytorch-cross-attention).
Derived from AIwork4me/qwen-image-rocm phase2-comfyui/launch_wrapper.py (MIT).
"""

import os
import runpy
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMFY_MAIN = os.path.join(REPO, "ComfyUI", "main.py")
sys.path.insert(0, os.path.dirname(COMFY_MAIN))  # for the `comfy` package

mode = os.environ.get("QIR_SDP", "")
if mode in ("math", "math-hard"):
    import torch

    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    print(f"note: math-only SDP forced (QIR_SDP={mode}) before ComfyUI import")
    if mode == "math-hard":
        _orig_flash = torch.backends.cuda.enable_flash_sdp
        _orig_mem = torch.backends.cuda.enable_mem_efficient_sdp

        def _block_reenable(name, orig, enable):
            if enable:
                print(f"note: math-hard blocked re-enable of {name} SDPA")
                return
            orig(enable)

        torch.backends.cuda.enable_flash_sdp = lambda e: _block_reenable(
            "flash", _orig_flash, e)
        torch.backends.cuda.enable_mem_efficient_sdp = lambda e: _block_reenable(
            "mem-efficient", _orig_mem, e)

sys.argv = [COMFY_MAIN] + sys.argv[1:]
runpy.run_path(COMFY_MAIN, run_name="__main__")
