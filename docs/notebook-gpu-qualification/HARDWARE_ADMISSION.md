# Hardware Admission — Physical W7900 Qualification

**Record:** `runs/notebook-gpu-qualification/w7900-20261106-073429/P0_HARDWARE_ADMISSION.json` ·
**Verdict: PASS** (strict doctor PASS with all six pinned weights digest-verified)

| Item | Value |
|---|---|
| GPU | AMD Radeon PRO W7900 class — `gfx1100`, 47.98 GiB VRAM, single device |
| ROCm / HIP | hip 7.2.53211-e1a6bc5663 (ROCm 7.2), driver via rocm-smi, `rocprofv3 1.1.0` |
| PyTorch | 2.9.1+gitff65f5b (ROCm build, never reinstalled) |
| OS / kernel | Ubuntu 24.04.4, Linux 6.8.0-79-generic |
| CPU | AMD EPYC 9334, host 128 cores, **cgroup quota 16 CPUs** |
| Memory | host 503 GiB, **cgroup memory.max 55.0 GiB** (the planning envelope for M5 CPU arms) |
| Disk | /workspace 98 GB; 26 GB free after staging the full 6-file pinned set (~59.7 GB) |
| OpenVINO | 2026.4.1 in isolated `.venv-ov`; IR artifacts byte-verified at `/opt/m51-artifacts` |
| ComfyUI | pinned commit `b5a432157de6…` (tree marker verified) |
| Weights | all six pinned files sha256-verified against `lab/weights.py` (2 re-hashed this session, 4 stamp-cache hits) |
| Jupyter | nbclient 0.10.4 · nbformat 5.10.4 · ipykernel 7.2.0 (real-kernel execution) |
| GPU smoke | 2048² bf16 matmul PASS, synchronized; execution proof only — no perf claim |
| Git | branch `feat/bilingual-jupyter-optimization-labs`, start HEAD `cef55be`, clean tree |

## Notes

- The two missing pinned weights (`qwen_image_2.1_bf16.safetensors` 14.23 GB, `qwen_image_2.1_int8_convrot.safetensors` 7.26 GB) were staged through the repository's pinned downloader (`lab/download_weights.py`, hf-mirror, 16-way ranged, safetensors-header verified) in 1.2 min; digests match the pinned table.
- The dedicated OpenVINO provisioning script needs ~55 GiB free for the FP32+BF16+INT8 set; this session reuses the byte-verified artifacts at `/opt/m51-artifacts` via `M51_OV_ARTIFACTS` (fp32 PASS, **bf16 PASS_WITH_WARNINGS** — the historical BF16 manifest does not pin the XML digest; distinction preserved, never rounded up; int8 PASS).
- Admission is fail-closed in the notebook runtime: wrong vendor, multi-GPU, or a busy device (>8 GB held) blocks GPU modes with student-readable guidance.
