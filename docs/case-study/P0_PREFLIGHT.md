# P0 — Repository Audit & Hardware Preflight (Linux W7900)

**Date:** 2026-11-03 · **Branch:** `feat/linux-w7900-optimization-lab`
**Upstream main imported at:** `acc318fc593c1be331f81de1a03a9ce15cffd314` (2026-11-01)
**Verdict: PASS** (all required execution conditions satisfied; GitHub push/PR blocked by
egress proxy + missing credentials — see §5)

## 1. Machine & software inventory (measured, not assumed)

| Item | Value | Evidence |
|---|---|---|
| OS | Ubuntu 24.04 family, kernel 6.8.0-79-generic x86_64 | `evidence/p0/uname.txt` |
| CPU | AMD EPYC 9334 32-Core (host, 128 threads visible to container) | `evidence/p0/rocminfo.txt` |
| Host RAM | 503 GiB total; container cgroup limit **55 GiB** (ComfyUI sees 56320 MB) | `evidence/p0/free.txt`, server log |
| Swap | none configured | `evidence/p0/free.txt` |
| GPU | **single AMD Radeon PRO W7900, gfx1100, 48 GB** (51522830336 B, torch reports 49136 MB) — 8 GPUs on host, exactly 1 exposed to this container | `evidence/p0/rocminfo.txt`, `rocm-smi.txt` |
| GPU PCIe | PCIe 4.0 x16 (16.0 GT/s, 16 lanes negotiated = max) at 0000:23:00.0 | sysfs `current_link_speed/width` |
| ROCm userspace | 7.2.1 (`/opt/rocm-7.2.1`), driver 6.16.13, HIP runtime 7.2.53211 | `rocminfo`, `rocm-smi` |
| PyTorch | **2.9.1+gitff65f5b (rocm7.2.1)** — pre-validated in the base env; `torch.cuda.is_available() == True`, device `gfx1100` | `evidence/p0/pip-freeze-lab-venv.txt` |
| Profilers | rocprofv3 1.1.0 (+rocprof v2, rocprof-sys, rocprof-compute) available | `evidence/p0/profilers.txt` |
| Disk | /workspace: 98 GB loop ext4, 72+ GB free at audit time | `evidence/p0/df.txt` |

**Requested-but-unavailable:** ROCm 7.14.1 (the Windows stack) does not exist on this
Linux host; the installed **ROCm 7.2.1 + torch 2.9.1+rocm7.2.1** stack is used instead —
it is the pre-installed, GPU-validated environment and was NOT modified (source-protection
rule 7). No PyTorch reinstallation was performed.

## 2. Repository audit

- Imported upstream `main` @ `acc318fc…` byte-complete (716 blobs, 108,006,488 B,
  verified size-exact against the GitHub tree API) into local branch `import-main`;
  work branch `feat/linux-w7900-optimization-lab` created from it.
  **git clone/push is blocked by the lab egress proxy** (CONNECT 503 on github.com);
  the repo was reconstructed via the codeload/jsDelivr mirrors with per-file size
  verification. Historical Windows results (`bench/results/2026-*`) are untouched.
- **Reusable on Linux as-is:** `bench/core.py` (protocol planning, schema),
  `bench/comfyui_client.py` (workflow builder, ws monitor, HTTP client),
  `bench/prompts.json` (frozen 3-prompt × 3-seed set), `bench/schema/result.schema.json`,
  `lab/weights.py` (pinned weight digests), `lab/download_weights.py`,
  `tools/snapshot_env.py` (pattern), `lab/comfyui_launch.py` + `extra_model_paths.yaml`.
- **Windows-specific, isolated (not modified):** `quickstart.ps1`, `bench/run_variant.ps1`,
  `bench/portops.ps1`, PowerShell lifecycle; `check_server()` grepping `"7900"` in the
  device name (Linux reports `cuda:0 AMD Radeon Graphics : native` → gate reimplemented
  in `lab/lab_env.py`); the VRAM sampler inside `RunMonitor` (same "7900"
  grep → subclassed in `lab/runner.py`).
- **ComfyUI:** pinned commit `b5a432157de6b8884f81104cd02926adf4d597e4` (v0.43.0)
  fetched via mirror, all 1329 files size-verified vs the tree API; marker
  `ComfyUI/.linux_lab_pinned_commit` (git HEAD check impossible without
  git transport; file-tree verification is the recorded evidence).
- **CPU test gate:** `pytest tests/` → **101 passed, 1 failed**. The failure
  (`test_quickstart.py::test_stamp_cache_cannot_mask_corruption`) is an environment
  artifact: this ext4 loop device returns **identical `st_mtime_ns` for rapid
  consecutive writes** (verified: two different writes → same mtime_ns), so the
  stamp-cache test's premise (new mtime on rewrite) cannot hold here. Not a code
  regression; owner's CI on GitHub runners passes. Recorded as a known lab limitation.
- **Historical manifests:** unchanged (`git status` clean on import for `bench/experiments/`).

## 3. Weights (all SHA256-verified against in-repo pinned digests)

Downloaded from `Comfy-Org/Qwen-Image-2.1` @ pinned revision `aacf31a6` (hf-mirror
transport, 16-way ranged parallel; integrity anchored to `lab/weights.py`
pins, never to the transport):

| File | Size | sha256 vs pin |
|---|---|---|
| `diffusion_models/qwen_image_2.1_bf16.safetensors` | 14.23 GB | OK |
| `diffusion_models/qwen_image_2.1_turbo_bf16.safetensors` | 14.23 GB | OK |
| `diffusion_models/qwen_image_2.1_int8_convrot.safetensors` | 7.26 GB | OK |
| `diffusion_models/qwen_image_2.1_turbo_int8_convrot.safetensors` | 7.26 GB | OK |
| `text_encoders/qwen3vl_8b_bf16.safetensors` | 17.53 GB | OK |
| `vae/qwen_image_2.1_vae_bf16.safetensors` | 0.68 GB | OK |

## 4. Lab environment (separate; base env untouched)

- `python3 -m venv .venv-lab` + `_rocm_base_env.pth` bridging read-only to the
  pre-validated base site-packages (torch 2.9.1+rocm7.2.1). All new installs
  (ComfyUI requirements) land in `.venv-lab`; `/opt/venv` never written.
- One deviation: `comfyui-workflow-templates` media-asset subpackages (~100 MB wheels)
  could not be pulled through the proxy (pypi.org stream stalls); the main template
  package + core installed. Harmless for headless API benchmarking (guarded imports).
- **First boot on W7900:** PASS — `Device: cuda:0 AMD Radeon Graphics : native`,
  `AMD arch: gfx1100`, `ROCm version: (7, 2)`, cudnn disabled (AMD perf default),
  **`Using pytorch attention`** (ComfyUI's aotriton flash probe PASSED on Linux —
  the Windows 7.14.1 AOTriton breakage does not reproduce here). Evidence:
  `lab/evidence/p0/comfyui-first-boot.log`.
- **First generation (smoke, 4 steps):** PASS — TE 1.8 s, ~1.35 s/step BF16,
  VAE 1.7 s, valid PNG. `evidence/p0/smoke-first-generation.png`.
- **CLI doctor:** all 7 checks PASS (GPU gate, torch/ROCm, pinned ComfyUI, weight
  digests, flash SDPA probe, rocprofv3) — `evidence/results/doctor.json`.

## 5. Resource limits & blockers

| Limit | Impact |
|---|---|
| Container RAM 55 GiB (host 503 GiB) | CPU text-encoder residency (H1) feasible but bounded; monitored |
| No swap | OOM = hard kill; experiments stop on first repeated OOM per discipline |
| Egress proxy: github.com git endpoints blocked; ~12 MB response cap on codeload; pypi.org stalls on large files | **GitHub push + Draft PR = BLOCKED** (no credentials either); weights fetched via ranged parallel download through hf-mirror |
| `ss`/`lspci` -vv LnkSta limited in container | PCIe width verified via sysfs instead |
| ext4 mtime_ns granularity quirk | 1 pre-existing CPU test fails here (see §2); stamp-cache caveat noted |

## 6. Installation strategy (for reproduction)

The lab adds **no OS-level changes**. Reproduction on a fresh W7900 Linux box:
1. ROCm 7.x + a torch `+rocm` build with working `torch.cuda` (validated here: 2.9.1).
2. Clone this repo; `python -m venv .venv-lab` bridged to the torch env
   (or install ComfyUI requirements directly into a dedicated venv).
3. Fetch ComfyUI @ `b5a4321` and weights @ `aacf31a6` (pinned digests in
   `lab/weights.py`; `lab/download_weights.py` for hostile networks).
4. `python -m lab doctor` must PASS before any measurement.

**Gate P0: PASS — proceed to Phase 1.**
