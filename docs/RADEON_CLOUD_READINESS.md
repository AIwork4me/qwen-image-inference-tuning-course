# Radeon Cloud Readiness Audit — P0

**Status: READY (validated on a live instance, 2026-11-04).**
This document records what was physically measured on a running Radeon
Cloud Linux Notebook instance — not inferred from documentation. Re-run
`bash scripts/cloud/start.sh` on YOUR instance to regenerate every check
below for your own hardware (the script prints the same report and writes
`runs/doctor.json`).

> Application (free GPU compute time): see the course README —
> platform <https://developer.amd.com.cn/radeon>, docs
> <https://amd-aim.github.io/radeon-cloud-docs/zh-cn/>.
> **Never assume** a fixed number of free hours, GPU models, mounted
> models or storage size: quotas and SKUs change. Everything below is what
> one live instance reported on one day; verify yours at first boot.

---

## 1. The one recommended student journey

```
apply for Radeon Cloud GPU time
  -> start a single-GPU Linux Notebook instance (ROCm + PyTorch image)
  -> clone this repo into /workspace
  -> bash scripts/cloud/start.sh
  -> M00..M07 in order
  -> evidence-backed results
  -> submit a Pull Request
```

There is exactly ONE course. Students without a GPU read the shipped
evidence (all modules have evidence-reading exercises); there is no
separate "no-GPU track".

## 2. Audited instance (2026-11-04, live)

| Item | Measured value | Course requirement | Verdict |
|---|---|---|---|
| GPU | AMD Radeon Graphics, gfx1100, 48.0 GiB VRAM (49 136 MB), 48 CUs, PCI device `0x744b` | single Radeon gfx1100-class, ≥ ~31 GB for int8 turbo / ≥ ~38 GB for bf16 baseline | **READY** |
| ROCm | 7.2.1 (`/opt/rocm/.info/version`), driver per `rocm-smi` | any ROCm with a working HIP torch | **READY** |
| PyTorch | 2.9.1+gitff65f5b, hip 7.2.53211-e1a6bc5663, `cuda.is_available()=True` | ROCm build; matches the validated reference stack (2.9.1 / ROCm 7.2.1) | **READY** |
| Python | 3.12.3 (`/opt/venv`) | 3.12 (CI parity) | **READY** |
| GPU visibility | 1 device via torch; `rocminfo` gfx1100; util/VRAM via `rocm-smi` OK | exactly 1 GPU | **READY** |
| CPU (container cgroup) | 16 cores (`cpu.max = 1600000/100000`; host has 128) | ≥ 4 for server + monitor | **READY** |
| RAM (container cgroup) | 55 GiB (`memory.max ≈ 59 GB`; host 503 GiB) | ≥ 32 GiB comfortable | **READY** |
| Persistent `/workspace` | 98 GiB ext4 loop volume (`/rgapi-one-click-*.img`), **69 GiB free at audit time** | ≥ ~28 GiB for the staged set; ~63 GiB for all six models | **READY (staged)** |
| Temporary storage | `/tmp` tmpfs 4 GiB; overlay `/` 3.5 TiB (shared, do not rely on it) | scratch during downloads | READY |
| ComfyUI | **not preinstalled** — fetched at pinned commit by `start.sh` | pinned `b5a4321` (v0.43.0) | **OK via script** |
| `rocprofv3` | present, v1.1.0 (`/opt/rocm/bin/rocprofv3`) | required for M2 kernel profiling | **READY** |
| `/models` platform mount | **none** (`/models` absent) | optional convenience | N/A |
| Pinned model files | **none present locally**; hf-mirror reachable at ~0.4 GB/s aggregate | staged download (see §4) | **OK via script** |

### GPU identification caveat (W7900 vs W7900D)

In-container tools (`torch`, `rocm-smi`, `lspci`) report gfx1100 / 48 GB /
device id `0x744b` — the W7900(-class) profile. The China-market W7900D
shares this identifier set and the same gfx1100/48GB envelope, so **the
two cannot be distinguished from inside the container** and are treated as
one hardware class here. Any different Radeon (e.g. W7800 32 GB, RX 7900
XTX 24 GB, MI-class) IS detectable (arch/VRAM differ) and gets an explicit
doctor WARN in cloud mode: the methodology applies unchanged, but the
numbers are an independent measurement, not a reproduction of the frozen
reference. The 32/24 GB cards cannot fit the bf16 baseline (~37.7 GB peak)
— use the int8/turbo experiments there.

## 3. Network reality on the audited instance (what start.sh does about it)

| Destination | Direct (python/requests) | Notes |
|---|---|---|
| `hf-mirror.com` | **allowed** (CONNECT 200, ~0.4 GB/s aggregate with 16-way ranged download) | default model source |
| `huggingface.co` | **blocked** (CONNECT silently dropped) | the platform `curl` is a wrapper that silently rewrites HF→hf-mirror, which HIDES the block — the lab always uses the mirror explicitly |
| `github.com` (git smart-HTTP) | flaky (intermittent CONNECT 503); works with `http.version=HTTP/1.1` + retries | repo clone OK (retry loop), ComfyUI clone falls back to a codeload tarball |
| `codeload.github.com` | **allowed** (200) | ComfyUI tarball fallback path |
| `api.github.com` | allowed | `gh` CLI works |
| `pypi.org` / `files.pythonhosted.org` | allowed | venv pip works |
| `pypi.tuna.tsinghua.edu.cn` (default index on the image) | allowed but **lags new packages** | start.sh falls back to official PyPI on failure |

**Consequence for students:** none of this needs manual handling —
`start.sh` encodes every workaround (mirror default, `--hf-base` override,
tarball fallback, pip index fallback, HTTP/1.1 for git).

## 4. Model provisioning strategy (storage math)

The complete six-file pinned set is **~61.2 GB**; a default small
persistent workspace cannot be assumed to hold it. The course therefore
stages models — a student is never forced to obtain every variant before
the first image:

| Stage | Files | Cumulative | Unlocks |
|---|---|---|---|
| 1 (default in `start.sh`) | turbo-int8 DiT (7.26 GB) + Qwen3-VL 8B TE (17.53 GB) + VAE (0.68 GB) | **25.5 GB** | first smoke image, BEST, LF2, LF3 |
| 2 | + bf16 DiT (14.23 GB) | 39.7 GB | M01 baseline B0, F1 |
| 3 | + int8 DiT (7.26 GB) | 47.0 GB | F2 |
| 4 | + turbo-bf16 DiT (14.23 GB) | 61.2 GB | L1, L1O, LF1, H1, H2 (all six) |

All files are SHA256-pinned in `lab/weights.py` (trust anchor is the
in-repo table, never a live endpoint). Downloads are 16-way ranged and
resumable (`lab/download_weights.py`); `start.sh --all-models` fetches
everything behind a disk-budget guard.

**Precedence (as required by the platform rules):**
1. read-only platform model mounts (`/models`-style) — **none existed on
   the audited instance**; if your instance ships one, point
   `extra_model_paths.yaml` at it and skip downloading (verify digests
   with `python -m lab doctor --cloud`);
2. supported large persistent storage (`/workspace`) — the validated path
   used by `start.sh`;
3. if neither can hold the staged set, the script **fails with the exact
   blocker** (free GiB vs needed GiB) and the actionable provisioning
   request: attach/expand the persistent volume, or select an instance
   flavor with ≥ 30 GB workspace for the staged path (≥ 65 GB for all
   six models).

## 5. Tooling deltas found on the audited instance (fixed in this branch)

1. `lab/` on the server's `sys.path` **shadowed the stdlib `profile`
   module** (cProfile → torch._dynamo → comfy_kitchen import chain),
   crashing ComfyUI at boot — fixed in `comfyui_launch.py`.
2. `rocm-smi --showmeminfo vram --csv` emits a **3-column** table on ROCm
   7.2.1 (the reference host emitted 4) — the device-VRAM sampler now
   locates the used-bytes column by header.
3. `PIP_INDEX_URL` on the image points at the Tsinghua mirror, which
   lacked a freshly-released ComfyUI dependency — start.sh retries with
   the official PyPI.
4. The platform `curl` binary silently rewrites `huggingface.co` to
   `hf-mirror.com`, masking the real egress policy — the lab uses
   `hf-mirror.com` explicitly so behavior is identical with or without
   the wrapper.

## 6. Residual risks / honest limitations

- **Instance restarts:** `/workspace` (repo, weights, venv, runs) persists;
  anything outside it (e.g. `/tmp`) does not. `start.sh` is idempotent —
  after a restart it verifies and repairs instead of re-downloading.
- **GPU minutes are metered:** stop or destroy instances you are not using
  (see the QUICKSTART budget table). The very slow H1/H2 CPU experiments
  are opt-in research tasks, never the default path.
- **The audited quotas (hours, credits, storage) are NOT documented here**
  because they are account- and time-dependent. Read them from the
  platform console before provisioning.
- **Non-W7900-class GPUs** produce honest WARN lines (never silent
  substitution) and their records remain valid independent measurements —
  but the frozen reference comparison in M01–M06 only applies to
  gfx1100/48 GB hosts.
