# Troubleshooting & Known Limitations (Linux W7900 Lab)

Collected from real failures during the lab build (each reproduced at least
once before the fix was trusted).

## Environment

| Symptom | Cause | Fix |
|---|---|---|
| `python -m venv` from a venv can't see torch | venv-of-venv doesn't inherit site-packages | create the lab venv from the base interpreter and add a `_rocm_base_env.pth` bridge to the validated ROCm site-packages |
| pip tries to replace torch | dependency resolution | never install over a working ROCm torch; bridge instead (P0 §4) |
| `comfyui-workflow-templates` media assets stall | ~100 MB wheels vs flaky proxy | templates+core install fine without media assets; guarded imports make them optional (server prints one ERROR line, harmless for API use) |
| 1 CPU test fails (`test_stamp_cache_cannot_mask_corruption`) | this ext4 returns identical `st_mtime_ns` for rapid writes (verified) | environment quirk, documented in P0 §2; passes on GitHub runners |

## Measurement

| Symptom | Cause | Fix |
|---|---|---|
| `vram_peak_mb: 0.0` in records | upstream sampler greps "7900" in the device name; Linux reports `AMD Radeon Graphics` | `LinuxRunMonitor` (lab/runner.py) samples any AMD device + rocm-smi device VRAM |
| `check_server` refuses a healthy W7900 | same name grep | `lab_env.check_server_linux` gates on any AMD CUDA device |
| rocm-smi utilization missing | `-g` shows clocks on this build | `rocm-smi -u --csv` for GPU use %, `--showmeminfo vram --csv` for VRAM |
| `rglob` returns nothing for existing files | ext4 loop fs readdir flakiness | use `iterdir()`/`glob()` explicitly |

## rocprofv3

| Symptom | Cause | Fix |
|---|---|---|
| No `*_kernel_trace.csv`, only `.dat` buffers in `<cwd>/.rocprofv3` | server killed with SIGTERM (or SIGINT to the app) — no finalization | **SIGINT the rocprofv3 wrapper process**; it forwards, finalizes, writes CSVs into `-d`, removes the scratch dir |
| Trace window captured no TE kernels | 90 s window opened ~1 s after gen start | TE cost measured at the pipeline layer (fresh-prompt arm); kernel-level TE via the H1 probe |
| 12 MB responses truncated | egress proxy cap on streamed responses | ranged parallel download (`lab/download_weights.py`) for any large file |

## Known limitations of the results themselves

- VRAM peaks are 0.5–1 s samples → **lower bounds** on transients (torch and
  device-level both recorded).
- H2's pipelined arm used one CPU TE worker; a multi-worker variant was not
  attempted (CPU TE was 160–221 s/prompt — more workers cannot rescue that).
- The 2048² resolution was NOT explored: the mission gates it behind stable
  1024² results, and quality review is still pending — recorded as future
  work, not as a failure.
- Windows vs Linux numbers are never compared as if equivalent (different GPU,
  attention path, stack).
- The quality gate's human verdict is **pending**; nothing in this lab may be
  cited as a quality approval until recorded.


## Radeon Cloud instances (from the 2026-11-04 live audit)

| Symptom | Cause | Fix |
|---|---|---|
| ComfyUI dies at boot: `ImportError: attempted relative import ... lab/profile.py` | `lab/` on the server's `sys.path` shadows the stdlib `profile` module (cProfile → torch._dynamo → comfy_kitchen) | already fixed in `lab/comfyui_launch.py` (strips lab/ from sys.path); `git pull` if you see it |
| `vram_device_peak_mb: 0.0` in records on ROCm 7.2.1 | rocm-smi VRAM CSV is 3-column here (4 on the reference host); old parser looked at a fixed column | fixed: runner locates the used-bytes column by its header |
| pip: `No matching distribution found for comfyui-workflow-templates-media-assets-02` | the image's default index (Tsinghua mirror) lags fresh packages | retry with `--index-url https://pypi.org/simple/` (start.sh does this automatically) |
| python `requests` to huggingface.co times out while `curl` "works" | the egress proxy blocks huggingface.co; the platform curl is a wrapper that silently rewrites to hf-mirror.com, hiding the block | use hf-mirror.com explicitly (the lab default; `--hf-base` to override) |
| `git clone` via the proxy: `CONNECT tunnel failed, response 503` | proxy rejects git's HTTP/2 tunnels intermittently | `git config --global http.version HTTP/1.1` + retry; ComfyUI falls back to the codeload tarball |
| cold run much slower than the reference (e.g. 95 s vs 74 s) | cloud persistent storage (loop-mounted) is slower at model load than the lab's local disk | expected platform difference; warm numbers are unaffected; recorded honestly in each record |
