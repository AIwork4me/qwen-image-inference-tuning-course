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
