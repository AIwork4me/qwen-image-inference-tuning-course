# M5.1 Environment and Admission

**Run:** `m51-20261105-cleanroom-01` · **Admission:** `GO_WITH_DEVIATIONS` · **Snapshot:** `runs/m51-20261105-cleanroom-01/environment.json`

## 1. Host

| Item | Measured |
|---|---|
| OS | Ubuntu 24.04.4 LTS, kernel 6.8.0-79-generic |
| CPU | AMD EPYC 9334 32-Core, 2 sockets × 32 physical cores, 128 logical threads |
| ISA | AVX2, AVX-512F, AVX-512 BF16, AVX-512 VNNI present |
| NUMA | 2 nodes |
| Host RAM | 503 GiB total |
| **Container cgroup** | **16 CPU quota** (`1600000 100000`), **55 GiB memory max** |
| GPU | gfx1100, 48 GiB VRAM, VBIOS `113-D7070910-100`, SKU `D7070910` |
| GPU string | `AMD Radeon Graphics` (driver string; not literally “Radeon PRO W7900”) |
| ROCm userspace | 7.2.1 |
| PyTorch | 2.9.1+gitff65f5b, HIP 7.2.53211-e1a6bc5663, GPU available |
| ComfyUI | v0.43.0 pinned `b5a432157de6b8884f81104cd02926adf4d597e4` |
| OpenVINO | 2026.4.1-22982-e213a147257 (isolated `.venv-ov`) |
| NNCF | 3.4.0 |
| Transformers | 4.57.6 |
| Python | 3.12.3 |
| Git HEAD at measurement | `20a7f7e5afcb078b3899470365464f4922db10cb` |
| Workspace | 98 GB loop mount; **47 GB free** after model/IR staging |
| `/tmp` | 4 GB tmpfs; all scratch redirected to `/workspace/m51/tmp` or `/opt/m51-artifacts` |

## 2. Model artifacts (SHA256 verified)

| Artifact | Bytes | SHA256 |
|---|---:|---|
| `qwen3vl_8b_bf16.safetensors` | 17,534,334,616 | `68bdc82bc1b66851162ae656225e7e2068166b603db19bd5d5a3b90eb12669a9` |
| `qwen_image_2.1_turbo_bf16.safetensors` | 14,230,280,616 | `8b74237a7f505773e2c66726fa696444b4c704585c93a347cff9044937f7914a` |
| `qwen_image_2.1_turbo_int8_convrot.safetensors` | 7,256,787,829 | `8144a0182964aa8403ad6dbc011db837304167d235161b9aabaa8532e967314c` |
| `qwen_image_2.1_vae_bf16.safetensors` | 675,509,688 | `bb21f7473051e1ac368515dd3f2e15cd44d7a11748ee8823e1ddca3e4876b7c9` |

The OpenVINO FP32 export records `source_sha256 = 68bdc82…` and `output_bin_sha256 = 5681d9a6…` (25.876 GiB). The BF16 deployment IR records `output_bin_sha256 = 62eee985…` (12.938 GiB). The INT8 IR records `output_bin_sha256 = 330f3a53…` (6.474 GiB).

## 3. Deviations from the reference environment

1. **cgroup, not bare metal:** 16 CPU quota and 55 GiB memory of a 128-thread / 503 GiB host. All CPU benchmarks use 16 threads; they are not host-wide.
2. **GPU product string:** the driver reports `AMD Radeon Graphics` / gfx1100 / 48 GiB. Architecture and VRAM match the W7900-class reference; the literal product name is a deviation.
3. **ComfyUI staged from codeload tarball:** git-over-proxy intermittently returned HTTP 503. The pinned tree was verified by `comfyui_version.py` (0.43.0), `main.py`, and `.lab_pinned_commit`; ComfyUI itself has no `.git` directory.
4. **ComfyUI requirements workaround:** `comfyui-workflow-templates==0.11.100` was installed `--no-deps`; its optional `comfyui-workflow-templates-media-assets-02==0.1.26` is not on PyPI. All runtime imports used by the lab were verified. This does not affect inference.
5. **OpenVINO isolation:** `.venv-ov` owns OpenVINO/NNCF and borrows the preinstalled ROCm PyTorch via a read-only `.pth` bridge. The working ROCm environment was not replaced.
6. **Large IR storage:** FP32 (26 GB), BF16-compressed (13 GB) and INT8 (6.5 GB) IRs live on `/opt/m51-artifacts` (1.6 TB root overlay) and are symlinked into the run because `/workspace` is only 98 GB. They are not committed.
7. **Downloads:** hf-mirror ranged transfers; every file was checked against the in-repo pins. The TE hash was independently recomputed after download.

## 4. Admission rationale

- One real gfx1100 AMD GPU with 48 GiB VRAM: PASS.
- ROCm PyTorch works and was preserved: PASS.
- 16-CPU / 55 GiB budget is sufficient: FP32 export peaked at **40.0 GiB**; FP32 runtime at **50.5 GiB**; BF16 at **30.0 GiB**; INT8 at **16.8 GiB**.
- Disk was sufficient after staging: 47 GB free; all runs completed.
- Verdict: **GO_WITH_DEVIATIONS**, not a perfect bare-metal W7900 reproduction.
