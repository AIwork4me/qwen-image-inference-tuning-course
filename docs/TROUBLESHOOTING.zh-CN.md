# 故障排查与已知局限（Linux W7900 实验）

收集自实验搭建过程中的真实故障（每一条在修复被信任之前都至少复现过一次）。
English version: [TROUBLESHOOTING.md](TROUBLESHOOTING.md)

## 环境

| 症状 | 原因 | 解决 |
|---|---|---|
| 在 venv 里 `python -m venv` 看不到 torch | venv 套 venv 不继承 site-packages | 用基础解释器创建实验 venv，并加 `_rocm_base_env.pth` 桥接到经过验证的 ROCm site-packages |
| pip 试图替换 torch | 依赖解析 | 绝不在可用的 ROCm torch 上安装；改为桥接（P0 §4） |
| `comfyui-workflow-templates` 媒体资产卡住 | 约 100 MB 的 wheel vs 不稳定的代理 | templates+core 不带媒体资产也能装；受保护的 import 使其为可选（服务端打印一行 ERROR，对 API 使用无害） |
| 1 个 CPU 测试失败（`test_stamp_cache_cannot_mask_corruption`） | 该 ext4 对快速连续写返回相同 `st_mtime_ns`（已验证） | 环境特例，记录于 P0 §2；GitHub runner 上通过 |

## 测量

| 症状 | 原因 | 解决 |
|---|---|---|
| 记录里 `vram_peak_mb: 0.0` | 上游采样器在设备名里 grep "7900"；Linux 报告 `AMD Radeon Graphics` | `LinuxRunMonitor`（lab/runner.py）采样任意 AMD 设备 + rocm-smi 设备 VRAM |
| `check_server` 拒绝健康的 W7900 | 同名 grep 问题 | `lab_env.check_server_linux` 以"任意 AMD CUDA 设备"为门 |
| rocm-smi 利用率缺失 | 该构建上 `-g` 显示的是频率 | 用 `rocm-smi -u --csv` 看 GPU 使用率、`--showmeminfo vram --csv` 看显存 |
| `rglob` 对存在的文件返回空 | ext4 loop 文件系统 readdir 不稳定 | 显式使用 `iterdir()`/`glob()` |

## rocprofv3

| 症状 | 原因 | 解决 |
|---|---|---|
| `<cwd>/.rocprofv3` 里只有 `.dat` 缓冲、没有 `*_kernel_trace.csv` | 服务端被 SIGTERM（或对应用发 SIGINT）杀死——没有终结化 | **对 rocprofv3 包装进程发 SIGINT**；它会转发、终结化、把 CSV 写入 `-d` 并删除临时目录 |
| 追踪窗口没捕到 TE 内核 | 90 秒窗口在生成开始约 1 秒后打开 | TE 成本在流水线层测量（新提示词臂）；内核级 TE 用 H1 探测 |
| 12 MB 响应被截断 | 出口代理对流式响应的限制 | 大文件一律用分段并行下载（`lab/download_weights.py`） |

## 结果本身的已知局限

- VRAM 峰值是 0.5–1 秒采样 → 瞬态的**下界**（torch 层与设备层都记录）。
- H2 的流水线臂只用了一个 CPU TE worker；未尝试多 worker 变体（CPU TE 每提示词
  160–221 s——加 worker 救不了）。
- 2048² 分辨率未探索：任务门将其放在"稳定的 1024² + 质量批准"之后，质量评审仍在
  进行——记录为后续工作，不是失败。
- Windows 与 Linux 的数字从不当作等价物比较（不同的 GPU、注意力路径、技术栈）。
- 质量门的人工判定对 BEST 仍为 **pending**；在判定记录之前，本实验中的任何内容
  都不得被引用为质量批准。（M5.1 C1b-vs-G 范围的所有者判定已于 2026-11-05 记录，
  见 `docs/release-closure/PR2_OWNER_QUALITY_VERDICT.md`，且不得外推。）

## Radeon Cloud 实例（来自 2026-11-04 实机审计）

| 症状 | 原因 | 解决 |
|---|---|---|
| ComfyUI 启动即崩：`ImportError: attempted relative import ... lab/profile.py` | 服务端 `sys.path` 上的 `lab/` 遮蔽了标准库 `profile` 模块（cProfile → torch._dynamo → comfy_kitchen） | 已在 `lab/comfyui_launch.py` 修复（从 sys.path 剥离 lab/）；看到就 `git pull` |
| 记录里 `vram_device_peak_mb: 0.0`（ROCm 7.2.1） | 该环境 rocm-smi VRAM CSV 是 3 列（参考主机 4 列）；旧解析器看固定列 | 已修复：runner 按表头定位 used-bytes 列 |
| pip：`No matching distribution found for comfyui-workflow-templates-media-assets-02` | 镜像默认索引（清华源）滞后新包 | 加 `--index-url https://pypi.org/simple/` 重试（start.sh 已自动处理） |
| python `requests` 访问 huggingface.co 超时而 `curl` "正常" | 出口代理封锁 huggingface.co；平台 curl 是个包装，会静默改写到 hf-mirror.com，掩盖封锁 | 显式使用 hf-mirror.com（实验默认；`--hf-base` 可覆盖） |
| 经代理 `git clone`：`CONNECT tunnel failed, response 503` | 代理间歇性拒绝 git 的 HTTP/2 隧道 | `git config --global http.version HTTP/1.1` 后重试；ComfyUI 会回退到 codeload tarball |
| 冷运行明显慢于参考（如 95 s vs 74 s） | 云持久存储（loop 挂载）加载模型比实验室本地盘慢 | 预期的平台差异；热态数字不受影响；每条记录如实记载 |
| ComfyUI 启动/生成崩溃：`LLVM ERROR: IO failure on output stream: No space left on device` 而 `/workspace` 还有空间 | **/tmp（约 4 GiB tmpfs）满了**——ROCm/LLVM 代码生成临时文件在那里，不在工作区卷 | `df -h /tmp` 后清理元凶（pytest 残留、过期下载）；start.sh 的诊断现在打印 /tmp 剩余空间并在 < 2 GiB 时告警 |
