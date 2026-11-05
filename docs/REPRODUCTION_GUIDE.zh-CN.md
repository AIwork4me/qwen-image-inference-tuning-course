# 复现指南 —— Linux W7900 实验

以下一切都在物理 W7900 实验室执行（ROCm 7.2.1、torch 2.9.1+rocm7.2.1、ComfyUI
v0.43.0 @ b5a4321、权重 @ aacf31a6、SHA256 固定），并于 2026-11-04 在真实 Radeon
Cloud 实例上端到端复验（同栈）。在其他硬件或技术栈上，你的绝对数字会不同——
*方法论*才是可复现的东西。
English version: [REPRODUCTION_GUIDE.md](REPRODUCTION_GUIDE.md)

## 0. 预检（失败即拒绝）

```bash
bash scripts/cloud/start.sh        # 一条命令安装（见 docs/RADEON_CLOUD_QUICKSTART.zh-CN.md）
.venv-lab/bin/python -m lab doctor --cloud   # 分级模式（首图模型集）
.venv-lab/bin/python -m lab doctor           # 严格模式（正式运行）——必须打印 verdict: PASS
```

严格 doctor 检查：恰好一块约 48 GB 的 AMD gfx1100 级 GPU、torch/ROCm 版本、
ComfyUI 固定提交标记、每个权重文件的 SHA256（首次全量散列；此后按大小+mtime 键的
戳记缓存——文件任何变化强制重新全量散列）、冻结提示词、实时 flash-SDPA 探测、
rocprofv3 可用性。输出写入 `runs/doctor.json`；随附的 `evidence/results/doctor.json`
属于冻结战役，永不被覆盖。

## 1. 环境

`start.sh` 构建一切。需要时的手动等价：

- 从当前 python（Radeon Cloud 上是已带 ROCm torch 的 `/opt/venv`）创建
  `.venv-lab`，加只读 `.pth` 桥接到系统 site-packages——**绝不在可用的 ROCm
  构建上重装 torch**；
- 把 ComfyUI 的 `requirements.txt` + `requirements-dev.txt` 装进 `.venv-lab`
  （配置的镜像缺新包时回退 `--index-url https://pypi.org/simple/`）；
- ComfyUI 位于 `ComfyUI`（固定提交 `b5a4321`；出口代理拦 git 时回退 tarball），
  权重位于 `weights/comfyui-split/`（固定摘要；恶劣网络用
  `lab/download_weights.py`——16 路分段续传；在被审计的云网络上 huggingface.co
  被封锁，允许的源是 hf-mirror.com）。

## 2. 冻结基线（阶段 1）

```bash
.venv-lab/bin/python -m lab run --experiment B0 --date <tag>
```

协议：3 提示词 × 种子 [42, 1337, 20261030]；冷（seed+1M）、预热（seed+2M，击穿
缓存）、每提示词 3 次测量、新提示词臂（加" ."后缀）——共 16 次运行、全新服务端。
**记录落在 `runs/<tag>-B0/`（schema 合法 JSON），原始 PNG 在
`runs/raw/<tag>-B0/`。** 冻结战役自己的两个可复现性会话在
`evidence/results/p1-baseline-B0/` 与 `p1-baseline-r2-B0/`：热态中位 55.24 vs
55.21 s（合并 18 次中位 55.21 s；P1 日志原文里的逐会话数字反映父项目最初的
报告口径）。

## 3. 剖析（阶段 2）

```bash
.venv-lab/bin/python -m lab profile --date <tag>
```

A 层聚合*你的* B0 记录（仅当你没有记录时回退到冻结参考记录）成流水线分解
（`pipeline_breakdown.json`，带显式实测-vs-推导的 `method` 映射）。B 层把真实
ComfyUI 服务端置于 `rocprofv3 --runtime-trace --output-format csv -P 230:90:1`
之下，由驱动客户端把一次热态生成对齐进有界窗口；关闭方式是对 **rocprofv3 包装
进程**发 SIGINT（任何地方发 SIGTERM 都会留下未转换的 `.dat` 缓冲——修复前复现过
两次）。输出：`runs/<tag>-profile/rocprof-{dit,te}/analysis.json` + 原始 CSV
（每个约 85 MB，不提交）。

## 4. 实验（阶段 3–5）

```bash
for EXP in L1 L1O F1 F2 LF1 LF2 LF3 BEST; do
  .venv-lab/bin/python -m lab run --experiment $EXP --date <tag>-$EXP
done
# 自选研究重跑（缓慢的 CPU 受限会话；默认从已提交证据讲授——见 M5 的预算说明）：
.venv-lab/bin/python -m lab run --experiment H1   # CPU TE 探测（约 15 分钟）
.venv-lab/bin/python -m lab run --experiment H2   # 流水线 A/B（约 15 分钟）
```

每次运行：全新服务端、健康门、schema 合法记录（在 `runs/` 下）、从服务端日志
抓取的注意力路径证据、图像 SHA256、torch + rocm-smi 双视角 VRAM 峰值、GPU 利用率
采样。runner 预检 manifest 所需的权重文件，缺失时打印精确的下载命令（分阶段模型
策略）。

## 5. 质量门（阶段 6）

```bash
.venv-lab/bin/python -m lab quality --experiment F2     # 保真度语义
.venv-lab/bin/python -m lab quality --experiment BEST   # 距离语义
```

该命令解析最新的原始图像目录（你在 `runs/raw/` 下的，或用 `--date <run-id>`
显式指定），从 manifest 的 `quality_semantics` 字段推导语义，把网格 + LPIPS 写到
`runs/quality/<exp>/`——`evidence/quality/` 下冻结战役的网格永不被覆盖。底层工具
仍可直接使用：

```bash
.venv-lab/bin/python -I lab/quality_gate.py \
  --baseline runs/raw/<tag>-B0 \
  --experiment runs/raw/<tag>-LF2 \
  --experiment-name LF2 --semantics distance \
  --baseline-desc "<描述基线>" \
  --noise-a <同配置图 A> --noise-b <同配置图 B> \
  --out runs/quality-LF2
```

语义：同模型数值变体（INT8）用 `fidelity`；不同调度（Turbo）用 `distance`。
噪声底来自两次独立的同配置运行（9/9 对实测 0.0000）。对 `quality_grid.png` +
`quality_crops_poster.png` 的人工三级判定是具有约束力的门——准备网格，不要
伪造判断。

## 6. 报告

```bash
.venv-lab/bin/python -m lab report                    # 冻结参考范围（默认，只读）
.venv-lab/bin/python -m lab report --scope student --write    # 只有你的 runs/（runs/SUMMARY.*）
.venv-lab/bin/python -m lab report --scope student --kind smoke --write   # 只有有界冒烟
.venv-lab/bin/python -m lab report --scope all --write  # 参考 + 学生并排展示，绝不合并
.venv-lab/bin/python -m lab report --write              # 重新生成冻结证据 SUMMARY（维护动作）
```

参考范围只聚合 `lab/reference_runs.json` 列出的运行目录——冒烟、重复与学生数据
永远无法污染已发表的中位数（B0 汇集两个可复现性会话，n=18；其余实验各 n=9）。

## 已知复现障碍（均已实测，见 P0/P2 与云端审计）

| 障碍 | 所用对策 |
|---|---|
| 出口代理杀 git-clone / 大的单响应 | 仓库用重试 + `http.version=HTTP/1.1`；ComfyUI 用 codeload tarball；权重从允许的镜像 16 路分段下载 |
| 平台 `curl` 静默把 huggingface.co 改写为 hf-mirror.com（掩盖真实出口策略） | 实验永远显式使用 `hf-mirror.com`（`--hf-base` 可覆盖） |
| ext4 loop 文件系统上 `rglob` 不可靠 | 分析代码显式 `iterdir`/`glob` |
| 文件系统 mtime_ns 在快速连续写时冻结 | 戳记缓存对任何变化重新散列；缓存命中可疑时删除 `weights/comfyui-split/.digest_cache.json` |
| rocprofv3 + SIGTERM → 没有 CSV | 永远对 rocprofv3 包装进程发 SIGINT |
| `rocm-smi -g` 显示频率而非利用率 | GPU 使用率用 `-u --csv`，设备显存用 `--showmeminfo vram --csv`（各 ROCm 版本列布局不同——按表头解析） |
| pip 镜像（清华）滞后新包 | 用 `--index-url https://pypi.org/simple/` 重试 |
| 服务端 sys.path 上的 `lab/` 遮蔽标准库 `profile` | `comfyui_launch.py` 剥离之（本仓库已修复） |
