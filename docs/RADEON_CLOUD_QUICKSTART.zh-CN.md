# Radeon Cloud 快速开始 —— 从零到你的第一张图

**目标读者：** 刚拿到 Radeon Cloud 实例批准的学生。在被审计实例上的总耗时：
**约 10–20 分钟**——主要由 25.5 GB 的分阶段模型下载决定，实测云网络（hf-mirror、
16 路分段）聚合约 0.4 GB/s。**你的网络会不同**：典型 20–40 MB/s 时同一下载需
11–22 分钟；2 MB/s 时约 3.5 小时——尽早启动，让可断点续传的下载器工作。重跑只需
几分钟（一切校验后跳过）。
English version: [RADEON_CLOUD_QUICKSTART.md](RADEON_CLOUD_QUICKSTART.md)

> 开始之前：申请免费 Radeon Cloud GPU 算时（平台：
> <https://developer.amd.com.cn/login?source=WmhWQBvrH>，中文图文教程：
> <https://mp.weixin.qq.com/s/dOAIzJ2qsWPmBSH67q41aA>，文档：
> <https://amd-aim.github.io/radeon-cloud-docs/zh-cn/>）。选择**预装
> PyTorch-ROCm 的 Linux Notebook 镜像**和**单 GPU**。配额（免费时长、GPU 型号、
> 存储）以平台控制台显示为准——它们会变化，本指南从不假设具体数字。

## 三条命令

在你的 Notebook 里打开终端，运行：

```bash
cd /workspace
git clone https://github.com/AIwork4me/qwen-image-inference-tuning-course.git
cd qwen-image-inference-tuning-course
bash scripts/cloud/start.sh
```

这一条脚本（所有步骤幂等——随时可重跑）会：

1. **诊断你的实例** —— GPU、ROCm、PyTorch、磁盘、代理。无密钥、无遥测；一切只
   打印给你本人。
2. **按固定提交 `b5a4321`（v0.43.0）获取 ComfyUI** —— 经 git；若云出口代理拦截
   git（偶尔会发生）则改用 tarball。
3. **构建 `.venv-lab`**，通过只读桥*借用*预装的 ROCm PyTorch——你工作中的 torch
   绝不被重装或升级。
4. **暂存出第一张图所需的最小模型集**（约 25.5 GB：turbo-int8 DiT + 8B 文本编码器
   + VAE），从你的网络允许的镜像 16 路可续传下载，然后校验每个文件的**固定
   SHA256**。
5. **运行云端 doctor** —— 失败即拒绝的检查（GPU、技术栈、摘要、提示词、SDPA
   探测、rocprofv3）。
6. **生成一张有界冒烟图**，打印其路径与你的下一课。

健康首次运行的结尾（在被审计的 W7900 级实例上实测）：

```
== 6/6 First image (bounded smoke, BEST config = turbo-int8 8-step) ==
[BEST] starting fresh server (flags=[], sdp=none) ...
  cold          p1-zh-scene seed=1000042 total=33.067s ...
  measure       p1-zh-scene seed=42      total=9.023s  ...
== YOUR FIRST IMAGE ==
  image  : /workspace/qwen-image-inference-tuning-course/runs/raw/<tag>-BEST-smoke/p1-zh-scene-42-r0.png
  record : /workspace/qwen-image-inference-tuning-course/runs/<tag>-BEST-smoke/p1-zh-scene-42-r0.json
```

你的测量存放在 `runs/` 下，**永远**无法修改 `evidence/` 下的冻结参考证据——这层
隔离由代码本身强制，不靠约定。

## 选项

| 标志 | 作用 |
|---|---|
| `--all-models` | 获取全部六个固定变体（约 61 GB；先检查磁盘预算）——B0/F1/F2/L1/L1O/LF1/H1/H2 需要 |
| `--skip-smoke` | doctor 之后停止（不出图） |
| `--smoke-only` | 只重跑冒烟（安装步骤自校验） |
| `--hf-base URL` | 模型源覆盖（默认 `https://hf-mirror.com`；网络可达时可用 `https://huggingface.co`） |

## 第一张图之后：贯穿课程的分阶段路径

```bash
bash scripts/cloud/start.sh --all-models     # 解锁 bf16 基线（+14.2 GB）
.venv-lab/bin/python -m lab baseline         # M01：冻结 B0 参考（约 16 分钟）
```

每个完整协议会话的 GPU 时间预算（W7900 级；冒烟变体每个 1–3 分钟，是课堂所用）：

| 实验 | 完整会话 | 冒烟 |
|---|---|---|
| B0 基线（40 步） | 约 16 分钟 | 约 3 分钟 |
| L1 / L1O（turbo 8 步） | 约 5 分钟 | 约 2 分钟 |
| F1（ck 注意力） | 约 13 分钟 | 约 3 分钟 |
| F2（INT8 40 步） | 约 15 分钟 | 约 3 分钟 |
| LF1–LF3、BEST | 约 4 分钟 | 约 2 分钟 |
| H1（CPU TE 探测） | **约 15 分钟，缓慢 CPU 工作——自主研究，默认从已提交证据讲授** |
| H2（流水线 A/B） | **约 15 分钟——自主研究，同上** |

**省 GPU 额度：** 不 actively 测量时就停止（或销毁）实例。`/workspace`——连同
仓库、权重、`.venv-lab` 与你的 `runs/`——在停止后保留，你什么都不会丢。

## 经实测的恢复指南（下列每个故障都在真实实例上复现过）

### `git clone` 失败：`CONNECT tunnel failed, response 503`
云出口代理间歇性拒绝 git 的 HTTP/2 隧道。`start.sh` 已自动用 HTTP/1.1 重试。若
克隆*课程*仓库仍失败：

```bash
git config --global http.version HTTP/1.1
git clone https://github.com/AIwork4me/qwen-image-inference-tuning-course.git
```

### 模型下载被中断（实例停止、网络掉线）
直接重跑 `bash scripts/cloud/start.sh`。下载器保留逐块 part 文件并按字节精确续传；
已完成文件按大小 + SHA256 戳记检测并跳过。

### 第 4 步 `digest verification FAILED`
某个文件损坏（残缺下载）。删除后重跑：

```bash
rm weights/comfyui-split/<所列文件>.safetensors
bash scripts/cloud/start.sh
```

### `BLOCKED: <N> GiB free ... needs ~M GiB of new downloads`
预算是**增量**的——已存在的固定文件绝不重复计数，`M` 只是仍需获取的部分（例如在
已完成的暂存集上 `--all-models` 只需要剩余的约 33 GiB，不是 61）。要么清空间
（`du -sh /workspace/*`——常见元凶是你已检查过的 `ComfyUI.unknown.*` 隔离目录），
挂载/扩容平台的持久存储，或换工作区更大的实例规格。这是供给问题——代码改动救
不了满盘。注意 `--smoke-only` 不下载任何东西、永不被磁盘阻塞。

### pip 步骤失败（`Could not find a version ...`）
镜像默认 pip 源（清华）有时滞后新包。`start.sh` 已自动对官方 PyPI 重试。手动
等价：

```bash
.venv-lab/bin/pip install --index-url https://pypi.org/simple/ \
  -r ComfyUI/requirements.txt -r requirements-dev.txt
```

### 严格 doctor 在 `gpu_gfx1100_single` 上失败而云端 doctor 通过
你在别的 Radeon 上（架构/显存不同）。学习完全没问题——云端 doctor（`--cloud`）
警告而不失败——但你的数字是独立测量，不是对冻结参考的复现，且 bf16 实验可能装不
进显存（看 WARN 文本）。

### 实例重启 / 重连
`/workspace` 下的一切都还在。重跑 `bash scripts/cloud/start.sh`——它逐阶段校验
（标记文件、摘要、venv 可导入性）并只修复缺失部分。若 `/tmp` 被清空，课程相关的
东西没有存在那里。

### 服务端启动中死亡（`ComfyUI died during boot`）
读该运行的 `server.log` 尾部：

```bash
tail -40 runs/<tag>-BEST-smoke/server.log
```

实战见过的两类启动杀手：损坏的权重文件（用上面的修复）与历史遗留的标准库
`profile` 遮蔽 bug（本仓库已修复——若看到提到 `lab/profile.py` 的
`ImportError: attempted relative import with no known parent package`，`git pull`）。

## 下一步去哪

1. [course/M01_baseline/LESSON.zh-CN.md](../course/M01_baseline/LESSON.zh-CN.md) ——
   测量冻结基线并读懂你的第一份运行记录。
2. [docs/RADEON_CLOUD_READINESS.md](RADEON_CLOUD_READINESS.md) —— 被审计实例的
   长相，以及 `start.sh` 编码进去的每个平台怪癖。
3. [course/SYLLABUS.zh-CN.md](../course/SYLLABUS.zh-CN.md) —— 完整 M00–M07 路径。
