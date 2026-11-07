# Jupyter 实验课程：GPU 优先的 Qwen-Image × ROCm 推理优化

[English](README.md) · [中文教学大纲](../course/SYLLABUS.zh-CN.md) · [仓库首页](../README.zh-CN.md)

**9 个模块、18 本中英文 Notebook、一条科学原则：没有证据就不能声称加速——而在 Radeon Cloud 上，首要证据来自你自己的 W7900 实测。**

实验课在物理 AMD Radeon PRO W7900 上完成完整实验闭环：
**理解 → 预测 → 跑 GPU 基线 → 实施或选择优化 → 跑 GPU 候选 → 测量 → 对比 → 质量验证 → 解释 → 贡献。**

## 三步开始（GPU_CLASSROOM，Radeon Cloud 课堂默认模式）

1. 先完成一次 [Radeon Cloud 快速入门](../docs/RADEON_CLOUD_QUICKSTART.zh-CN.md)：它会布置锁定的 ROCm 环境、ComfyUI 与模型权重，并运行 `python -m lab doctor`（不得 FAIL）。
2. 从**仓库根目录**启动 Jupyter，用课程环境的 Python 内核打开 `notebooks/00_experimental_discipline.zh-CN.ipynb`（或英文版）。
3. 在**第一个代码单元**设置模式，然后 *Kernel → Restart & Run All*：

```python
COURSE_MODE = "GPU_CLASSROOM"
```

横幅会醒目显示已准入的 GPU（`AMD ... gfx1100 ... 48GB`），或给出学生可读的阻断原因与确切修复方法。每次物理运行都会打印 `run_id`，记录只追加地写入 `runs/<run-id>/`——不覆盖、绝不触碰冻结的 `evidence/`。

还没有 GPU？保持 `COURSE_MODE = "REFERENCE"`：笔记本改为分析已提交的冻结实验数据（纯 CPU、离线、无网络）——相同的单元，全部标注 `[HISTORICAL]`。

## 三种模式

| 模式 | 运行内容 | 输出位置 | 适用场景 |
|---|---|---|---|
| `REFERENCE` | 仅分析冻结的已提交证据 | 无（仅分析） | CPU CI、离线学习、调试 |
| `GPU_CLASSROOM` | **你自己在这台 W7900 上的有界实测**（每臂 1 提示词 × warmup+2 测量，约 5 分钟/臂） | `runs/` | 默认学生实验模式 |
| `GPU_RESEARCH` | 完整冻结协议（3 提示词 × 3 种子 × warmup+3 测量 + fresh-prompt 臂） | `runs/` | 毕业课题、研究级复现 |

运行时与测试双重保障：同一时刻仅一个 GPU 负载（文件锁）；执行前必须硬件准入；课堂模式有按实验定制的超时预算；research 档位仅限 `GPU_RESEARCH` 模式；记录只追加；旧版 smoke 助手仍保留 `QIR_NOTEBOOK_GPU_ACK` 显式确认。

**课堂练习是有界物理实验，不是正式性能结论。** 正式结论需要研究协议（多提示词/种子重复、匹配总体、声明顺序）——笔记本会自动为每次对比标注 `BOUNDED_PHYSICAL_COMPARISON` 或 `FORMAL_PROTOCOL_MEASURED`。

## 每个数字从哪里来？（学生实验溯源）

仓库自带真实的 W7900 物理记录（`runs/` 下的教师实验），因此全新克隆里已经存在**
不属于你**的物理证据。笔记本展示的每个数字都绑定八种机器校验来源之一
（[溯源契约](../docs/notebook-gpu-qualification/STUDENT_PROVENANCE_CONTRACT.md)）：

`CURRENT_SESSION_PHYSICAL`（你的 GPU、本次会话）· `SAME_STUDENT_PRIOR_PHYSICAL`
（你更早的会话，按精确 run id 选择）· `SHARED_CAMPAIGN_PHYSICAL` ·
`COMMITTED_PHYSICAL_REFERENCE`（已提交进 Git——教师证据，有价值但**绝不是
“你今天实测的”**）· `FROZEN_REFERENCE`（`evidence/`）· `SIMULATED` ·
`NOT_RUN` · `BLOCKED`。

在笔记本第一个代码单元中选择证据来源：

```python
RUN_OWN_GPU_ARMS = True    # 在 THIS GPU 上运行我自己的有界实验臂（默认）
# 或者在计量实例上复用早前的运行以节省 GPU 时间：
# RUN_OWN_GPU_ARMS = False
# REUSE_TE_RUNS = {"cuda": "<精确 run id>", "cpu": "<run id>", "h3": "<run id>"}
# REUSE_PIPELINE_RUN = "<精确 run id>"
```

选择复用必须给出**精确 run id**（它显示在每个测量旁边，并保存在
`runs/` 下的 `notebook-sessions/` 会话记录里）。绝不存在“取最新文件”式选择；复用的
运行绝不会被计为第二次独立测量——例如中文与英文 M5.1 笔记本可以共享同一次物理
A/C 运行，但会被标注为复用。

**出问题时（全部可恢复，且不会覆盖任何数据）：**

- *运行被中断 / 超时*——重新运行该单元：每次尝试都会获得新的 `run_id`；已完成
  的记录原样保留。
- *OpenVINO 工件缺失（M5.1 准入 `BLOCKED`）——按打印的补救指令操作：
  `bash scripts/m51/provision_ov.sh`，或把 `$M51_OV_ARTIFACTS` 指向共享的已布置
  工件根目录（如 `/opt/m51-artifacts`）。
- *工件 `FAIL`（摘要/身份不符）——工件无效；依赖阶段立即停止。重新布置；绝不要
  带着未验证的工件继续。
- *GPU 忙——另有负载占用设备；先完成它（同一时刻仅一个 GPU 负载，由锁强制）。
- *RAM 准入 `BLOCKED`——释放内存或换更大实例；安全余量绝不会被静默调低。

## 9 个实践模块

| 模块 | 中文 Notebook | English | 物理练习（GPU 模式） |
|---|---|---|---|
| M0 · Environment / 实验规范 | [打开](00_experimental_discipline.zh-CN.ipynb) | [Open](00_experimental_discipline.ipynb) | 设备准入、同步 vs 异步计时、模型指纹校验 |
| M1 · Baseline / 基准统计 | [打开](01_baseline_and_uncertainty.zh-CN.ipynb) | [Open](01_baseline_and_uncertainty.ipynb) | 你自己的 B0（BF16、40 步）基线 + 与 55.206 秒参考值对比 |
| M2 · Profiling / 性能剖析 | [打开](02_profile_and_bottlenecks.zh-CN.ipynb) | [Open](02_profile_and_bottlenecks.ipynb) | 流水线分解 + 真实 rocprofv3 内核轨迹、Amdahl 上限 |
| M3 · Less compute / 少计算 | [打开](03_less_computation.zh-CN.ipynb) | [Open](03_less_computation.ipynb) | B0 vs L1（Turbo 8 步）物理 A/B、混淆变量分析 |
| M4 · Faster compute / 快计算 | [打开](04_faster_computation.zh-CN.ipynb) | [Open](04_faster_computation.ipynb) | F1 ck 注意力与 F2 INT8 双臂、后端生效验证 |
| M5.0 · Negative offload / CPU 卸载负结果 | [打开](05_cpu_offload_negative.zh-CN.ipynb) | [Open](05_cpu_offload_negative.ipynb) | GPU vs CPU 文本编码器试点（实测负结果） |
| M5.1 · OpenVINO fair A/B / 公平对照 | [打开](06_openvino_and_fair_ab.zh-CN.ipynb) | [Open](06_openvino_and_fair_ab.ipynb) | OV 工件准入、编码器 A/B、流水线 A/C、公平因子实验 |
| M6 · Combined & quality / 组合与画质 | [打开](07_combined_and_quality.zh-CN.ipynb) | [Open](07_combined_and_quality.ipynb) | B0 vs BEST 物理对比 + 你自己的成对质量网格 |
| M7 · Evidence PR / 开源挑战 | [打开](08_capstone_and_pr.zh-CN.ipynb) | [Open](08_capstone_and_pr.ipynb) | 用你的记录组装毕业报告 + PR 草稿 |

## 质量与完整性

- 中英文笔记本**可执行单元完全一致**（CI 强制检查），学习文字各自本地化。
- 源输入只读。任何笔记本都不写 `evidence/`。
- 每行摘要都用标签区分 实测 / 历史 / 估计 / 模拟 值。
- M5.0 负结果保持可见。M5.1 INT8 数值 **FAIL 永远是 FAIL**；C1b 所有者验收只限定于其历史评估的部署范围。M6 BEST 人工质量判定保持 **PENDING_HUMAN_REVIEW**——没有任何单元会自动批准质量。
- 讲师可无头复跑全部 18 本笔记本：
  `python scripts/notebook-qualification/run_all.py --mode REFERENCE`（CPU CI）或
  在 W7900 主机上 `--mode GPU_CLASSROOM`。

**环境要求：** Python 3.10+（推荐 3.12）。REFERENCE 模式仅需标准库。GPU 模式需要快速入门布置的锁定 `.venv-lab` ROCm 环境（绝不重装云镜像的 torch）。可选 Notebook 依赖：`requirements-notebooks.txt`。

讲师指南：[English](INSTRUCTOR_GUIDE.md) · [中文](INSTRUCTOR_GUIDE.zh-CN.md)。
