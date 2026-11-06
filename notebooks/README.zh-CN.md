# Jupyter 实验课程：Qwen-Image × ROCm 推理优化

[English](README.md) · [中文教学大纲](../course/SYLLABUS.zh-CN.md) · [仓库首页](../README.zh-CN.md)

**9个实践主题、18本中英文 Notebook、一条原则：没有证据就不能声称加速。** 每个实验都直接分析已提交的 W7900 真实测量，贯穿基准统计、性能剖析、少计算、快计算、异构并行、公平 A/B、画质门禁和开源 PR。

## 三步开始：没有 GPU 也能 Run All

1. 克隆仓库后，从**仓库根目录**打开 Jupyter（Radeon Cloud 可直接使用现有 Notebook 界面）。如缺少 Jupyter，仅在独立 Notebook 环境按 `requirements-notebooks.txt` 安装；**不要改动已有 ROCm PyTorch 环境**。
2. 打开 `notebooks/00_experimental_discipline.zh-CN.ipynb`，选择 Python 3 内核，点击 **Run All**。默认只读取本地固定 JSON，无需网络、下载模型或运行 GPU。
3. 按课程顺序运行，逐步对照 Markdown 理论、源 JSON 和自己的计算结果。

**重要：** Jupyter 应从克隆后的仓库根目录启动；第一个代码单元会寻找 `lab/reference_runs.json`。若启动目录不对，请导航到仓库后再打开 Notebook。

## 9 个实践模块

| 模块 | 中文 Notebook | English | 实践主题 |
|---|---|---|---|
| M0 · Environment / 实验规范 | [打开](00_experimental_discipline.zh-CN.ipynb) | [Open](00_experimental_discipline.ipynb) | 来源可验证的 CPU 数据重算 |
| M1 · Baseline / 基准统计 | [打开](01_baseline_and_uncertainty.zh-CN.ipynb) | [Open](01_baseline_and_uncertainty.ipynb) | 来源可验证的 CPU 数据重算 |
| M2 · Profiling / 性能剖析 | [打开](02_profile_and_bottlenecks.zh-CN.ipynb) | [Open](02_profile_and_bottlenecks.ipynb) | 来源可验证的 CPU 数据重算 |
| M3 · Less compute / 少计算 | [打开](03_less_computation.zh-CN.ipynb) | [Open](03_less_computation.ipynb) | 来源可验证的 CPU 数据重算 |
| M4 · Faster compute / 快计算 | [打开](04_faster_computation.zh-CN.ipynb) | [Open](04_faster_computation.ipynb) | 来源可验证的 CPU 数据重算 |
| M5.0 · Negative offload / CPU 卸载负结果 | [打开](05_cpu_offload_negative.zh-CN.ipynb) | [Open](05_cpu_offload_negative.ipynb) | 来源可验证的 CPU 数据重算 |
| M5.1 · OpenVINO fair A/B / 公平对照 | [打开](06_openvino_and_fair_ab.zh-CN.ipynb) | [Open](06_openvino_and_fair_ab.ipynb) | 来源可验证的 CPU 数据重算 |
| M6 · Combined & quality / 组合与画质 | [打开](07_combined_and_quality.zh-CN.ipynb) | [Open](07_combined_and_quality.ipynb) | 来源可验证的 CPU 数据重算 |
| M7 · Evidence-backed PR / 开源挑战 | [打开](08_capstone_and_pr.zh-CN.ipynb) | [Open](08_capstone_and_pr.ipynb) | 来源可验证的 CPU 数据重算 |

## 三种学习轨道

**A · 无 GPU / 证据分析（默认）：** 只依赖 Python 标准库，读取固定 `evidence/` 与 `runs/` 数据。可选 matplotlib 图表默认关闭。GitHub CPU CI 会自动运行所有 Notebook 的代码单元，验证无 GPU 模式。

**B · GPU 冒烟（自愿）：** 部分 Notebook 提供 `RUN_GPU=False` 开关。学生确认硬件/额度与模型准备后，再显式改成 `True`，并在自己的可信环境设置 `QIR_NOTEBOOK_GPU_ACK=YES`。只运行白名单中的 `--smoke`，带超时，写入 `runs/`，绝不污染 `evidence/`。运行前先读[显卡云入门](../docs/RADEON_CLOUD_QUICKSTART.zh-CN.md)；**GitHub CPU CI 不等于真机验收**。

**C · 正式研究复现：** 遵守[完整复现指南](../docs/REPRODUCTION_GUIDE.zh-CN.md)，匹配实验总体、模型/源码 SHA、CPU/GPU 预算、质量验收并附真实 PR 证据。Notebook 不会自动启动完整长时间评测。

## 双语一致性与安全规则

- 两种语言的**可执行代码逐单元完全相同**，由 CI 强制检查；教学目标、任务、解释分别使用中英文。
- `evidence/` 只读；M5.0 的负结果不改写，M5.1 的人工接受只覆盖 C1b，INT8 数值 FAIL 保留，M6 BEST 人工画质仍然 Pending。
- `python -m pytest tests/test_notebooks.py -q` 可在无 GPU 环境执行全部 Notebook；`python scripts/course-quality/check_notebooks.py` 检查结构与双语代码一致性。
- 学生分析记录必须显式授权，写入 `runs/notebook-student/`；引用数据重算不能冒充新的 GPU 实测。

**环境：** Python 3.10+，推荐 3.12；数据分析只依赖标准库。可在单独的 Notebook 环境选择性安装 `requirements-notebooks.txt`，不要重装 ROCm torch。

教师说明：[中文](INSTRUCTOR_GUIDE.zh-CN.md) · [English](INSTRUCTOR_GUIDE.md)。
