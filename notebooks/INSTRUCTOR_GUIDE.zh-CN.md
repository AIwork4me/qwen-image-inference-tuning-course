# 讲师指南 —— GPU 优先 Jupyter 实验课授课

[English](INSTRUCTOR_GUIDE.md)

**授课对象：** 已完成 Radeon Cloud 入门、持有真实 Radeon PRO W7900 实例、具备基础 Python 能力的工科研究生。**模式：** 9 个顺序实验。`GPU_CLASSROOM` 是默认学生体验；`REFERENCE` 仍是离线/CI 备选；`GPU_RESEARCH` 面向毕业课题。

### 每个学生实例的前置条件

1. 已完成 `bash scripts/cloud/start.sh`（布置权重、锁定 ComfyUI、`.venv-lab`，cloud doctor 不 FAIL）。M4/M6 各臂与完整矩阵需要 `--all-models`（约 61 GB）。
2. Jupyter 从仓库根目录启动，且 `.venv-lab` 内核可用。
3. M5.1：已布置 OpenVINO 工件（`bash scripts/m51/provision_ov.sh`），或 `M51_OV_ARTIFACTS` 指向已校验的 IR。`PASS_WITH_WARNINGS`（XML 摘要未锁定）是诚实状态——要讲透，不要四舍五入。

### 建议授课节奏

M0 + M1 热身（准入横幅、同步纪律、自己的基线 vs 55.206 秒参考值）；M2 + M3 剖析/Amdahl 与少计算 A/B（**先声明混淆变量**）；M4 注意力/量化并验证后端真实生效；M5.0 + M5.1 实测负结果与 OpenVINO 恢复（编码器加速比 ≠ 流水线收益；公平 2×2 因子）；M6 组合优化 + 人工质量评审；M7 毕业课题 PR。要求学生**先预测再运行**，然后为自己的分母、来源、样本总体和局限辩护。

### 评分标准

- 20% 准入与来源纪律（模式诚实、实测/历史/估计标签）
- 25% 自有 GPU 测量质量（warm 总体、离散度、VRAM 记账）
- 25% 匹配的 A/B 设计与不混淆的加速比（禁止 F1×F2 相乘）
- 15% 保真/距离语义 + 人工判定边界（PENDING_HUMAN_REVIEW）
- 15% 毕业课题：可复现报告、诚实的证据等级（PILOT vs FORMAL）

### 课堂运维

- 同一时刻仅一个 GPU 负载：运行时用文件锁跨内核强制串行。
- 一个课堂臂约 5 分钟（启动 + cold + warmup + 2 次测量）。每个 GPU 模块预留约 20 分钟；9 个模块的课堂通过可在 2–3 小时实验课内完成。
- `GPU_CLASSROOM` 下 Run All 天然有界（按实验的超时、单提示词）。学生必须显式修改 `COURSE_MODE`；没有任何路径会静默升级到研究协议。
- 真实内核执行已纳入 CI：`tests/test_notebooks.py` 在可用时通过真实 Jupyter 内核（nbclient）以 REFERENCE 模式执行全部 18 本笔记本。

**教师检查（无 GPU）：** `python -m pytest tests/test_notebooks.py tests/test_notebook_gpu_protocol.py -q`、`python scripts/course-quality/check_notebooks.py`。
**教师检查（W7900）：** `python scripts/notebook-qualification/run_all.py --mode GPU_CLASSROOM --campaign runs/notebook-gpu-qualification/<id>`，随后 `validate_evidence.py --campaign ...`。

**故障政策：** 准入被阻断本身就是课程素材——笔记本会打印确切原因，学生切换到 `REFERENCE` 学习同一套方法论。有据可查的实测退化得满分；照抄历史数字不得分。
