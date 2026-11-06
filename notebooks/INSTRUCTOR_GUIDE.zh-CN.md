# 教师指南：Jupyter 实验课

[English](INSTRUCTOR_GUIDE.md)

**对象：** 具备 Python 基础的高年级本科生/研究生。**形式：** 九个连续实践，每个约20–40分钟；也可将 M0/M1、M2/M3、M5.0/M5.1、M6 组合为100分钟精讲。不要把拥有 GPU 当成学习科研方法的前提。

### 教学节奏

M0/M1 讲基准与证据；M2/M3 讲 GPU 性能剖析与 Amdahl；M4 讲注意力、量化、显存；M5.0/M5.1 讲负结果恢复与公平 2×2；M6 讲画质门禁；M7 按开放源码 PR 验收。

每个实验都先让学生**预测结果，再运行代码**，最后解释分母、样本范围、设备/模型身份和不确定性。

### 建议评分

- 25% 证据来源、基准抽样与冷/热/冒烟区分
- 25% 从原始 JSON 独立计算
- 25% 公平 A/B 设计、辨别混杂变量
- 15% 理解 LPIPS 保真/距离与人工质量裁定边界
- 10% 可证伪假设、复现实验和 PR 说明

### 课堂注意

Jupyter 从仓库根目录启动，推荐 Python 3.12。默认 Run All 无 GPU；matplotlib 图表可自愿开启。只有准备好的 ROCm 教学环境才可以另行授权 GPU smoke，且冒烟数据不能作为发布统计。M6 BEST 仍等待独立人工画质审核，不能擅自标为 PASS。

CPU CI 通过 pytest 执行 Notebook 代码单元以证明默认安全性，**不意味着 GPU 真机验证或 Jupyter UI 可用性已验收**。学生结果不得覆盖 evidence/。

检查命令：`python -m pytest tests/test_notebooks.py -q`、`python scripts/course-quality/check_notebooks.py`。

**失败教学：** 设备不足时，照样从仓库真实数据完成统计、质量和归因练习。诚实的负结果与正向性能提升一样值得满分。
