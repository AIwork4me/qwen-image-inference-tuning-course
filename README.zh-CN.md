# 让生成式 AI 更快 —— 一门以实测为本的优化课程

**以 Qwen-Image 2.1 在 AMD Radeon PRO W7900 上的推理加速调优为例**
（[English README](README.md)）

> **一块 GPU。一个模型。一条基线。三种优化策略。可测量的结果 ——
> 每一条结论都能追溯到一份已提交的运行记录。**

本仓库讲授**如何像专业工程师一样加速图像生成负载**：

**基线 → 剖析 → 更少计算 → 更快计算 → 并行与异构 → 组合优化 → 质量门禁 → 公开复现。**

全部内容锚定在一个真实案例上：2026-11-03 在单张 Radeon PRO W7900
（gfx1100，48 GB，Linux ROCm）上对 Qwen-Image-2.1 的实测 ——
146 条通过 schema 校验的运行记录，全部提交在 `evidence/` 下。

## 核心实测结果（暖态 = 9 次中位数，1024²）

| 阶段 | 配置 | 暖态 | 相对基线 | 峰值显存 |
|---|---|---|---|---|
| **B0** | bf16、40 步（冻结基线） | 55.21 s | 1.00× | 37.7 GB |
| L1 | Turbo 8 步（更少计算） | 12.13 s | 4.55× | 37.7 GB |
| F1 | ck 注意力（更快计算） | 45.09 s | 1.22× | 37.7 GB |
| F2 | INT8 权重（首先是省显存：−6.7 GB） | 51.09 s | 1.08× | **31.0 GB** |
| **BEST** | Turbo INT8 + 默认注意力 | **9.02 s** | **6.12×** | **31.0 GB** |

另有两个**带证据的负结果**（课程中最有教学价值的部分）：CPU 文本编码
在数值等价（余弦 ≥ 0.999999）的前提下仍比 GPU 慢 152–391×；CPU-GPU
请求流水线即使有 50.4 秒**经时间戳证实的真重叠**，吞吐仍差 14×。

质量：INT8 近无损（LPIPS 保真 0.005–0.031）；Turbo 改变了采样调度
（距离 0.19–0.30），其**三档人工终审（无感/略降/不可接受）尚待完成** ——
速度永远不能凌驾于质量门禁之上。

## 课程结构（8 学时 + 实验 + 90 分钟讲义）

七个模块 + 学生挑战，见 [course/SYLLABUS.md](course/SYLLABUS.md)；
90 分钟课堂演示包（8 页幻灯 + 4 个互动问答，数字全部真实）：
[course/LECTURE_DEMO.md](course/LECTURE_DEMO.md)。

## 自己动手跑实验

```bash
python -m lab doctor                # 故障关闭式预检 —— 必须 PASS
python -m lab baseline              # 冻结 B0 基线（W7900 约 16 分钟）
python -m lab profile               # 流水线 + rocprofv3 内核剖析
python -m lab run --experiment L1   # 或 F1 F2 LF1 LF2 LF3 L1O BEST H1 H2
python -m lab quality --experiment F2
python -m lab report                # SUMMARY + 图表
```

环境要求与逐步指引见 [docs/REPRODUCTION_GUIDE.md](docs/REPRODUCTION_GUIDE.md)。
无 GPU 的学生可基于已附证据完成 M0–M2 练习。

## 五条铁律（本课程真正教的东西）

1. 先测量，再优化。
2. 一次只改一个变量。
3. 永远不要把加速比相乘 —— 组合必须端到端实测。
4. 速度 ≠ 质量；人工终审具有约束力。
5. 结论不得超过证据。

## 来源与许可

本课程衍生自社区基准项目
[AIwork4me/qwen-image-rocm](https://github.com/AIwork4me/qwen-image-rocm)
（MIT）—— 其 Linux W7900 工作流提供了实验代码与全部测量数据。
代码：**MIT**；课程文本、证据数据与报告：**CC-BY-4.0**。
模型权重不入库，遵循上游许可。本项目为独立教学材料，非 AMD 或
阿里/Qwen 官方项目。
