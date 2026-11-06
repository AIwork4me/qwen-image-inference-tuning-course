# 课程大纲 — 让生成式 AI 更快

**以 Qwen-Image-2.1 在 AMD Radeon PRO W7900 上的推理加速调优为例**
*（一门基于真实、证据齐全的测量战役构建的动手推理加速工程课程）*

- **受众：** 会写 Python、用过 PyTorch 的研究生 / 工程师；不要求 AMD/ROCm 经验。
- **形式：** 7 个模块 ≈ 8 学时 + 一次 100 分钟讲义（2026-11-19 16:30–18:10）+ 实验。
- **实验硬件：** 免费 Radeon Cloud GPU 实例（单 GPU、PyTorch-ROCm 镜像），经
  `bash scripts/cloud/start.sh` 配置——见[快速开始](../docs/RADEON_CLOUD_QUICKSTART.zh-CN.md)。
- **语言：** 英文为规范版，提供完整中文对照（本文件即中文完整版；
  英文版见 [SYLLABUS.md](SYLLABUS.md)）。
- **方法论主线：** [METHODOLOGY.zh-CN.md](METHODOLOGY.zh-CN.md)——测量 → 建模 →
  假设 → 优化 → 验证 → 解释 → 复现 → 贡献；建议 M1 之前与 M6 之后各读一遍。

## 一句话故事

> 一块 GPU。一个模型。一条基线。三条优化策略。
> 可测量的结果——每个主张都能追溯到一份已提交的运行记录。

## 模块

| # | 模块 | 方法论阶段 | 关键实测事实（来自案例研究） |
|---|---|---|---|
| 0 | [环境与纪律](M00_setup_and_discipline/LESSON.zh-CN.md) | — | doctor 失败即拒绝；证据先于主张 |
| 1 | [先测量：冻结基线](M01_baseline/LESSON.zh-CN.md) | 基线 | 热态 55.21 s、1.345 s/步（两个 9 次会话共 18 次，离散 0.3%） |
| 2 | [剖析：时间去向](M02_profiling/LESSON.zh-CN.md) | 剖析 | GEMM 61.3% / 注意力 32.4%（占 GPU 繁忙 97.8%） |
| 3 | [减少计算](M03_less_computation/LESSON.zh-CN.md) | 少算 | Turbo 8 步：4.55×——以及为什么不是 5× |
| 4 | [加速计算](M04_faster_computation/LESSON.zh-CN.md) | 快算 | ck 注意力 1.22×；INT8 1.08× + 显存 −6.7 GB |
| 5.0 | [PyTorch CPU 文本编码基线](M05_parallel_heterogeneous/LESSON.zh-CN.md) | 并行基线 | 实测 14× 吞吐*损失*（负结果） |
| 5.1 | [OpenVINO CPU 文本编码器](M05_parallel_heterogeneous/M5_1_OPENVINO_CPU_TE.zh-CN.md) | 研究延伸 | 同栈 W7900 A/B **2026-11-05 完成**（编码器正提示词 331–480×；热态单请求服务延迟 ≈8% 改善、实际四请求批量吞吐 ≈3% 改善；冷包含 1.30–1.33×；峰值显存省 17.26 GB；INT8 负向条件门 FAIL → 仅探索性；C1b-vs-G 人工质量判定已被所有者接受——范围见 [PR2_OWNER_QUALITY_VERDICT](../docs/release-closure/PR2_OWNER_QUALITY_VERDICT.md)；详见 [M51_RESULTS](../docs/M51_RESULTS.md)） |
| 6 | [组合优化与质量门](M06_combined_quality/LESSON.zh-CN.md) | 组合 + 质量 | BEST 9.02 s = 6.12×；质量判定独立进行中 |
| 7 | [学生挑战](M07_student_challenge/CHALLENGES.zh-CN.md) | 开源 | 7 个分级任务，各带验收标准 |

实验流程（冒烟 vs 正式、GPU 预算表）：
[docs/CLASSROOM_WORKFLOWS.zh-CN.md](../docs/CLASSROOM_WORKFLOWS.zh-CN.md)

另有：[LECTURE_DEMO.zh-CN.md](LECTURE_DEMO.zh-CN.md)——100 分钟课堂包
（8 页讲义 + 4 组互动问答 + 有界的教师演示与学生练习，全部由案例数据支撑）。

## 配套 Jupyter 实践

九个教学单元（M0–M7，包含 M5.1）均有[中文](../notebooks/README.zh-CN.md)与 [English](../notebooks/README.md) Notebook。默认 Run All **离线、CPU、只读**，直接重算真实 W7900 数据；GPU 冒烟需单独明确授权，不能作为正式发布基准。教学安排见[中文教师指南](../notebooks/INSTRUCTOR_GUIDE.zh-CN.md)。

## 评分建议

| 组成 | 权重 | 及格线 |
|---|---|---|
| 实验记录本（M1–M6 练习） | 40% | 命令与输出已提交、问题已回答 |
| 在自己 GPU 上复现基线 | 15% | schema 合法记录、离散 < 5% |
| 一项优化实验（A/B） | 25% | 成对测量 + 正确的语义标注 |
| 质量推理 | 10% | 不混淆保真度与距离；判定协议正确 |
| 挑战任务 PR（M7） | 10% | 证据齐备、主张 ≤ 证据 |

**学术诚信规则（本课程的核心价值）：** 没有运行记录的性能数字不是结果。伪造测量
按抄袭处理。如实报告的负结果得满分。

## 本课程五定律

1. **先测量再优化**——没有基线，就没有优化主张。
2. **一次一个变量**——每个 A/B 都有成对参考。
3. **永远不要把加速比相乘**——组合必须端到端实测。
4. **速度 ≠ 质量**——每个提速改动都要过声明语义的质量门；人工判定具有约束力。
5. **主张 ≤ 证据**——实践样貌见
   [docs/CLAIM_EVIDENCE_MATRIX.md](../docs/CLAIM_EVIDENCE_MATRIX.md)
   （25 条主张 → 文件）；机器可读登记表：
   [course/data/verified_claims.json](data/verified_claims.json)。

## 仓库地图

```
course/    教学模块（本目录）+ 讲义 + 方法论（中英对照）
lab/       可运行实验：python -m lab doctor|baseline|profile|run|report
evidence/  案例研究的全部测量（记录、网格、图表）——冻结不可改
docs/      复现指南、主张矩阵、故障排查、最终报告（含中文版）
docs/case-study/  P0–P6 工程日志（阅读作业）
docs/quality-v2/  course-v2 独立审计与开销研究
```
