# 让生成式 AI 更快 —— 在免费 Radeon Cloud GPU 上的实测优化课程

**以 Qwen-Image-2.1 推理加速调优为例**（[English README](README.md)）
*一门动手的推理加速工程课程。下文每一条结论都能追溯到一份已提交的运行
记录 —— 测量于单张 AMD Radeon PRO W7900（gfx1100，48 GB，ROCm 7.2.1，
PyTorch 2.9.1）—— 而你将在免费申请的 Radeon Cloud GPU 上亲手复现整套方法论。*

---

## 实测结果（真实数据，不是宣传图）

同一提示词、同一随机种子、同一块 GPU —— 改变的只有工程手段：

| B0 冻结基线 · bf16，40 步 · **55.21 s** | BEST · turbo-int8，8 步 · **9.02 s** |
|:---:|:---:|
| <img src="evidence/samples/B0/p1-zh-scene-42-r0.png" width="380" alt="B0：江南水乡，55.21 s"> | <img src="evidence/samples/BEST/p1-zh-scene-42-r0.png" width="380" alt="BEST：同一提示词，9.02 s"> |
| <img src="evidence/samples/B0/p2-en-scene-42-r0.png" width="380" alt="B0：山脊徒步者"> | <img src="evidence/samples/BEST/p2-en-scene-42-r0.png" width="380" alt="BEST：同一提示词"> |
| <img src="evidence/samples/B0/p3-poster-text-42-r0.png" width="380" alt="B0：文字海报"> | <img src="evidence/samples/BEST/p3-poster-text-42-r0.png" width="380" alt="BEST：同一提示词 —— 文字是质量验收轴"> |

**端到端实测 6.12×**（暖态中位数；B0 汇总两个独立 9 次 session，
其余各 n=9，1024×1024）：

| 阶段 | 配置 | 暖态 | 相对基线 | 峰值显存 |
|---|---|---|---|---|
| **B0** | bf16、40 步（冻结基线） | 55.21 s | 1.00× | 37.7 GB |
| L1 | Turbo 8 步（更少计算） | 12.13 s | 4.55× | 37.7 GB |
| F1 | ck 注意力（更快计算） | 45.09 s | 1.22× | 37.7 GB |
| F2 | INT8 权重（首先是省显存：−6.7 GB） | 51.09 s | 1.08× | **31.0 GB** |
| **BEST** | Turbo INT8 + 默认注意力 | **9.02 s** | **6.12×** | **31.0 GB** |

> ⚠ **必须诚实说明的限制（本课程的核心价值）**：6.12× 的速度伴随一个
> **仍处于 `PENDING_HUMAN_REVIEW` 的质量终审**。Turbo 使用了不同的采样
> 调度，BEST 的图像与 B0 合理地存在差异（LPIPS 距离 0.19–0.30；INT8
> 量化本身近乎无损，保真 0.005–0.031，噪声底 0.0000）。Turbo 的输出是否
> 可接受 —— 尤其是原生分辨率下的海报文字 —— 是**仓库所有者尚未做出的
> 三档人工终审（无感/略降/不可接受）**。速度永远不能凌驾于质量门禁之上。
> 完整对比网格：[`evidence/quality/BEST/`](evidence/quality/BEST/quality_grid.png)，
> 原生分辨率文字裁切：
> [`quality_crops_poster.png`](evidence/quality/BEST/quality_crops_poster.png)。

另有两个**带证据的负结果**（课程中最有教学价值的部分）：CPU 文本编码
在数值等价（余弦 ≥ 0.999999）的前提下仍比 GPU 慢 152–391×；CPU-GPU
请求流水线即使有 50.4 秒**经时间戳证实的真重叠**，吞吐仍差 14×。

---

## 开始学习（三步，约 15 分钟出第一张图）

**① 申请免费 Radeon Cloud GPU。** 申请免费的 AMD Radeon Cloud GPU 算力
（平台：<https://developer.amd.com.cn/radeon>，文档：
<https://amd-aim.github.io/radeon-cloud-docs/zh-cn/>），然后启动一台
**单 GPU、预装 PyTorch-ROCm 的 Linux Notebook**。配额（时长、GPU 型号、
存储）以平台控制台显示为准 —— 本课程从不承诺具体数字。
[一台真实实例的就绪审计 →](docs/RADEON_CLOUD_READINESS.md)

**② 把课程克隆到 `/workspace`：**

```bash
cd /workspace
git clone https://github.com/AIwork4me/qwen-image-inference-tuning-course.git
```

**③ 运行一条命令，拿到你的第一张图：**

```bash
cd qwen-image-inference-tuning-course
bash scripts/cloud/start.sh
```

脚本会：诊断你的实例（GPU/ROCm/PyTorch/磁盘/代理，不需要任何密钥）→
以只读方式复用预装的 ROCm PyTorch（绝不重装）→ 获取固定提交的 ComfyUI →
分阶段下载 SHA256 锁定的最小模型集（约 25.5 GB，可断点续传）→ 运行
fail-closed 体检 → 生成一张有界的冒烟图并打印路径与下一课。
[完整快速上手 + 经过实测的故障恢复指南 →](docs/RADEON_CLOUD_QUICKSTART.md)

---

## 你将学到什么：让推理变快的三条路

**① 更少计算（Less Computation）** —— 先删掉工作，而不是加速它。
步数蒸馏：40 → 8 步去噪是最大的单杠杆（4.55×）。但它*不是* 5×
（Amdahl 定律把上限压在 ~4.6×），每步耗时不变（1.345 → 1.337 s ——
证明收益纯粹来自删除的计算），而且输出合理地改变了 → 这是必须由
人来裁决的质量权衡。

**② 更快计算（Faster Computation）** —— 同样的工作，更快的内核。
注意力后端（ck：1.22×）与 INT8 量化（1024² 下 1.08×，但省 6.7 GB 显存
—— 在小显存卡上这是"能跑与不能跑"的差别）。硬件利用率不等于延迟：
只有配对的端到端测量才算数，而且独立收益永远不能相乘
（4.55 × 1.08 × 1.25 会"预测"出 6.14×；实测 6.12× —— 而 ck 在
int8-turbo 路径上无可测收益：LF2 与 LF3 中位数 9.023 vs 9.022 s，
各 n=9，分布完全重叠）。

**③ 并行与异构计算（Parallel & Heterogeneous Computing）** —— 单 GPU
+ 单 CPU 做不到的事。我们实测了那个诱人的想法（CPU 文本编码与 GPU
去噪重叠），结果**慢了 14×**：重叠是真的（50.4 秒，时间戳可证），但
8B 的 CPU 编码器慢 152–391×，而且它的矩阵乘会饿死 GPU 的启动线程。
带证据的负结果比乐观的猜测更有价值。

## 课程结构（8 学时 + 实验 + 100 分钟讲义）

| 模块 | 讲授 | 关键实测事实 |
|---|---|---|
| [M0 环境与纪律](course/M00_setup_and_discipline/LESSON.md) | fail-closed 预检、证据文化 | doctor 门禁、全栈锁定 |
| [M1 基线](course/M01_baseline/LESSON.md) | 先测量再优化 | 55.21 s、1.345 s/步、可复现性 |
| [M2 剖析](course/M02_profiling/LESSON.md) | 流水线+内核追踪、Amdahl | GEMM 61.3% / 注意力 32.4%（GPU 忙时） |
| [M3 更少计算](course/M03_less_computation/LESSON.md) | 步数蒸馏、归因 | 4.55× —— 以及为什么不是 5× |
| [M4 更快计算](course/M04_faster_computation/LESSON.md) | 注意力后端、INT8、组合 | ck 1.22×；INT8 1.08× + 省显存；LF2=LF3 |
| [M5 并行与异构](course/M05_parallel_heterogeneous/LESSON.md) | 单 GPU + 单 CPU 做不到什么 | 重叠可证、吞吐反而差 14× |
| [M6 组合与质量门禁](course/M06_combined_quality/LESSON.md) | BEST 选取、LPIPS 语义 | 6.12×；人工终审具有约束力 |
| [M7 学生挑战](course/M07_student_challenge/CHALLENGES.md) | 开源贡献实践 | 分级任务 + Fork→PR 教程 |

完整计划：[course/SYLLABUS.md](course/SYLLABUS.md) ·
讲义（2026-11-19，100 分钟）：[course/LECTURE_DEMO.md](course/LECTURE_DEMO.md)。

## 亲手运行实验

```bash
bash scripts/cloud/start.sh --all-models   # 分阶段环境准备（也可稍后补模型）
.venv-lab/bin/python -m lab doctor         # 严格预检（正式测量前必须 PASS）
.venv-lab/bin/python -m lab baseline       # 冻结 B0 基线（W7900 级约 16 分钟）
.venv-lab/bin/python -m lab profile        # 流水线 + rocprofv3 内核剖析
.venv-lab/bin/python -m lab run --experiment L1 [--smoke]   # F1 F2 LF1-3 BEST H1 H2
.venv-lab/bin/python -m lab quality --experiment F2
.venv-lab/bin/python -m lab report [--scope reference|student|all] [--write]
#   （默认只读；--kind smoke|all 过滤学生 smoke 运行；--write 才写出
#    SUMMARY+图表 —— reference scope 此时改写冻结证据文件，属于刻意的
#    维护动作）
```

**GPU 时间是计量付费的学生资源。** 课堂练习使用有界的 `--smoke` 变体
（1–3 分钟）；完整协议 session 每个 4–16 分钟（预算表见
[快速上手](docs/RADEON_CLOUD_QUICKSTART.md)）。耗时的 CPU 实验（H1/H2）
默认以已提交的证据讲授，自愿复现为进阶研究任务。不用实例时请及时停止或
销毁 —— `runs/` 随 `/workspace` 持久保留。你的测量写入 `runs/`，**永远
不可能**修改 `evidence/` 下的冻结参考证据。

## 证据与主张

- `evidence/results/` —— 全部运行记录（计时、显存、注意力路径日志证据、
  图像哈希）、环境快照、剖析结果、SUMMARY 与图表。官方测量活动固定在
  [`lab/reference_runs.json`](lab/reference_runs.json)；新测量落在 `runs/`
  （代码级隔离，用 `report --scope student --write` 单独汇总；
  smoke 运行默认不计入正式中位数）。
- `evidence/quality/` —— 成对网格 + 原生分辨率文字裁切（供人工终审）；
  `evidence/samples/` —— B0 与 BEST 的实测图像。
- [docs/CLAIM_EVIDENCE_MATRIX.md](docs/CLAIM_EVIDENCE_MATRIX.md) ——
  每条主张对应的证据文件与限制说明。
- [docs/case-study/](docs/case-study/) —— P0–P6 工程日志原文（阅读作业）。
- [docs/FINAL_ENGINEERING_REPORT.md](docs/FINAL_ENGINEERING_REPORT.md) ——
  摘要、结论与建议。

## 五条定律（本课程真正教的东西）

1. 先测量，再优化。
2. 一次只改一个变量。
3. 永远不要把加速比相乘 —— 组合必须端到端实测。
4. 速度 ≠ 质量；人工终审具有约束力。
5. 主张 ≤ 证据。

## 出处与许可

本课程派生自社区基准测试项目
[AIwork4me/qwen-image-rocm](https://github.com/AIwork4me/qwen-image-rocm)
（MIT）—— 其 Linux W7900 工作流提供了实验代码与全部测量。代码：
**MIT**。课程文本、证据数据与报告：**CC-BY-4.0**。模型权重不包含在
仓库中，遵循其上游许可。详见 [LICENSE](LICENSE)、
[DATA_LICENSE.md](DATA_LICENSE.md)、[ATTRIBUTION.md](ATTRIBUTION.md)。

独立的教学材料 —— 非 AMD 或阿里/Qwen 官方项目。
