# M7 — 学生挑战：Radeon AI 推理加速挑战

**学习目标（学完本课你能够）：**
1. 选择一个与自身资源匹配的真实缺口（纯 CPU / 需要 GPU / 高级），并按 Fork → 编辑 → 测试 → PR 的流程提交；
2. 为每个提交附上可检验的证据（前后输出、运行记录、表格），使主张 ≤ 证据；
3. 在 PR 中诚实报告负结果与未验证项，通过仓库所有者的评审。

**问题。** 你已经完成课程——现在为本开源项目做一点真实的贡献。下面的任务是本仓库
**真实存在的缺口**：每个都可独立检验、自带验收标准与证据要求，且没有一个已被实现
（已存在的功能——报告图表、quality CLI——列在"已实现"里，避免你重复提案）。

**所有任务的通用规则**

1. 先复现冻结的 B0 基线（`python -m lab baseline`）；你的改动只有相对它才有意义。
2. 每个实验一个变量；独立加速比永不相乘。
3. 没有运行记录（schema 合法、带注意力证据）就等于没发生。
4. 欢迎负结果——但要带测量。
5. 硬件按实测 VRAM 峰值诚实分级：bf16 40 步实验（B0、F1）峰值约 37.7 GB，需要
   gfx1100 级 48 GB 卡（W7900/W7900D）；int8/turbo 实验（F2、LF1–LF3、BEST）峰值
   约 31 GB，需要 ≥ 约 33 GB。更小的 Radeon（24 GB 级）可以做本课程的证据阅读练习，
   也可跑降分辨率/冒烟变体，但不承诺装下已提交的实验矩阵——云上 `doctor --cloud`
   对非 W7900 卡打 WARN 放行，你的数字是独立测量，绝不与参考合并。
6. 流程：Fork → 编辑 → 测试 → PR——遵循
   [docs/CONTRIBUTING_TUTORIAL.zh-CN.md](../../docs/CONTRIBUTING_TUTORIAL.zh-CN.md)。
   推送前 CPU 测试必须通过（`python -m pytest tests/ -v`）。

**已实现（不要作为新功能重复提案）：** `report` 的 SUMMARY 图表（柱状+瀑布，M2 有
链接）、带 manifest 语义的 `python -m lab quality` CLI、范围化报告
（`--scope reference|student|all`）、runs/ 证据隔离、云端分级 doctor。对它们的改进
可以作为任务 1–3 提交。

---

## 任务 1 — 快速开始与诊断加固（★☆☆ good-first-issue）

**缺口：** `scripts/cloud/start.sh` 只在*一台*被审计的实例上验证过（W7900、
ROCm 7.2.1、有代理）。其他云镜像（不同 ROCm 版本、无代理、其他 Radeon 型号、更小的
`/workspace`）可能踩到未处理的路径。

**做什么：** 在你的实例上运行脚本；当某步失败或降级时，修复脚本（或 `lab/doctor.py`）
以清晰检测并报告你的情况——绝不是为了掩盖它。

**预期证据：** 你硬件上的前后脚本输出 + PR 描述中的实例事实（GPU 架构、VRAM、ROCm、
可用磁盘）。

**验收：** 失败模式被检测并给出可行动的信息（无静默回退）；修复后重跑成功或报告确切
阻塞点；`bash -n` 干净；日志无密钥。

## 任务 2 — 证据导航：SUMMARY 链接与对比（★☆☆–★★☆）

**缺口：** 今天读证据要手工开 JSON。SUMMARY 表（参考 + 学生）可以给每行链接到其
运行目录、记录与质量网格——还可以加一个 `report --diff`，按实验对比学生与参考的
中位数、离散度与 n。

**做什么：** 扩展 `lab/report.py`（只改 markdown——图表已有），逐行输出相对链接，
并加一个只读对比视图。

**预期证据：** 重新生成的 SUMMARY.md（reference 范围的数字必须逐字节不变——这是
纯格式改动）+ 一份与参考对比的学生 SUMMARY。

**验收：** 纯 CPU 测试断言 (a) 表格单元格中参考数字逐字节不变，(b) 每个链接目标
存在，(c) 对比只列出两个范围都有的实验，带 n 与 Δ。

## 任务 3 — 质量工具回归测试（★☆☆ 语义敏感）

**缺口：** `lab/quality_gate.py`（网格、裁切、LPIPS 表、判定复选框）的 CLI 包装有
单元测试，但纯函数没有：`collect_measured` 文件名解析、`make_paired_grid` /
`make_native_crops` 几何、`lpips_table` 统计，以及 manifest 语义映射。

**做什么：** 用微型合成 PNG（Pillow）加纯 CPU pytest 覆盖——不需要 GPU、不需要
lpips 模型（mock `_lpips_pair_impl`）。

**预期证据：** 新测试文件在 CPU CI 上通过。

**验收：** 测试至少覆盖：奇/偶文件集、配对不匹配（空交集 → SystemExit）、裁切坐标
vs 画布大小、逐提示词中位数绝不跨提示词合并、每个 manifest 的
`quality_semantics` 保真度/距离映射。

## 任务 4 — 内核剖析调查：为什么 ck 在 int8-turbo 上毫无增益？
（★★☆ 中级，需要 GPU）

**背景：** M2 的剖析测得注意力占 GPU 繁忙的 32.4%（`attn_fwd`，平均 13.8 ms，
n=1248/代），而 ck 注意力让 bf16 流水线的 DiT 快 1.25×——却在 int8 turbo 路径上
无可测收益（LF2 vs LF3：中位 9.023 vs 9.022 s，各 n=9，分布完全重叠）。至今没有
人在内核层面解释原因。

**复现：** `python -m lab profile --date attn-deep`，然后跨一次 bf16 运行与一次
int8 运行对比 `rocprof-dit/analysis.json`（manifest F2/LF2，按 REPRODUCTION_GUIDE §3
的有界 `-P` 窗口）。

**预期产出：** bf16-DiT vs int8-DiT 的注意力内核调用表（名称、grid、时长），
解释 LF2=LF3。

**验收：** 解释引用已提交追踪中的内核行，不是猜想；附一段建议（还有余量吗？）。

**证据：** 追踪 CSV 很大——不要提交；提交分析 JSON + 一份 markdown 报告到
`docs/research/`。

## 任务 5 — INT8 内核分析：为什么 40 步只有 1.08×、turbo 上却有 1.35×？
（★★☆，需要 GPU）

**背景：** F2（int8、40 步）步时 1.220 s vs B0 1.345 s；LF2（turbo int8）
0.916 s vs L1 1.337 s。同样的量化格式，相对增益差别很大。假设空间：GEMM 形状相关
的反量化开销、减半权重的 L2 驻留、逐形状的内核选择。

**复现：** 用有界 rocprofv3 窗口剖析 F2 与 LF2；按形状给 GEMM 内核分桶（M,N,K 可
从 grid 维度推出）；对比 bf16 vs int8 每步 GEMM 总时长。

**验收：** 逐形状 GEMM 时间表 + 对 int8 路径时间去向（反量化内核 vs GEMM）的
量化陈述。

## 任务 6 — 单 GPU 批处理与吞吐实验（★★★ 高级，需要 GPU）

**背景：** 课程的所有测量都是 batch_size=1。单 GPU 上唯一未测的吞吐方向：把多张图
批进一次去噪（提高每步 GPU 利用率）vs 顺序 BEST 请求（已测边界：新提示词约
6.6 张/分）。

**做什么：** 新增 `BATCH1` manifest + 工作流变体（EmptyLatentImage batch_size=N；
提示词复制）——用*新*实验 ID，绝不编辑既有 manifest；在 N ∈ {2, 4} 下测量每图热态
延迟与每分钟图像数，对比顺序 BEST。

**验收：** 成对的每图延迟 + 吞吐表（N=1 vs N=2 vs N=4）、VRAM 峰值，以及诚实的
质量注记（批处理改变无批归一化的 DiT 数学吗？在宣称质量对等前先验证数值逐行独立）。

**证据：** 运行记录放 `runs/`（绝不放 `evidence/`）。

## 任务 7 — 2048² 探索（BLOCKED，直至所有者记录 1024² 质量判定）

任务门把 2048² 放在"稳定的 1028² + 质量批准"之后。在 48 GB 卡上，曾在 24 GB 卡上
OOM 的 bf16 VAE 解码也许装得下——但在人工判定落地之前不要运行。

---

**课程 v2 新增参考：** M5.1 之后的"编排开销"研究方向（热态服务 1.081× vs 实际批量
1.029× 的差距）已作为 course-v2 开销研究做了首次剖析
（[docs/quality-v2/PIPELINE_OVERHEAD_STUDY.md](../../docs/quality-v2/PIPELINE_OVERHEAD_STUDY.md)）——
在其"研究延伸"一节中仍有未做的后续假设，可作为任务 4/6 之外的 CPU/GPU 混合课题。

**提交：** PR 到 `main`，约定式提交，CPU 测试全绿（`pytest tests/ -v`），GPU 证据
以 `runs/` 下的运行记录目录附上。仓库所有者亲自评审所有质量判定。
