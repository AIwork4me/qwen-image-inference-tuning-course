# M5.1 — 用 OpenVINO 让 CPU 文本编码变快

## 以 Notebook 方式运行（三种声明模式）

每个模块配有中英文对照 Jupyter Notebook。在第一个单元设置
`COURSE_MODE`：`REFERENCE`（离线 CPU 分析冻结实验数据）、`GPU_CLASSROOM`
（你自己的 W7900 有界物理练习，每臂约 5 分钟——默认实验体验）或
`GPU_RESEARCH`（本文档所述完整协议）。见
[notebooks/README.zh-CN.md](../../notebooks/README.zh-CN.md)。

**在你自己的 W7900 上实测（首选）。** 在 `GPU_CLASSROOM` 且默认
`RUN_OWN_GPU_ARMS = True` 时，Notebook 06 会在全新克隆上执行**你自己的**完整
M5.1 流程：严格的 BF16 工件准入（`PASS` / `PASS_WITH_WARNINGS` / `FAIL` /
`BLOCKED`，fail closed）、你自己的 GPU / CPU PyTorch TE 试点、你自己的 H3
OpenVINO BF16 运行，以及你自己的物理 A/C 四请求批次——每个数字都标注来源
（`CURRENT_SESSION_PHYSICAL` … `COMMITTED_PHYSICAL_REFERENCE`，见
[学生溯源契约](../../docs/notebook-gpu-qualification/STUDENT_PROVENANCE_CONTRACT.md)）。
如果你选择不消耗新的 GPU 时间，复用必须给出**精确 run id** 并标注为复用——
本讲引用的已提交实验数字永远是参考值，绝不是你的测量。

**学习目标（学完本课你能够）：**
1. 设计一个受控的编码器 A/B（G/C0/C1a/C1b/C2 五臂），说明每臂控制的因果轴（后端、执行精度、量化）；
2. 用条件张量对齐数据（rel L2 / cosine / token IDs）判定数值门，并解释"331–480×"必须限定在正提示词范围；
3. 把"冷包含 1.33× / 热态服务 1.081× / 实际批量吞吐 1.029×"三个数字分别归因（CLIP 重载、编码占比、编排开销），拒绝混为一谈；
4. 陈述 INT8 空负向门 FAIL（28.25%）为何不可被任何主观接受推翻，以及它在 cfg=1 与 cfg>1 下的不同后果。

**M5.0 基线：** [PyTorch CPU 下放与四请求流水线](LESSON.zh-CN.md)
**实验状态：** **已于 2026-11-05 在物理 gfx1100 48 GiB W7900 级 GPU 上完成。**
**决策：** OpenVINO BF16 CPU 部署 **GO**；**INT8 仅为探索性**（负向条件门失败）。
**完整证据：** [M51_RESULTS.md](../../docs/M51_RESULTS.md) · 运行 `runs/m51-20261105-cleanroom-01/`。

---

## 1. 两项真实的上游研究（历史参考，非本地测量）

| 研究 | 测的硬件与结果 | 可复用什么——什么不可 |
|---|---|---|
| [qwen3vl-cpu-benchmark](https://github.com/AIwork4me/qwen3vl-cpu-benchmark) | Ryzen AI MAX+ PRO 395 (Zen5)：OpenVINO **INT8** 条件编码 **0.187 / 0.377 / 0.687 s** vs ComfyUI **INT8 ConvRot PyTorch CPU 产品路径** **12.98 / 13.73 / 15.43 s**，即编码器 **22–69×** | 可复用其[条件桥接](https://github.com/AIwork4me/qwen3vl-cpu-benchmark/blob/main/scripts/ov_encoder.py)与[定制节点](https://github.com/AIwork4me/qwen3vl-cpu-benchmark/blob/main/custom_nodes/qwen3vl_openvino_cpu/__init__.py)作参考；**不要**把 22–69× 套到本课程的 **PyTorch FP32 手动转型** M5.0 上 |
| [qwen-image21-openvino-alignment](https://github.com/AIwork4me/qwen-image21-openvino-alignment) | EPYC 9334 (Zen4) + W7900D gfx1100：OV CPU BF16 **0.39–1.33 s**、INT8 **0.22–1.08 s**（25–171 token）；干净 OV FP32 vs PyTorch FP32 条件 rel L2 **7.7e−6** | 可复用其[数值对齐方法](https://github.com/AIwork4me/qwen-image21-openvino-alignment/blob/main/CONCLUSIONS.md)、[复现指南](https://github.com/AIwork4me/qwen-image21-openvino-alignment/blob/main/reports/REPRODUCTION.md)、[局限](https://github.com/AIwork4me/qwen-image21-openvino-alignment/blob/main/reports/LIMITATIONS.md)；其工件/ROCm 栈不同 |

**重要：** Zen5 的 22–69× 是 **INT8-vs-INT8** 后端对比；本课程的 C0 是 ComfyUI PyTorch
**FP32 手动转型计算**（BF16 存储权重）。本地 M5.1 运行不把那些历史比值当成本地结果。

---

## 2. 受控实验

| 臂 | 文本编码器 | 硬件 | 主要对比 |
|---|---|---|---|
| **G — 参考** | ComfyUI / PyTorch | GPU | GPU 编码 + GPU DiT/VAE |
| **C0 — M5.0 基线** | ComfyUI / PyTorch（BF16 存储、FP32 手动转型） | **CPU** | 冻结的 CPU PyTorch 参考 |
| **C1a — 对齐** | OpenVINO **FP32** | **同一 CPU** | 后端 + 高精度参考 |
| **C1b — 主候选** | OpenVINO **BF16 执行**（FP16 存储部署 IR） | **同一 CPU** | **主 C1 vs C0 对比** |
| **C2 — 可选** | OpenVINO **INT8 权重压缩** | 同一 CPU | 量化 A/B，单独判定 |

控制变量：冻结提示词与 token IDs、同一 TE 权重谱系、两 CPU 臂同 16 线程 cgroup 预算、
同一 DiT（turbo BF16）、VAE、调度器、8 步、cfg 1.0、种子 42、1024²。

### 条件契约

```text
ComfyUI QwenImage21Tokenizer（T2I 系统前缀, thinking=True）
        |
        +--> C0 PyTorch CPU（FP32 手动转型）
        +--> C1a/C1b/C2 OpenVINO CPU
        |
   第 36 层残差，最终 RMSNorm 之前
   在第二个 <|im_start|> 处截断
   [1, seq_kept, 4096] FP32
        |
   LinuxLabCondSafetensors（安全，无 pickle）
        |
   原版 W7900 GPU DiT -> 原版 VAE -> PNG
```

不用 logits、不用 RMSNorm 之后的特征、不用通用 chat 模板、不造假张量。

---

## 3. 本地实测结果（2026-11-05）

### 3.1 同 CPU 编码器 A/B

| 臂 | p1 66 tok | p2 51 tok | p3 78 tok | 空负向 | 冷→首请求 | CPU RSS |
|---|---:|---:|---:|---:|---:|---:|
| C0 PyTorch CPU | **176.02 s** | **136.81 s** | **207.45 s** | **61.51 s** | 179.4 s | 34.5 GB |
| C1a OV FP32 | 0.779 s | 0.648 s | 0.875 s | 0.398 s | 95.0 s | 50.5 GB |
| **C1b OV BF16** | **0.532 s** | **0.285 s** | **0.555 s** | **0.187 s** | **5.5 s** | **30.0 GB** |
| C2 OV INT8 | 0.361 s | 0.249 s | 0.406 s | 0.144 s | 7.2 s | 16.8 GB |

相对 C0 加速：**C1a 226/211/237/155×**，**C1b 331/480/374/328×**，**C2 488/549/511/427×**
（p1/p2/p3/负向）。C0 数字在 1% 内复现了历史 M5.0 CPU 证据（177.8/136.8/207.1 s）。
**引用 331–480× 时必须限定"正提示词"**——空负向是 328×。

### 3.2 数值门

| 对比 | 正向 rel L2 | 负向 rel L2 | 状态 |
|---|---:|---:|---|
| C1a OV FP32 vs C0 | **3–5×10⁻⁶** | 5.4×10⁻⁶ | **PASS** |
| **C1b OV BF16 vs C0** | **1.09–1.25%** | **1.91%** | **PASS** |
| C2 OV INT8 vs C1a | **5.6–6.4%** | **28.25%** | 正向 PASS，**负向 FAIL** |

（另有 C1b-未压缩变体——FP32 存储 IR + BF16 执行——负向 6.63% **FAIL**，保留作精度参考。）
token IDs 完全一致；形状 `[1,52/37/64/8,4096]`；张量全部有限；每臂内重复哈希一致。

### 3.3 W7900 上的真实图像（turbo 8 步、1024²）

| 臂 | 图像延迟 | PSNR vs G | SSIM vs G | LPIPS vs G | VRAM 峰值 |
|---|---:|---:|---:|---:|---:|
| G 原生 GPU TE | 13.03 s | — | — | — | 34.15 GB |
| C1a OV FP32 | 12.03 s* | 34.74–48.62 dB | 0.9885–0.9974 | 0.0015–0.0106 | 16.95 GB |
| **C1b OV BF16** | **12.03 s*** | **35.57–47.03 dB** | **0.9876–0.9972** | **0.0019–0.0080** | **16.77 GB** |
| C2 OV INT8 | 12.04 s* | 27.37–38.77 dB | 0.9483–0.9941 | 0.0081–0.0378 | 16.92 GB |
| C0 原生 CPU TE | **264.49 s** | 34.97 dB | 0.9887 | 0.0090 | 33.10 GB（峰值） |

\* 预计算条件；编码器本身 0.14–0.56 s，不在 12.03 s 图像时间内。串行新提示词 C1b ≈ **12.58 s**。

**VRAM：** 全流水线峰值 **34.15 → 16.77 GB（节省 17.26 GB）**；驻留 **33.09 → 16.29 GB
（节省 16.80 GB）**。这是该配置下实测的全流水线效应，不是从 16.8 GB TE 分配推出来的。
（另一口径：平衡审计会话中峰值 34.15 → 17.98 GB，即 **16.18 GB**——两个数字、两种
定义，不可互换。）

图像网格：`runs/m51-20261105-cleanroom-01/e2e/grids/`；原生分辨率中文海报裁切：
`e2e/typography/p3_poster_zh_native_crop.jpg`。**人工质量判定：所有者已于 2026-11-05
接受（PASS），范围仅限 C1b 对比 G**——出处与非传播范围见
[`PR2_OWNER_QUALITY_VERDICT.md`](../../docs/release-closure/PR2_OWNER_QUALITY_VERDICT.md)；
INT8 与 FP32 存储 IR 变体仍为 FAIL。

### 3.4 四请求流水线——以及三个不同的"加速比"

| 臂 | 总时 | 充填 | CPU TE | GPU 去噪 | 每分钟图像 | 瓶颈 |
|---|---:|---:|---:|---:|---:|---|
| A GPU TE 顺序 | 66.62 s | — | — | 13.03 s | 3.60（稳态估算 4.60） | GPU |
| B PyTorch CPU TE | **680.00 s** | 176.17 s | **147.14 s** | 12.07 s | **0.35** | **CPU** |
| **C OV BF16** | **50.05 s** | 0.56 s | 0.436 s | 12.19 s | **4.80** | GPU |
| **D OV INT8** | **49.75 s** | 0.40 s | 0.405 s | 12.13 s | **4.82** | GPU |

B 是 A 的 **0.098×**（慢 10 倍）。重叠由时间戳证明：B 37.09 s、C 1.62 s、D 1.40 s。
16 线程 CPU 负载下无可测的 GPU 减速。

**不要把 C-vs-A 压成一个数字——有三个，各自回答不同的问题**（发布闭环审计独立实测，
热态 n=8/8、区间不重叠；原始 JSON：[`PR2_LATENCY_VS_THROUGHPUT.json`](../../docs/release-closure/PR2_LATENCY_VS_THROUGHPUT.json)）：

| 口径 | 数值 | 含义 |
|---|---:|---|
| 冷包含批量比 | **1.33×**（66.62 → 50.05 s） | 从空闲起的一个会话：C 免于重载 16.8 GB GPU CLIP |
| 热态单请求**服务延迟** | **1.081×**（13.031 → 12.060 s） | 一个请求者等待每张图的时间 |
| 热态**实际四请求批量吞吐** | **1.029×**（4.42 → 4.55 张/分） | 含请求间编排的每分钟图像数 |

历史上"**≈1.04× 稳态**"的数字是战役对最后一行的*估算*；审计用直接测量取代了它。
为什么吞吐收益（2.9%）小于服务收益（8.1%）？预计算臂每请求多付约 0.54 s 的编排
（条件暂存 + 工作流提交），而 A 自己的间隙（中位约 0.45 s）部分与其 GPU TE 工作
重叠。用服务时间求和算出的"张/分"必须标注为朴素上界（A：朴素 4.60 vs 实际 4.42；
C：朴素 4.98 vs 实际 4.55）。

### 3.5 为什么端到端收益这么小（Amdahl）

```text
顺序周期   = GPU_TE + GPU 去噪 ≈ 0.60 + 12.43 = 13.03 s
流水线周期 = max(OV_TE, 负载下 GPU 去噪) + 交接
           ≈ max(0.44, 12.19) ≈ 12.19 s
仅编码器   = 176.02 / 0.532 = 331×
周期估算   = 13.03 / 12.19 ≈ 1.07×
```

编码器约占图像周期的 4%。它一旦被藏进流水线，再快 300 倍也打不破去噪上界——
周期估算 1.04–1.07× 恰好包住**实测**热态服务比 **1.081×**。冷包含 1.33× 主要来自
**避免 16.8 GB GPU CLIP 重载**，不是重叠。而**实际批量吞吐**被压得更低（1.029×），
因为填满服务收益缺口的是编排开销——不是编码器时间。三个指标、三种机制、同一个
Amdahl 天花板。

---

## 4. 实现状态

实验代码提供：

```bash
# 可选工件准备（不属于默认 M0–M4 引导）
bash scripts/m51/provision_ov.sh

# 有界冒烟与正式运行
.venv-lab/bin/python -m lab run --experiment H3_OV_BF16 --smoke
.venv-lab/bin/python -m lab run --experiment H3_OV_BF16
.venv-lab/bin/python -m lab run --experiment H3_OV_INT8 --smoke
```

CLI 检查隔离的 `.venv-ov`、工件清单、磁盘/内存预算，并写出显式的 `PASS`/`FAIL`/
`BLOCKED` 记录。若可选工件缺失，它打印准备命令并拒绝伪造运行。所有者判定记录
（`docs/release-closure/PR2_OWNER_QUALITY_VERDICT.json`）现在会让 H3_OV_BF16 的
人工质量门传播为 PASS（范围严格匹配）；H3_OV_INT8 / H3_OV_FP32 保持 PENDING。

实现文件：`lab/ov_export.py`、`lab/ov_compress.py`、`lab/ov_quantize.py`、
`lab/ov_encoder.py`、`lab/ov_worker.py`、`lab/pt_worker.py`、
`lab/custom_nodes/linuxlab_cond_safetensors.py`、`lab/m51_*.py`、
`scripts/m51/provision_ov.sh`、`tests/test_m51_contract.py`。

---

## 5. 学生练习（按难度分层）

**核心问题：300× 的编码器加速，有多少能活到图像流水线？**

- 入门：读上面的实测表，计算顺序周期中编码器的占比 `TE / (TE + KSampler + VAE)`；
  用流水线 JSON 验证 C 的 CPU 编码区间落在 GPU 去噪区间之内。
- 进阶：计算盈亏平衡——去噪时间是多少时，0.53 s 的编码器会占周期的 50%？
  （答：约 0.53 s；本 GPU 去噪约 12 s。）对比 C1b 的 1.25% 条件差与 C0 原生的
  图像差异指标：为什么 7e−6 的 FP32 条件差也能产生 LPIPS ≈ 0.01？
  （扩散放大微扰；图像差异指标不是质量判定。）
- 挑战：解释 INT8 的 28% 负向条件误差为何在 cfg>1 时要紧、cfg=1 时不要紧
  （cfg=1 不消费负向分支；cfg>1 消费）。再解释三个"加速比"（1.33×/1.081×/1.029×）
  各自的机制，以及为什么把它们混成一个数字必然误导。
- **科学公平性（出版收尾练习）：** 一位开发者把 C 客户端从每四请求批 52.76 s
  优化到 50.32 s（持久监视器，+4.85%，运行
  `overnight-course-v2-a7fa69a-p2ab-152000`），然后拿 **C-fast 对比 A-original**
  （4.77 对 4.42 张/分，"1.079×"）。为什么这**不是**对 OpenVINO 架构收益的
  公正估计？设计一个把"客户端效应"与"架构效应"分开的实验。
  （答案：A-original 同样背负每请求约 0.5–1.0 s 的监视器拆除开销（两个臂都
  实测过），所以该对比在两臂之间混用了两代客户端——"收益"里有一部分是在修
  基线的测量装置，而不是部署本身的收益。公正的估计需要 2×2 因子设计
  {A, C} × {原客户端, 快客户端}、同样热会话、交错批次：架构效应是**同等客户端
  下**的 A 对 C；客户端效应是**各臂内部**的原对快；交互项说明修客户端是掩盖
  还是揭示架构差异。实测答案见 P4 公平客户端因子实验
  [`docs/publication-closure/P4_FAIR_CLIENT_AB.md`](../../docs/publication-closure/P4_FAIR_CLIENT_AB.md)
  与注册条目 `RATIO-CFAST-VS-AWARM-1.079`——后者仅作为该因子实验的声明的
  跨总体动机保留。）

**验收：** 学生能解释条件桥接、计算公平比值、复现盈亏平衡，并拒绝把 331× 编码器
增益说成 331× 出图增益。

### 亲手复现公平性实验（已验证命令）

环境要求：物理 W7900 级 gfx1100 主机、战役级 cgroup 预算（16 CPU / 55 GiB）、
固定技术栈（`.venv-lab`、`.venv-ov`、OpenVINO 2026.4.1、ComfyUI 固定检出）、
以及已制备的 OpenVINO 工件（见上文 OV 制备小节；`M51_OV_ARTIFACTS` 是**你的**
工件根目录——`/opt/m51-artifacts` 只是本课程主机的示例位置）。

```bash
# 1) 工件校验准入（按 conversion_manifest.json 中钉住的摘要/大小/身份逐文件
#    流式校验；不下载、不修改）。不要把 JSON 清单直接喂给 sha256sum -c ——
#    它不是 GNU 校验清单格式：
export M51_OV_ARTIFACTS=/opt/m51-artifacts
bash scripts/m51/provision_ov.sh --check-only        # -> ARTIFACTS_VERIFIED

# 2) 干净工作区检查 —— 运行会记录精确的源码身份：
git status --porcelain                                # 应无任何输出

# 3) 2x2 因子实验（全程相同的冻结提示词/种子/初始 latent）：
M51_OV_ARTIFACTS=$M51_OV_ARTIFACTS .venv-lab/bin/python -m lab.m51_pipeline_fair_ab \
  --run runs/<new-unique-run-id> --port 8213 --pairs 3 --blocks 2
```

- 预期输出：`<run>/pipeline_profile/pipeline_profile.json` —— 每格批次墙
  （每格 n=6）、每请求时间线、每提示词条件 SHA256、VRAM/RSS，以及
  `source_provenance` 源码清单（逐模块 SHA256；仅当工作区干净时才标记
  `EXACT_SOURCE_VERIFIED`）。
- 预算：完整 96 请求设计约 25 分钟 GPU；`--pairs 1 --blocks 2` 约 15 分钟
  （每格 2 批）的有界回归——这种小样本只能验证方向，绝不能确立统计等价。
- 复核你的重跑：将 `analysis.views` 各格与
  `docs/publication-closure/P6_P4_INDEPENDENT_REDERIVATION.md` 对比，并核对
  每提示词条件 SHA256 与本研究记录逐字节一致。
- 局限：单主机、单会话范围；描述性统计（批次区间不重叠，不报 p 值）；原战役
  的源码溯源为 `SOURCE_RECONSTRUCTED_WITH_EVIDENCE`（见
  `docs/publication-closure/P6_SOURCE_PROVENANCE_AUDIT.md`）——这正是运行器
  现在快照源码身份的原因。

---

## 6. 思考——带实测答案

1. **为什么 Zen5 的 22–69× 回答不了本课程的 C0→C1 问题？** 它对比的是不同的 CPU、
   权重、线程预算与 INT8-vs-INT8 路径。本地 C1b vs C0 是 328–480×，但对手是本课程
   的 FP32 手动转型 CPU 路径。
2. **若 GPU 去噪 10.8 s、GPU TE 0.6 s、OV CPU TE 0.44 s、交接约 0.2 s，周期加速
   是多少？** `(10.8+0.6)/(max(0.44,10.8)+0.2) ≈ 11.4/11.0 ≈ 1.04×`，不是 20×。
   实测：服务延迟 1.081×、实际批量吞吐 1.029×——周期估算落在两者之间，因为它
   不含请求间编排。
3. **为什么 cosine ≥ 0.997 不等于排版自动 PASS？** 扩散会放大微小条件差；要看
   原生分辨率裁切与人工评审。我们的 p3 裁切是证据，不是 PASS。
4. **BF16-vs-BF16 后端对比，还是 BF16-vs-INT8？** C1b vs C0 改的是后端 + BF16
   执行；C2 vs C1a 改的是量化。它们是两条独立的因果轴。

---

## 7. 学生必须引用的局限

- turbo 8 步、cfg 1.0、一套种子；C1b-vs-G 的人工质量判定已被所有者接受（范围与
  非传播见 `docs/release-closure/PR2_OWNER_QUALITY_VERDICT.md`）。
- C2 INT8 未通过空负向门（28.25%）；不得称为生产可用。
- C0 每提示词仅 1 个热态样本（另有两个编码重复，差 <0.3%）；审计补了 3 组配对
  会话（p2 加速 480.7×，n=3/6）；p1/p3 仍是单会话。
- 主机是 EPYC 9334 的 16-CPU/55 GiB cgroup，不是整机。
- GPU 驱动字符串为 `AMD Radeon Graphics` gfx1100 48 GiB（W7900 级，非字面产品名）。
- C1b 的 BF16 指**执行精度**，不是 BF16 权重存储。
- 不要把 Zen5 的 22–69× 当作本地结果引用。
