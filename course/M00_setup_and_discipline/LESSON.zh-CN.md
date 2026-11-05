# M0 — 环境搭建与实验纪律

**学习目标（学完本课你能够）：**
1. 用一条命令在 Radeon Cloud GPU 实例上完成课程环境搭建，并解释它做了什么、没做什么；
2. 说出 fail-closed（失败即拒绝）预检的 7 项检查各自防御什么事故；
3. 解释 SHA256 固定（pinning）与"戳记缓存"（stamp-cache）的验证语义，以及它仍然能抓到的攻击；
4. 区分 cold / fresh / warm / boot 四种延迟口径，并说明为什么混用它们是经典错误。

**问题。** 你刚拿到一台云 GPU，想立刻让图像模型变快。专业人士的第一步恰恰相反：
*在动手之前先证明环境可信*。一个悄悄回退到 CPU 的基准、一个半夜自动升级的服务端，
产出的都是"自信的胡说八道"——而你还会把它发表出去。

**概念（先给初学者的术语表）。**

| 术语 | 通俗含义 | 在本课为什么重要 |
|---|---|---|
| **GPU / VRAM** | 加速卡及其专用显存（W7900 级：48 GB） | 模型必须*装得下*；权重+激活都算数 |
| **ROCm** | AMD 的 GPU 软件栈（驱动+库），对应 NVIDIA 的 CUDA | "torch+rocm" 表示为 AMD GPU 构建的 PyTorch |
| **gfx1100** | GPU 架构世代（RDNA3, Navi31） | 内核行为逐代不同；我们的证据来自 gfx1100 |
| **ComfyUI** | 通过 HTTP 运行扩散模型的节点图服务端 | 我们*穿过*它做基准测试，就像真实部署 |
| **固定提交 / SHA256 pin** | 冻结软件版本 / 文件的密码学指纹 | 会漂移的服务端或损坏的权重会让一切对比失效 |
| **fail-closed** | 检查无法执行时，工具*拒绝*继续 | 反面（fail-open）什么都测不了却照样报数 |
| **warm / cold / fresh** | 重复 / 首次 / 新提示词运行 | 延迟不同；混用是 M1 要讲的经典错误 |

冻结的技术栈：

| 层 | 我们用什么 | 为什么固定 |
|---|---|---|
| GPU | 一块 AMD Radeon gfx1100 级（参考战役为 W7900 48 GB） | 只做单 GPU 主张 |
| OS / 栈 | Linux, ROCm 7.2.1, torch 2.9.1+rocm | 经过验证的环境；研究中绝不"升级" |
| 服务端 | ComfyUI v0.43.0 @ 提交 `b5a4321` | 服务端一动，所有对比作废 |
| 权重 | Comfy-Org/Qwen-Image-2.1 @ `aacf31a6`，SHA256 固定 | 完整性锚定在仓库内（`lab/weights.py`） |
| 协议 | 3 提示词 × 3 种子，euler/simple/CFG 1 | 冻结，使所有 A/B 可比 |

**运行（课堂用时：首次约 15 分钟，日常 2 分钟）。**

```bash
bash scripts/cloud/start.sh          # 一条命令完成搭建（M0 的实践核心）
.venv-lab/bin/python -m lab doctor --cloud   # 分级预检
.venv-lab/bin/python -m lab doctor           # 严格预检（正式实验）
```

预算：首次 `start.sh` 下载约 25.5 GB（实测云网络 10–20 分钟）；重跑只需数秒完成校验。
`doctor` 的 GPU 开销远小于 1 分钟（只启动一个极小的注意力内核）。

**观察（来自一台真实 Radeon Cloud 实例的输出，2026-11-04）。**

```
PASS gpu_amd_single_working: AMD Radeon Graphics / gfx1100 / 49136MB
PASS torch_rocm: torch 2.9.1+gitff65f5b hip 7.2.53211-e1a6bc5663
PASS comfyui_pinned: v0.43.0 @ b5a432157de6
PASS weights_stage1_integrity: {...逐文件判定...}
PASS prompts_frozen: 3 prompts x 3 seeds, ids intact
PASS flash_sdpa_probe: flash attention launches + syncs on gfx1100
PASS rocprofv3: version: 1.1.0
verdict: PASS  evidence: runs/doctor.json
```

再看一个真实的失败——失败比成功教得更多。一个损坏（大小不符）的权重文件：

```
FAIL weights_stage1_integrity: {'per_file': [
  {'rel': 'diffusion_models/qwen_image_2.1_turbo_int8_convrot.safetensors',
   'ok': False, 'reason': 'size-mismatch',
   'local_size': 5968230400, 'expected_size': 7256787829, ...}]}
```

doctor 的输出写入 `runs/doctor.json`——仓库自带的 `evidence/results/doctor.json`
属于冻结战役，永远不会被你的运行覆盖。

**解释。** 这套纪律请读两遍：

1. **Fail-closed 预检** —— doctor 拒绝在错误的 GPU、漂移的 ComfyUI 或损坏的权重上
   测量。读懂失败信息，修复原因，重跑。
2. **证据而非轶事** —— 每次运行写出 schema 合法的 JSON 记录（计时、VRAM 采样、
   从服务端日志抓取的注意力路径证据、图像 SHA256、环境快照）。记录只增不改，落在
   `runs/`，与 `evidence/` 下的冻结参考物理隔离。
3. **延迟词汇** —— cold / fresh / warm / boot 是*不同的数字*，永不混用。
4. **VRAM 是采样值而非精确值** —— 0.5–1 秒轮询的峰值是下界；我们同时记录 torch 层
   与设备层（rocm-smi）两个视角。

SHA256 校验的真实行为（精确读——文档声称的正是这个）：每个文件的首次检查完整散列；
成功后打上戳记（大小 + mtime + 摘要），此后文件未变则返回 `stamp-cache-hit` 不再
散列；文件*任何*变化都会使戳记失效并强制完整重散。大小不符则不散列直接失败。
只查大小会漏掉什么？——长度相同但字节不同的文件，例如断点续传拼出"正确大小"的
损坏下载（父项目的 E1 turbo 基准正是死在这上面：`lab/download_weights.py` 里的
safetensors 头部门就是为此而设）。

案例锚点：[docs/case-study/P0_PREFLIGHT.md](../../docs/case-study/P0_PREFLIGHT.md)
——那次审计既发现了健康的 W7900 技术栈，也发现了真实的阻塞项（出口代理、一个
环境相关的测试失败），全部记录，无一隐瞒。全仓库通用的判定词汇：
**PASS / FAIL / BLOCKED / NOT_RUN**。

**思考（答案在 `<details>` 中）。**

1. 为什么 doctor 要散列整个文件，而不是相信文件大小？
   <details><summary>答案</summary>大小碰撞太容易制造（断线下载、半截写入）。
   SHA256 固定让仓库内表格成为信任锚：任何字节级损坏无论长度如何都会被抓到。</details>
2. 你的实例显示 `gfx1100`，同事的显示 `gfx1201`，doctor 打印 WARN 而非 FAIL
   （cloud 模式）。你们都能上这门课吗？能把数字合并进同一个中位数吗？
   <details><summary>答案</summary>学习可以——方法论与硬件无关。合并不行：不同架构
   是不同的测量总体；每套数字只对自己的主机有效（这正是 `report --scope student`
   绝不与冻结参考混合的原因）。</details>
3. 为什么 doctor 自己的报告写进 `runs/` 而不是 `evidence/`？
   <details><summary>答案</summary>`evidence/` 是官方战役的不可变记录；你的机器产出
   的是*你的*证据。往里追加会污染已发表的中位数——正是本仓库用代码防范的那类错误。</details>

**学生练习（按难度分层）。**
- 入门：运行 `doctor --cloud`，把输出存为 `runs/doctor.json`，用一句话解释每个 PASS
  防御的事故。验收：7 项检查各有一句正确解释。
- 进阶：人为制造一次失败（如临时改一个权重文件名字使大小检查失败），观察 fail-closed
  行为，然后恢复。提交：失败输出 + 修复后的 PASS 输出。
- 挑战：阅读 `lab/download_weights.py` 中的 safetensors 头部门，解释它与
  "同长度不同字节"攻击的关系。提交：150 字说明 + 指出具体代码行。

**研究延伸（可检验的假设）。** 假设：在低出站带宽的云实例上，把 25.5 GB 权重改为
分阶段按需下载（先 turbo + VAE，后补 40 步权重）可以把"到首张图"时间缩短 40% 以上，
而不破坏 SHA256 完整性门。设计一个可测的实验（度量"到首图"时间，n≥3），预测其
下界，并说明你会如何证明它没有破坏 fail-closed 语义。

**验收标准。**
- [ ] `start.sh` 跑完并打印了首图路径（或你记录了它报告的可行动阻塞项）
- [ ] `doctor --cloud` 判定 PASS（六模型齐后再跑严格版 PASS）
- [ ] 你能说出 7 项严格检查的名称与各自防御的对象（打开 `runs/doctor.json`）
- [ ] 你用自己的话解释了戳记缓存行为与它仍能抓到的攻击
