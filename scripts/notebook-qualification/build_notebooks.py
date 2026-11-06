#!/usr/bin/env python3
"""Build the 18 bilingual GPU-first course notebooks from one spec.

Code cells are byte-identical between the EN and zh-CN variants (checked by
scripts/course-quality/check_notebooks.py); only markdown differs. Run from
the repository root:

    python3 scripts/notebook-qualification/build_notebooks.py [--check]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NB = ROOT / "notebooks"

# ---------------------------------------------------------------------------
# shared code snippets (identical bytes in EN and zh-CN notebooks)
# ---------------------------------------------------------------------------

def header(nbid: str) -> str:
    return f'''from pathlib import Path
import json
import statistics
import sys
ROOT = next((p for p in (Path.cwd().resolve(), *Path.cwd().resolve().parents)
             if (p / "lab/reference_runs.json").is_file()), None)
if ROOT is None:
    raise FileNotFoundError("Open Jupyter from the course repository root (see notebooks/README.md).")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
COURSE_MODE = "REFERENCE"  # <- edit once: REFERENCE | GPU_CLASSROOM | GPU_RESEARCH
from lab import notebook_runtime as rt
from lab import notebook_support as ns
session = rt.begin(COURSE_MODE, notebook="{nbid}")
'''

SUPPORT_IMPORTS = '''from lab import notebook_support as ns
import statistics
'''

MEASURED_TAG = '''def _stamp(kind: str, value: str) -> None:
    """Print a visible origin tag so no number is mistaken for another kind."""
    print(f"[{kind}] {value}")

MEASURED_TODAY = "measured on THIS GPU in THIS session (physical)"
HISTORICAL   = "frozen reference evidence from the committed campaign"
ESTIMATE     = "analytical estimate or simulation - NOT a measurement"
'''

GPU_OR_SKIP = '''def gpu_cell(fn):
    """Run a physical exercise only in GPU modes; print why otherwise."""
    if session.mode != "REFERENCE":
        return fn()
    print("skipped physical exercise: COURSE_MODE=REFERENCE (no GPU work launched)")
    return None
'''

# ---------------------------------------------------------------------------
# markdown blocks shared by all modules
# ---------------------------------------------------------------------------

def modes_md(lang: str) -> str:
    if lang == "en":
        return """## The three course modes

| Mode | What runs | Where | Use it for |
|---|---|---|---|
| `REFERENCE` | frozen committed evidence only (CPU, offline) | `evidence/` | learning the method, CPU CI, debugging |
| `GPU_CLASSROOM` | **your own bounded W7900 measurements** | `runs/` | the default lab experience on Radeon Cloud |
| `GPU_RESEARCH` | full-protocol physical A/B campaigns | `runs/` | capstone / research-grade reproduction |

Set `COURSE_MODE = "GPU_CLASSROOM"` in the first code cell, then
**Kernel → Restart & Run All**. A classroom exercise is a *bounded physical
experiment*: it proves the pipeline works on your GPU and gives a first
comparison — it is **not** a formal performance claim (that needs the
research protocol: repeated prompts × seeds, matched populations).
Historical numbers in `evidence/` are references to investigate, never a
substitute for your own measurement."""
    return """## 三种课程模式

| 模式 | 运行内容 | 数据位置 | 适用场景 |
|---|---|---|---|
| `REFERENCE` | 仅分析已冻结的历史证据（CPU、离线） | `evidence/` | 学习方法论、CPU CI、调试 |
| `GPU_CLASSROOM` | **你自己在这台 W7900 上的有界实测** | `runs/` | Radeon Cloud 课堂默认模式 |
| `GPU_RESEARCH` | 完整协议的物理 A/B 实验 | `runs/` | 毕业课题 / 研究级复现 |

在第一个代码单元中设置 `COURSE_MODE = "GPU_CLASSROOM"`，然后
**Kernel → Restart & Run All**。课堂练习是一次*有界物理实验*：它证明流水线
在你的 GPU 上真实可用，并给出初步对比——但它**不是**正式性能结论（正式结论
需要研究协议：多提示词 × 多种子重复、匹配的样本总体）。
`evidence/` 中的历史数字是用来研究对照的参考值，永远不能替代你自己的测量。"""


def acceptance_md(lang: str) -> str:
    if lang == "en":
        return """## Acceptance criteria

- [ ] You can state which numbers are **measured on this GPU today**, which are
      **historical reference values**, and which are **estimates/simulations**.
- [ ] Every physical run you produced has a visible `run_id` under `runs/`.
- [ ] You did not present a smoke/classroom population as a formal result.
- [ ] You can explain one limitation of each measurement you reported."""
    return """## 验收标准

- [ ] 你能指出哪些数字是**今天在这块 GPU 上实测的**、哪些是
      **历史参考值**、哪些是**估计/模拟**。
- [ ] 你产生的每次物理运行都在 `runs/` 下有可见的 `run_id`。
- [ ] 你没有把 smoke/课堂样本冒充为正式协议结果。
- [ ] 你能解释你报告的每项测量的一个局限性。"""


def repro_md(lang: str, exp: str) -> str:
    if lang == "en":
        return f"""## Reproduce & contribute

```bash
python -m lab doctor                                  # admission evidence -> runs/doctor.json
python -m lab run --experiment {exp} --date <your-tag>  # full research protocol
python -m lab report --scope student --kind all       # aggregate YOUR runs only
```

Commit your `runs/<run-id>/*.json` records (never `evidence/`) and open a PR
describing: hypothesis → protocol → numbers → limitations. Negative results
are welcome — a measured regression is a scientific outcome, not a failure."""
    return f"""## 复现与贡献

```bash
python -m lab doctor                                  # 准入证据 -> runs/doctor.json
python -m lab run --experiment {exp} --date <your-tag>  # 完整研究协议
python -m lab report --scope student --kind all       # 只汇总你自己的 runs
```

提交你的 `runs/<run-id>/*.json` 记录（绝不写 `evidence/`），并在 PR 中说明：
假设 → 协议 → 数字 → 局限。欢迎负结果——测得的退化是科学结果，不是失败。"""


HUMAN_QUALITY_EN = (
    "> **Quality decisions stay human.** No cell auto-approves an image-quality "
    "verdict. Historical owner verdicts apply **only** to their originally "
    "evaluated deployment scope; a new physical run needs its own review "
    "(`PENDING_HUMAN_REVIEW` until then).")
HUMAN_QUALITY_ZH = (
    "> **质量判定永远由人完成。** 任何单元都不会自动批准图像质量结论。"
    "历史所有者判定**只**适用于其原始评估的部署范围；新的物理运行需要自己的"
    "评审（在此之前保持 `PENDING_HUMAN_REVIEW`）。")


# ---------------------------------------------------------------------------
# cell builders
# ---------------------------------------------------------------------------

def code(src: str) -> tuple[str, str]:
    return ("code", src.strip("\n"))


def md(en: str, zh: str) -> tuple[str, str, str]:
    return ("md", en.strip("\n"), zh.strip("\n"))


def build_cells(spec: list, lang: str) -> list[dict]:
    cells = []
    for idx, item in enumerate(spec):
        if item[0] == "md":
            cells.append({"cell_type": "markdown", "id": f"md-{idx:02d}",
                          "metadata": {},
                          "source": item[1] if lang == "en" else item[2]})
        else:
            cells.append({"cell_type": "code", "id": f"code-{idx:02d}",
                          "execution_count": None,
                          "metadata": {}, "outputs": [],
                          "source": item[1]})
    for c in cells:
        src = c["source"]
        c["source"] = src.splitlines(keepends=True)
    return cells


def write_notebook(path: Path, lang: str, title: str, cells_spec: list) -> None:
    nb = {
        "cells": build_cells(cells_spec, lang),
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
            "course": {"language": "en" if lang == "en" else "zh-CN",
                       "gpu_default": False},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n",
                    encoding="utf-8")


# ---------------------------------------------------------------------------
# module specs — each function returns the cells for one module
# ---------------------------------------------------------------------------

def m00() -> list:
    return [
        md("# M0 · GPU readiness and experimental discipline\n"
           "\n"
           "**Learning objective:** before any optimization claim, prove that "
           "(1) the GPU you are measuring on is the GPU you think it is, "
           "(2) your timers respect GPU asynchrony, and (3) every number you "
           "report has a declared origin.\n\n"
           "Reference lesson: [M00 lesson](../course/M00_setup_and_discipline/LESSON.md) · "
           "Switch language: [中文](./00_experimental_discipline.zh-CN.ipynb)\n\n" + modes_md("en"),
           "# M0 · GPU 就绪与实验纪律\n"
           "\n"
           "**学习目标：** 在提出任何优化结论之前，先证明：(1) 你测量的 GPU 确实是"
           "你以为的那块 GPU；(2) 你的计时方式尊重 GPU 异步性；(3) 你报告的每个数字"
           "都有明确的来源声明。\n\n"
           "参考讲义：[M00 讲义](../course/M00_setup_and_discipline/LESSON.zh-CN.md) · "
           "切换语言：[English](./00_experimental_discipline.ipynb)\n\n" + modes_md("zh")),
        code(header("00") + "\n" + MEASURED_TAG + "\n" + GPU_OR_SKIP),
        md("## 1 · Hardware admission — is this really a W7900?\n"
           "The admission banner above already answered in GPU modes. The cell "
           "below records the physical device facts and the exact model files "
           "this course pins (size + pinned sha256 digests — the trust anchor "
           "lives in `lab/weights.py`, never in a live endpoint).",
           "## 1 · 硬件准入 —— 这真的是一块 W7900 吗？\n"
           "GPU 模式下，上方的准入横幅已经给出答案。下面的单元记录物理设备事实，"
           "以及本课程锁定的模型文件（体积 + 锁定的 sha256 摘要——信任锚点在 "
           "`lab/weights.py`，绝不依赖在线端点）。"),
        code('''from lab import weights as weights_mod

if session.mode != "REFERENCE":
    d = session.admission.detail
    _stamp(MEASURED_TODAY,
           f"device={d['device_name']} gfx={d['gfx_arch']} "
           f"vram={d['vram_used_mib']}/{d['vram_total_mib']} MiB hip={d['hip']}")
else:
    print("REFERENCE mode: physical admission is skipped by design; "
          "the frozen campaign device facts are in evidence/results/env/")
for state in weights_mod.weights_state():
    r = weights_mod.verify_weight(state["rel"])
    print(f"  {state['state']:8s} {state['rel'].split('/')[-1]}  ({r['reason']})")'''),
        md("## 2 · GPU asynchrony — why unsynchronized timing is wrong\n"
           "GPUs return control before the work finishes. Reading a timer "
           "immediately after launching a kernel measures *launch*, not "
           "*execution*. The cell below demonstrates it physically in GPU "
           "modes (a large bf16 matmul), and with a CPU thread analogy in "
           "REFERENCE mode so the lesson is never skipped.",
           "## 2 · GPU 异步性 —— 为什么不同步的计时是错的\n"
           "GPU 会在工作完成之前就交还控制权。启动内核后立刻读计时器，测到的是"
           "*启动*而不是*执行*。下面的单元在 GPU 模式下用一次大矩阵乘法物理演示这"
           "一点；REFERENCE 模式下用 CPU 线程类比，保证这一课永远不会被跳过。"),
        code('''import time

def demo_async_gpu():
    import torch
    torch.manual_seed(0)
    a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(20):
        b = a @ a
    t_no_sync = time.perf_counter() - t0      # WRONG: measures launch queue
    torch.cuda.synchronize()                  # correct: wait for completion
    t_synced = time.perf_counter() - t0       # RIGHT: includes execution
    print(f"WITHOUT sync: {t_no_sync:.4f}s   WITH sync: {t_synced:.4f}s")
    assert t_synced >= t_no_sync
    _stamp(MEASURED_TODAY, f"sync gap = {t_synced - t_no_sync:.3f}s on this GPU")

def demo_async_cpu_threads():
    import threading
    done = threading.Event()
    result = {}
    def work():
        x = sum(i * i for i in range(3_000_000))
        result["x"] = x
        done.set()
    t0 = time.perf_counter()
    threading.Thread(target=work).start()
    t_no_join = time.perf_counter() - t0      # WRONG: worker still running
    done.wait()                               # correct: join first
    t_joined = time.perf_counter() - t0
    print(f"WITHOUT join: {t_no_join:.5f}s   WITH join: {t_joined:.5f}s")
    _stamp(ESTIMATE, "CPU-thread analogy of GPU asynchrony (not a GPU measurement)")

gpu_cell(demo_async_gpu) or demo_async_cpu_threads()'''),
        md("## 3 · Smoke vs cold vs warm — three different populations\n"
           "A **smoke** run proves the pipeline executes (one shot, no warmup, "
           "no statistics). A **cold** run includes first-load costs (weights "
           "from disk, allocator growth). **Warm** runs are the comparable "
           "population for latency claims. Pooling them is a classic "
           "measurement bug. Check the frozen campaign for real examples of "
           "each kind:",
           "## 3 · Smoke / cold / warm —— 三种不同的样本总体\n"
           "**smoke** 运行证明流水线可以执行（单次、无预热、无统计）。**cold** 运行"
           "包含首次加载成本（权重读盘、分配器增长）。**warm** 运行才是延迟结论可比"
           "的总体。把它们混在一起是经典的测量错误。在冻结实验数据中找每种真实样本："),
        code(SUPPORT_IMPORTS + '''import json

kinds = {}
for d in ns.read_json("lab/reference_runs.json")["reference_run_dirs"]["B0"]:
    for f in sorted((ROOT / "evidence/results" / d).glob("*.json")):
        rec = json.load(open(f))
        kinds.setdefault(rec["run_kind"], 0)
        kinds[rec["run_kind"]] += len(rec.get("runs", []))
print("B0 reference populations:", kinds)
_stamp(HISTORICAL, "population labels above come from the frozen campaign")'''),
        md("## 4 · Your turn\n"
           "1. Predict the warm/cold gap for B0 on your instance, then (in "
           "GPU_CLASSROOM) compare with M1.\n"
           "2. Find in the frozen records one warm sample slower than the "
           "cold sample of another session — why does 'cold is slowest' fail "
           "as an invariant?\n"
           "3. Explain why a smoke PASS can never establish a speedup.",
           "## 4 · 动手练习\n"
           "1. 预测你这台实例上 B0 的 warm/cold 差距，然后在 GPU_CLASSROOM 模式下"
           "与 M1 对比。\n"
           "2. 在冻结记录里找出某个 warm 样本比另一会话 cold 样本更慢的例子——"
           "为什么“cold 一定最慢”不是不变量？\n"
           "3. 解释为什么 smoke PASS 永远不能证明加速比。"),
        md(acceptance_md("en"), acceptance_md("zh")),
    ]


def m01() -> list:
    return [
        md("# M1 · Your own physical baseline on the W7900\n"
           "\n"
           "**Learning objective:** produce a real Qwen-Image-2.1 B0 "
           "(BF16, 40 steps, 1024²) result on **your** GPU with a declared "
           "protocol, and treat the published 55.206 s median as a *hypothesis "
           "to investigate*, never as your answer.\n\n"
           "**Predict first:** write down your expected warm latency before "
           "running (a W7900 reference is 55.2 s; your cloud instance may "
           "differ — that difference is course material, not an error).\n\n"
           "Reference lesson: [M01 lesson](../course/M01_baseline/LESSON.md) · "
           "[中文](./01_baseline_and_uncertainty.zh-CN.ipynb)",
           "# M1 · 在 W7900 上建立你自己的物理基线\n"
           "\n"
           "**学习目标：** 在**你的** GPU 上以声明的协议产生一个真实的 "
           "Qwen-Image-2.1 B0（BF16、40 步、1024²）结果，并把已发表的 55.206 秒"
           "中位数当作*待研究的假设*，而不是你的答案。\n\n"
           "**先预测：** 运行前先写下你预期的 warm 延迟（W7900 参考值为 55.2 秒；"
           "你的云实例可能不同——这种差异本身就是课程素材，不是错误）。\n\n"
           "参考讲义：[M01 讲义](../course/M01_baseline/LESSON.zh-CN.md) · "
           "[English](./01_baseline_and_uncertainty.ipynb)"),
        code(header("01") + "\n" + MEASURED_TAG + "\n" + GPU_OR_SKIP),
        md("## 1 · The frozen reference (always available, never yours)\n"
           "Recompute the published median from the two pinned B0 sessions — "
           "this is the number your own run will be compared against.",
           "## 1 · 冻结参考值（永远可用，但永远不是你的）\n"
           "从两个锁定的 B0 会话重算已发表的中位数——你自己的运行将与它对比。"),
        code(SUPPORT_IMPORTS + '''
raw = ns.reference_warm("B0")
claimed = ns.summary_rows()["B0"]
observed = statistics.median(raw)
assert len(raw) == claimed["n_measure"] == 18
assert abs(observed - claimed["warm_total_median_s"]) < 0.03
_stamp(HISTORICAL, f"B0 reference median {observed:.3f}s "
      f"(min {min(raw):.2f}, max {max(raw):.2f}, n={len(raw)}, 2 sessions x 9)")'''),
        md("## 2 · Run B0 on YOUR W7900\n"
           "`GPU_CLASSROOM` runs a bounded exercise (1 prompt × cold+warmup+2 "
           "measured ≈ 5 min). `GPU_RESEARCH` runs the full frozen protocol "
           "(3 prompts × 3 seeds × warmup+3 measured + fresh-prompt arm). "
           "Both write append-only records under `runs/<run-id>/`.",
           "## 2 · 在**你的** W7900 上运行 B0\n"
           "`GPU_CLASSROOM` 运行有界练习（1 个提示词 × cold+warmup+2 次测量，约 5 分钟）。"
           "`GPU_RESEARCH` 运行完整冻结协议（3 提示词 × 3 种子 × warmup+3 次测量 + "
           "fresh-prompt 臂）。两者都以只追加方式把记录写入 `runs/<run-id>/`。"),
        code('''def run_own_b0():
    profile = "classroom" if session.mode == "GPU_CLASSROOM" else "research"
    bundle = session.run("B0", profile=profile)
    return bundle

b0 = gpu_cell(run_own_b0)
if b0 is None:
    shared = session.latest_campaign_run("B0")
    if shared:
        b0 = session.consume(shared, why="qualification-campaign B0 record")'''),
        md("## 3 · Your numbers, your statistics\n"
           "Median, spread, per-step time, and peak VRAM — computed from the "
           "records of the run above (or, in REFERENCE mode, still from the "
           "frozen campaign, clearly labelled).",
           "## 3 · 你的数字，你的统计\n"
           "中位数、离散度、每步时间与 VRAM 峰值——全部由上面那次运行的记录计算"
           "（REFERENCE 模式下仍来自冻结实验数据，并明确标注）。"),
        code('''if b0 is None:
    vals, origin, steps_hint = raw, HISTORICAL, 40
    cold = None
else:
    st = b0.stats()
    vals, origin, steps_hint = b0.warm_totals(), MEASURED_TODAY, st["steps"]
    cold = b0.cold_total()

med = statistics.median(vals)
print(f"population n={len(vals)}  median={med:.3f}s  mean={statistics.mean(vals):.3f}s  "
      f"min={min(vals):.3f}s  max={max(vals):.3f}s")
if cold is not None:
    print(f"cold (first load) = {cold:.3f}s  -> warm/cold gap {cold-med:.3f}s")
_stamp(origin, f"n={len(vals)} median={med:.3f}s steps={steps_hint}")
if b0 is not None:
    print("per-step (steady median):", b0.step_stats())
    print("device VRAM peak (MB):", st["vram_device_peak_mb"], "| torch peak:",
          st["vram_peak_mb"])'''),
        md("## 4 · Compare against the published number — explain, don't copy\n"
           "Reference deviation = your median − 55.206 s. A few percent is "
           "normal instance-to-instance variation; a large gap means a "
           "different population (cold mixed in), contention (another GPU "
           "tenant), or thermal/clock state — name the most likely cause.",
           "## 4 · 与已发表数字对比——要解释，不要照抄\n"
           "参考偏差 = 你的中位数 − 55.206 秒。几个百分比的差异属于正常的实例间"
           "波动；大差距则意味着样本总体不同（混入 cold）、资源争用（GPU 有其他"
           "租户）或温度/频率状态——请指出最可能的原因。"),
        code('''dev = med - 55.206
print(f"reference deviation: {dev:+.3f}s ({dev/55.206*100:+.1f}% vs frozen campaign)")
print("limitation: classroom n is small; treat the deviation as a signal to "
      "investigate, not a verdict")
try:
    fig = rt.plot_ab({"label": "B0: reference vs yours",
                      "classification": "REFERENCE vs " + ("YOUR PHYSICAL RUN"
                      if b0 is not None else "REFERENCE"),
                      "baseline": {"experiment": "B0 ref", "n_warm": 18,
                                   "median_s": 55.206},
                      "candidate": {"experiment": "B0 yours", "n_warm": len(vals),
                                    "median_s": round(med, 3)}})
    if fig: fig.show()
except Exception as e:
    print("plot skipped:", e)'''),
        md("## 5 · Your turn\n"
           "1. Re-run the bounded exercise and compare the two medians — is "
           "the run-to-run spread smaller or larger than the reference "
           "deviation?\n"
           "2. Which single change would most reduce your measurement "
           "uncertainty, and why?",
           "## 5 · 动手练习\n"
           "1. 重跑一次有界练习并比较两个中位数——运行间波动比参考偏差更大还是"
           "更小？\n"
           "2. 哪一个改动最能降低你的测量不确定性？为什么？"),
        md(acceptance_md("en"), acceptance_md("zh")),
        md(repro_md("en", "B0"), repro_md("zh", "B0")),
    ]


def m02() -> list:
    return [
        md("# M2 · Physical profiling — where does GPU time go?\n"
           "\n"
           "**Learning objective:** collect a real kernel trace of the "
           "Qwen-Image workload on your W7900 with `rocprofv3`, attribute GPU "
           "busy time to GEMM / attention / norm / elementwise classes, and "
           "separate *GPU busy time* from *end-to-end latency*. The frozen "
           "campaign's 61.3% GEMM share is a hypothesis for your host.\n\n"
           "Reference lesson: [M02 lesson](../course/M02_profiling/LESSON.md) · "
           "[中文](./02_profile_and_bottlenecks.zh-CN.ipynb)",
           "# M2 · 物理性能剖析 —— GPU 时间都花在哪里？\n"
           "\n"
           "**学习目标：** 用 `rocprofv3` 在你的 W7900 上采集 Qwen-Image 工作负载的"
           "真实内核轨迹，把 GPU 忙碌时间归因到 GEMM / 注意力 / 归一化 / 逐元素"
           "各类，并区分*GPU 忙碌时间*与*端到端延迟*。冻结实验数据中的 61.3% GEMM "
           "占比只是针对你这台主机的假设。\n\n"
           "参考讲义：[M02 讲义](../course/M02_profiling/LESSON.zh-CN.md) · "
           "[English](./02_profile_and_bottlenecks.ipynb)"),
        code(header("02") + "\n" + MEASURED_TAG + "\n" + GPU_OR_SKIP),
        md("## 1 · Pipeline decomposition from run records\n"
           "First the cheap layer: split warm end-to-end latency into "
           "KSampler (DiT denoise), VAE decode, text-encode and derived "
           "residual, using node timings from your (or the reference) B0 "
           "records.",
           "## 1 · 由运行记录做流水线分解\n"
           "先做低开销的一层：用 B0 运行记录中的节点计时，把 warm 端到端延迟拆成 "
           "KSampler（DiT 去噪）、VAE 解码、文本编码与推导残差。"),
        code('''def own_pipeline_layer():
    out = session.run_profile(layers="pipeline")
    return out

pipe_dir = gpu_cell(own_pipeline_layer)
if pipe_dir is not None:
    pb = json.loads((pipe_dir / "pipeline_breakdown.json").read_text())
    _stamp(MEASURED_TODAY, f"pipeline layer from YOUR records: {pipe_dir.name}")
else:
    pb = ns.read_json("evidence/results/p2-profile-profile/pipeline_breakdown.json")
    _stamp(HISTORICAL, "frozen campaign pipeline breakdown")
for k in ("n_warm", "T_total_warm_s", "T_dit_denoise_s", "T_vae_decode_s",
          "T_text_encoder_cached_s", "T_other_warm_s", "step_median_s"):
    print(f"  {k:28s} {pb.get(k)}")
print("  method note:", pb.get("method", {}).get(
    "T_other_warm_s", pb.get("note", "derived by subtraction")))'''),
        md("## 2 · The kernel layer (rocprofv3)\n"
           "The expensive layer traces the **real ComfyUI server process** "
           "under a bounded `rocprofv3` window (DiT phase and text-encoder "
           "phase). Interception adds overhead, so **traced wall times are "
           "never performance claims** — composition only. In class you "
           "usually *consume* the qualification-campaign trace; "
           "`RUN_KERNEL_TRACE = True` (GPU modes) launches your own bounded "
           "window (~10 min).",
           "## 2 · 内核层（rocprofv3）\n"
           "昂贵的一层会在有界 `rocprofv3` 窗口内追踪**真实的 ComfyUI 服务器进程**"
           "（DiT 阶段与文本编码阶段）。拦截会引入额外开销，因此**追踪时的墙钟时间"
           "永远不是性能结论**——只用于构成分析。课堂上通常*消费*资格认证实验的"
           "轨迹；设置 `RUN_KERNEL_TRACE = True`（GPU 模式）可启动你自己的有界窗口"
           "（约 10 分钟）。"),
        code('''RUN_KERNEL_TRACE = False  # opt-in: ~10 min of physical GPU tracing

FROZEN_TRACE = ROOT / "evidence/results/p2-profile-profile/rocprof-dit/analysis.json"

def load_kernel_analysis():
    cands = sorted((ROOT / "runs").glob("*-profile/rocprof-dit/analysis.json"))
    if cands:
        return json.loads(cands[-1].read_text()), MEASURED_TODAY
    if FROZEN_TRACE.is_file():
        return json.loads(FROZEN_TRACE.read_text()), HISTORICAL
    return None, None

loaded = gpu_cell(load_kernel_analysis)
ka = loaded[0] if loaded else None
ka_origin = loaded[1] if loaded else None
if ka is None and RUN_KERNEL_TRACE and session.mode != "REFERENCE":
    session.run_profile(layers="pipeline,kernel")
    loaded = load_kernel_analysis()
    ka, ka_origin = (loaded[0], loaded[1]) if loaded else (None, None)
if ka is not None:
    _stamp(ka_origin, "kernel trace analysis (traced run; composition only)")
    print("mode:", ka.get("mode"), "| kernels:", ka.get("n_kernels"),
          "| gpu_busy_s:", ka.get("gpu_busy_s"), f"({ka.get('gpu_busy_pct')}%)")
    by = ka.get("by_class", {})
    total = sum(v["total_s"] for v in by.values()) or 1.0
    for name, v in sorted(by.items(), key=lambda kv: -kv[1]["total_s"]):
        print(f"  {name:12s} {v['total_s']:8.2f}s  {v['total_s']/total*100:5.1f}%  "
              f"n={v.get('n', v.get('count', '?'))}")
else:
    print("No kernel trace available on this host")'''),
        md("## 3 · From composition to an Amdahl ceiling\n"
           "If GEMM is fraction `g` of GPU busy time, even an infinitely fast "
           "GEMM caps the kernel-layer speedup at 1/(1−g). Compute the "
           "ceiling from **your** trace — and remember the ceiling applies "
           "to GPU busy time, not end-to-end latency (queueing, TE, VAE, "
           "host overhead sit outside it).",
           "## 3 · 从构成到 Amdahl 上限\n"
           "若 GEMM 占 GPU 忙碌时间的比例为 `g`，即使 GEMM 无限快，内核层加速上限"
           "也只有 1/(1−g)。用**你自己的**轨迹计算这个上限——并记住它只作用于 GPU "
           "忙碌时间，而不是端到端延迟（排队、文本编码、VAE、主机开销都不在其中）。"),
        code('''if ka is not None:
    by = ka.get("by_class", {})
    gemm = by.get("gemm", {}).get("total_s", 0.0)
    total_busy = sum(v["total_s"] for v in by.values()) or 1.0
    g = gemm / total_busy
    ceiling = 1.0 / (1.0 - g) if g < 1 else float("inf")
    _stamp(ESTIMATE,
           f"Amdahl kernel-layer ceiling {ceiling:.2f}x from YOUR measured "
           f"g={g:.3f} (derived, not measured)")
    print(f"gemm share g = {g*100:.1f}%  ->  ceiling {ceiling:.2f}x")
else:
    g, ceiling = 0.613, 1.0 / (1 - 0.613)
    _stamp(HISTORICAL, f"frozen campaign g=61.3% -> ceiling {ceiling:.2f}x")'''),
        md("## 4 · Your turn\n"
           "1. Why can the profiled run be slower end-to-end while the "
           "kernel composition stays valid?\n"
           "2. Your end-to-end warm median (M1) is much larger than total GPU "
           "busy time — list three concrete contributors to the gap.\n"
           "3. Which class would you attack next, and what is its own "
           "Amdahl ceiling if perfectly eliminated?",
           "## 4 · 动手练习\n"
           "1. 为什么被追踪的运行端到端更慢，但内核构成分析依然有效？\n"
           "2. M1 的端到端 warm 中位数远大于 GPU 忙碌总时间——列出造成差距的三个"
           "具体因素。\n"
           "3. 你下一个会优化哪一类？若把它完全消除，它对应的 Amdahl 上限是多少？"),
        md(acceptance_md("en"), acceptance_md("zh")),
        md(repro_md("en", "B0"), repro_md("zh", "B0") + "\n\n"
           "```bash\npython -m lab profile --layers pipeline,kernel  # physical rocprofv3 trace\n```"),
    ]


def m03() -> list:
    return [
        md("# M3 · Less computation — Turbo 8 steps vs baseline 40\n"
           "\n"
           "**Learning objective:** measure a *less computation* candidate "
           "(Turbo distilled DiT, 8-step schedule) against the B0 baseline on "
           "your W7900, compare observed speedup with an Amdahl prediction, "
           "and reason honestly about the **confound**: B0→L1 changes the "
           "model variant AND the sampling schedule, not just a step-count "
           "knob.\n\n"
           "**Predict first:** naive step ratio 40/8 = 5×. Will you measure "
           "5×? Write your prediction and your reason.\n\n"
           "Reference lesson: [M03 lesson](../course/M03_less_computation/LESSON.md) · "
           "[中文](./03_less_computation.zh-CN.ipynb)",
           "# M3 · 少算 —— Turbo 8 步 vs 基线 40 步\n"
           "\n"
           "**学习目标：** 在你的 W7900 上测量*少算*候选（Turbo 蒸馏 DiT、8 步调度）"
           "与 B0 基线的差距，把实测加速比与 Amdahl 预测对比，并诚实分析**混淆变量**："
           "B0→L1 同时改变了模型变体和采样调度，并不是只改步数这一个旋钮。\n\n"
           "**先预测：** 朴素步数比 40/8 = 5×。你会测到 5× 吗？写下你的预测与理由。\n\n"
           "参考讲义：[M03 讲义](../course/M03_less_computation/LESSON.zh-CN.md) · "
           "[English](./03_less_computation.ipynb)"),
        code(header("03") + "\n" + MEASURED_TAG + "\n" + GPU_OR_SKIP),
        md("## 1 · The confound, stated up front\n"
           "B0 = full BF16 DiT, 40 steps. L1 = Turbo BF16 DiT, 8 steps. "
           "A *clean* step-count experiment would fix the model and vary only "
           "steps (e.g. Turbo at 8 vs 40 steps, same sampler/scheduler). "
           "This lab measures the **deployable configuration pair**; keep the "
           "distinction explicit in anything you report.",
           "## 1 · 先说清混淆变量\n"
           "B0 = 完整 BF16 DiT、40 步。L1 = Turbo BF16 DiT、8 步。*干净的*步数实验"
           "应当固定模型、只改变步数（例如同一 Turbo 模型在 8 步与 40 步、相同采样"
           "器/调度器下对比）。本实验测的是**可部署配置对**；在任何报告里都要显式"
           "区分这两件事。"),
        code('''CONF0UND_NOTE = ("B0->L1 changes model variant + schedule + steps; "
    "it is NOT a single-variable step-count experiment")
print(CONF0UND_NOTE)
ref = ns.summary_rows()
for k in ("B0", "L1"):
    r = ref[k]
    _stamp(HISTORICAL, f"{k}: median {r['warm_total_median_s']}s n={r['n_measure']}")'''),
        md("## 2 · Run both arms on YOUR W7900\n"
           "Baseline B0 (bounded or full, matching your mode) then candidate "
           "L1. If you already produced a B0 in M1 in this campaign it is "
           "reused with its run_id shown — records are append-only, waste is "
           "not repeated.",
           "## 2 · 在**你的** W7900 上运行两个臂\n"
           "先跑基线 B0（与你的模式匹配，有界或完整），再跑候选 L1。如果你在本轮"
           "M1 已产生 B0，将直接复用并显示其 run_id——记录只追加，不重复浪费算力。"),
        code('''def run_less_pair():
    prof = "classroom" if session.mode == "GPU_CLASSROOM" else "research"
    b0 = session.run("B0", profile=prof)
    l1 = session.run("L1", profile=prof)
    return b0, l1

pair = gpu_cell(run_less_pair)
if pair is None:
    b0_id = session.latest_campaign_run("B0")
    l1_id = session.latest_campaign_run("L1")
    if b0_id and l1_id:
        pair = (session.consume(b0_id, why="campaign B0"),
                session.consume(l1_id, why="campaign L1"))
if pair is not None:
    b0b, l1b = pair
    ab = rt.ab_compare(b0b, l1b, label="B0 (40 steps) vs L1 (Turbo 8 steps)")
    print({k: ab[k] for k in ("classification", "speedup_median",
                              "latency_delta_s", "vram_delta_mb")})'''),
        md("## 3 · Predicted vs observed speedup\n"
           "Amdahl with the DiT share `d` from your M2 pipeline "
           "decomposition (or the reference 0.83): predicted = "
           "1/(d·(8/40) + (1−d)). Compare with the measured ratio and explain "
           "the residual.",
           "## 3 · 预测加速比 vs 实测加速比\n"
           "用 M2 流水线分解中的 DiT 占比 `d`（或参考值 0.83）做 Amdahl 预测："
           "predicted = 1/(d·(8/40) + (1−d))。与实测比值对比并解释残差。"),
        code('''if pair is not None:
    b0b, l1b = pair
    bs, ls = b0b.stats(), l1b.stats()
    d = 0.83  # DiT share of warm latency (replace with YOUR M2 value)
    predicted = 1.0 / (d * (8 / 40) + (1 - d))
    measured = bs["median_s"] / ls["median_s"]
    b0_steps = (bs.get("steps") or 40) or 40
    l1_steps = (ls.get("steps") or 8) or 8
    per_step_b0 = bs["median_s"] / b0_steps
    per_step_l1 = ls["median_s"] / l1_steps
    print(f"predicted (Amdahl, d={d}): {predicted:.2f}x   measured: {measured:.2f}x")
    print(f"per-step: B0 {per_step_b0:.3f}s  L1 {per_step_l1:.3f}s "
          f"(ratio {per_step_b0/per_step_l1:.2f}x)")
    _stamp(MEASURED_TODAY, f"measured speedup {measured:.2f}x [{ab['classification']}]")
    _stamp(ESTIMATE, f"Amdahl prediction {predicted:.2f}x (derived from model share)")
    print("limitation: Turbo changes the model too; per-step costs differ; "
          "quality is NOT addressed by this comparison (see section 4)")'''),
        md("## 4 · Quality is a separate axis\n"
           "Less sampling is not free: Turbo changes output statistics. The "
           "course gate scores L1 vs B0 with **distance semantics** (LPIPS as "
           "a perceptual *distance*, not a fidelity score) plus human review "
           "of paired grids. Never claim Turbo is 'equivalent' to the 40-step "
           "baseline.\n\n" + HUMAN_QUALITY_EN,
           "## 4 · 质量是另一个维度\n"
           "少采样不是免费的：Turbo 改变了输出统计特性。课程质量门对 L1 vs B0 采用"
           "**距离语义**（LPIPS 是感知*距离*，不是保真度分数），并对成对网格做人工"
           "评审。绝不要声称 Turbo 与 40 步基线“等价”。\n\n" + HUMAN_QUALITY_ZH),
        code('''qs = ns.summary_rows().get("L1", {})
print("frozen campaign L1 quality verdict:",
      qs.get("quality_verdict", "see evidence/quality/L1/"))
print("paired grids + native-res poster crops: evidence/quality/L1/")
print("in GPU modes run your own gate:  session.run_quality('L1', reference='B0')")
print("your verdict on YOUR images stays PENDING_HUMAN_REVIEW until you review the grid")'''),
        md("## 5 · Your turn — design the clean experiment\n"
           "Specify a controlled step-count experiment: fixed Turbo BF16 "
           "model + sampler, steps swept 8/16/40, seeds and prompts frozen. "
           "What remains incomparable to B0 even then? (model capacity, "
           "distillation, CFG=1 schedule...)",
           "## 5 · 动手练习 —— 设计干净实验\n"
           "设计一个受控的步数实验：固定 Turbo BF16 模型与采样器，步数取 "
           "8/16/40，冻结种子与提示词。即便如此，还有什么与 B0 不可比？"
           "（模型容量、蒸馏、CFG=1 调度……）"),
        md(acceptance_md("en"), acceptance_md("zh")),
        md(repro_md("en", "L1"), repro_md("zh", "L1")),
    ]


def m04() -> list:
    return [
        md("# M4 · Faster computation — attention backend & INT8 weights\n"
           "\n"
           "**Learning objective:** validate two *faster-computation* levers "
           "on your W7900 at the prescribed 40-step workload: F1 = ck "
           "attention kernel, F2 = INT8-quantized DiT weights. Both arms "
           "change execution speed, not the amount of sampling — so quality "
           "gates use **fidelity semantics**. A measured regression is a "
           "valid scientific outcome and stays visible.\n\n"
           "**Predict first:** F1 ≈ 45.1 s and F2 ≈ 51.1 s medians on the "
           "reference host — but attention backend effects are "
           "architecture-specific; write your own expectation.\n\n"
           "Reference lesson: [M04 lesson](../course/M04_faster_computation/LESSON.md) · "
           "[中文](./04_faster_computation.zh-CN.ipynb)",
           "# M4 · 快算 —— 注意力后端与 INT8 权重\n"
           "\n"
           "**学习目标：** 在你的 W7900 上按规定的 40 步负载验证两个*快算*手段："
           "F1 = ck 注意力内核，F2 = INT8 量化 DiT 权重。两个臂只改变执行速度、"
           "不改变采样量——因此质量门采用**保真语义**。测得退化是有效的科学结果，"
           "必须保留展示。\n\n"
           "**先预测：** 参考主机上 F1 ≈ 45.1 秒、F2 ≈ 51.1 秒中位数——但注意力后端"
           "的效果高度依赖架构；请写下你自己的预期。\n\n"
           "参考讲义：[M04 讲义](../course/M04_faster_computation/LESSON.zh-CN.md) · "
           "[English](./04_faster_computation.ipynb)"),
        code(header("04") + "\n" + MEASURED_TAG + "\n" + GPU_OR_SKIP),
        md("## 1 · Run F1 (ck attention) and F2 (INT8) on YOUR W7900\n"
           "B0 comes from M1/this campaign (shown run_id). Each arm boots a "
           "fresh server with its pinned flags — the runner refuses to "
           "benchmark when the attention backend or model silently falls "
           "back.",
           "## 1 · 在**你的** W7900 上运行 F1（ck 注意力）与 F2（INT8）\n"
           "B0 来自 M1/本轮实验（显示 run_id）。每个臂都会以锁定的服务端参数启动全新"
           "服务器——如果注意力后端或模型发生静默回退，运行器会拒绝测量。"),
        code('''def run_faster_arms():
    prof = "classroom" if session.mode == "GPU_CLASSROOM" else "research"
    b0 = session.run("B0", profile=prof)
    f1 = session.run("F1", profile=prof)
    f2 = session.run("F2", profile=prof)
    return b0, f1, f2

arms = gpu_cell(run_faster_arms)
if arms is None:
    ids = [session.latest_campaign_run(x) for x in ("B0", "F1", "F2")]
    if all(ids):
        arms = tuple(session.consume(i, why=f"campaign {x}")
                     for i, x in zip(ids, ("B0", "F1", "F2")))
if arms is not None:
    ab_f1 = rt.ab_compare(arms[0], arms[1], label="B0 vs F1 (ck attention)")
    ab_f2 = rt.ab_compare(arms[0], arms[2], label="B0 vs F2 (int8 weights)")
    for ab in (ab_f1, ab_f2):
        print({k: ab[k] for k in ("classification", "speedup_median",
                                  "latency_delta_s", "vram_delta_mb")})'''),
        md("## 2 · Verify the optimization was actually active\n"
           "A claimed attention win means nothing if the server silently ran "
           "the default backend. The records carry scraped server-log "
           "evidence — assert it before interpreting speedups.",
           "## 2 · 验证优化确实生效\n"
           "如果服务器静默使用了默认后端，所谓的注意力收益毫无意义。记录中带有从"
           "服务器日志抓取的证据——解读加速比之前先断言它。"),
        code('''def attention_evidence(bundle):
    evs = [r.get("attention_evidence") or {} for r in bundle.records]
    active = [e for e in evs if e]
    if not active:
        return "NO_EVIDENCE_IN_RECORDS"
    return active[-1]

if arms is not None:
    for name, b in (("F1", arms[1]), ("F2", arms[2])):
        st = b.stats()
        print(f"{name}: attn={st['attention_backend']} unet={st['unet']} "
              f"precision={st['precision']}")
        print(f"   server-log evidence: {attention_evidence(b)}")
    b0s = arms[0].stats()
    print("B0 attention:", b0s["attention_backend"],
          "| F1 must show the ck backend ACTIVE, F2 the int8 unet file")'''),
        md("## 3 · Read the numbers honestly\n"
           "1. Never multiply independent speedups: measured F1×F2 ≠ a "
           "measured combined effect (interactions are real; that is M6's "
           "job).\n"
           "2. INT8 often pays in **VRAM** more than in raw speed — quantify "
           "both.\n"
           "3. If your measured F1 is a regression, report it as measured.",
           "## 3 · 诚实解读数字\n"
           "1. 永远不要把独立加速比相乘：实测 F1×F2 ≠ 实测组合效果（交互是真实的；"
           "那是 M6 的工作）。\n"
           "2. INT8 的收益常在 **VRAM** 而非原始速度——两者都要量化。\n"
           "3. 如果你测得 F1 是退化，就按实测报告。"),
        code('''if arms is not None:
    b0s, f1s, f2s = (a.stats() for a in arms)
    print(f"{'arm':4s} {'median_s':>9s} {'n':>3s} {'vram_dev_mb':>11s}")
    for nm, st in (("B0", b0s), ("F1", f1s), ("F2", f2s)):
        print(f"{nm:4s} {st['median_s']:9.2f} {st['n_warm']:3d} "
              f"{st['vram_device_peak_mb'] or st['vram_peak_mb']:11.0f}")
    print("\\nINT8 VRAM saving (MB):",
          (b0s['vram_device_peak_mb'] or b0s['vram_peak_mb'])
          - (f2s['vram_device_peak_mb'] or f2s['vram_peak_mb']))
    for ab, nm in ((ab_f1, "F1"), (ab_f2, "F2")):
        _stamp(MEASURED_TODAY,
               f"{nm} speedup {ab['speedup_median']}x [{ab['classification']}]")
    print("reminder: do NOT report F1*F2 as a combined measured gain")'''),
        md("## 4 · Fidelity semantics for the quality gate\n"
           "F1/F2 keep the same 40-step schedule as B0, so paired images are "
           "checked as *numerics variants* (LPIPS fidelity threshold triages "
           "human review; it never auto-passes).\n\n" + HUMAN_QUALITY_EN,
           "## 4 · 质量门的保真语义\n"
           "F1/F2 与 B0 保持相同的 40 步调度，因此成对图像按*数值变体*检查"
           "（LPIPS 保真阈值用于分流人工评审；它绝不会自动通过）。\n\n" + HUMAN_QUALITY_ZH),
        code('''print("run your own fidelity gate in GPU modes:")
print("  session.run_quality('F1', reference='B0')")
print("  session.run_quality('F2', reference='B0')")
print("frozen campaign artifacts: evidence/quality/F1, evidence/quality/F2")
_stamp(HISTORICAL, "reference F2 fidelity gate triaged human review; INT8 "
      "numerical FAIL (in M5.1's H3_OV_INT8 encoder) stays FAIL - never promoted")'''),
        md("## 5 · Your turn\n"
           "1. Explain why ck attention might help one ROCm arch and not "
           "another.\n"
           "2. INT8 changed VRAM by ~6.6 GB on the reference host — what "
           "does that enable on a 48 GB card?\n"
           "3. Design the one-arm experiment that would isolate "
           "*dequantization overhead* in F2.",
           "## 5 · 动手练习\n"
           "1. 解释为什么 ck 注意力可能只在某些 ROCm 架构上有收益。\n"
           "2. 参考主机上 INT8 降低了约 6.6 GB VRAM——这在 48 GB 显卡上能带来什么？\n"
           "3. 设计一个能分离 F2 中*反量化开销*的单臂实验。"),
        md(acceptance_md("en"), acceptance_md("zh")),
        md(repro_md("en", "F1"), repro_md("zh", "F1") + "\n\n"
           "```bash\npython -m lab run --experiment F2 --date <your-tag>   # INT8 arm\n```"),
    ]


def m05() -> list:
    return [
        md("# M5.0 · The negative heterogeneous experiment — naive CPU offload\n"
           "\n"
           "**Learning objective:** prove *physically* why moving the 8B text "
           "encoder from GPU to CPU PyTorch can **lose** throughput: the GPU "
           "idles during multi-minute CPU encodes. This negative result is "
           "taught with real work (bounded pilots on your host), never only "
           "with a toy simulator — and the simulator, where used, is labelled "
           "`SIMULATED`.\n\n"
           "**Predict first:** the GPU text encoder encodes in ~0.55 s warm. "
           "How long will the same encode take on 16 CPU cores in PyTorch? "
           "Write your estimate (the frozen campaign says two orders of "
           "magnitude — verify on your host).\n\n"
           "Reference lesson: [M05 lesson](../course/M05_parallel_heterogeneous/LESSON.md) · "
           "[中文](./05_cpu_offload_negative.zh-CN.ipynb)",
           "# M5.0 · 负结果异构实验 —— 朴素的 CPU 卸载\n"
           "\n"
           "**学习目标：** 用*物理实验*证明为什么把 8B 文本编码器从 GPU 挪到 CPU "
           "PyTorch 可能**降低**吞吐：CPU 编码要花几分钟，GPU 全程空转。这个负结果"
           "用真实工作负载（在你主机上的有界试点）来教，绝不只用玩具模拟器——"
           "凡是用到模拟器的地方都标注 `SIMULATED`。\n\n"
           "**先预测：** GPU 文本编码器 warm 约 0.55 秒完成一次编码。同样任务在 16 个 "
           "CPU 核上的 PyTorch 需要多久？写下你的估计（冻结实验数据说要差两个数量级"
           "——在你主机上验证）。\n\n"
           "参考讲义：[M05 讲义](../course/M05_parallel_heterogeneous/LESSON.zh-CN.md) · "
           "[English](./05_cpu_offload_negative.ipynb)"),
        code(header("05") + "\n" + MEASURED_TAG + "\n" + GPU_OR_SKIP),
        md("## 1 · The frozen reference case (GPU vs PyTorch-CPU encoder)\n"
           "The committed cleanroom study measured the same frozen prompts on "
           "the same host: GPU PyTorch vs CPU PyTorch text encoder. Inspect "
           "the per-prompt warm medians and the ratio.",
           "## 1 · 冻结参考案例（GPU vs PyTorch-CPU 编码器）\n"
           "已提交的洁净室研究在同一主机上测量了同样的冻结提示词：GPU PyTorch 与 "
           "CPU PyTorch 文本编码器。查看每提示词 warm 中位数与比值。"),
        code(SUPPORT_IMPORTS + '''
enc = ns.read_json("runs/m51-20261105-cleanroom-01/performance/encoder_ab.json")
g = enc["arms"]["G_PyTorch_GPU"]["prompts"]
c0 = enc["arms"]["C0_PyTorch_CPU"]["prompts"]
for pid in ("p1-zh-scene", "p2-en-scene"):
    gm = g[pid]["warm_median_s"]
    cm = c0[pid]["warm_median_s"]
    print(f"{pid:14s} GPU {gm:7.3f}s   CPU {cm:8.2f}s   CPU/GPU = {cm/gm:7.1f}x")
_stamp(HISTORICAL, "frozen cleanroom campaign: PyTorch CPU TE is ~2 orders "
      "of magnitude slower than GPU TE on the same host")'''),
        md("## 2 · Your own bounded physical pilot (GPU modes)\n"
           "One isolated probe per device, one prompt, one warm repeat — "
           "a **bounded pilot**, explicitly not a full-protocol claim. The "
           "CPU arm can take minutes; that slowness *is* the measurement.",
           "## 2 · 你自己的有界物理试点（GPU 模式）\n"
           "每个设备一次隔离探针、一个提示词、一次 warm 重复——这是**有界试点**，"
           "明确声明不是完整协议结论。CPU 臂可能要花几分钟；这个“慢”本身就是测量"
           "结果。"),
        code('''def run_te_pilots():
    gpu = session.run_m51_baseline("cuda", repeats=1)
    cpu = session.run_m51_baseline("cpu", repeats=1)
    return gpu, cpu

pilots = gpu_cell(run_te_pilots)
own = None
if pilots is not None:
    import json as _j
    gpr = _j.loads((pilots[0] / "probe.json").read_text())
    cpr = _j.loads((pilots[1] / "probe.json").read_text())
    gp = next(iter(gpr["prompts"].values()))
    cp = next(iter(cpr["prompts"].values()))
    own = {"gpu_warm_s": gp["warm_samples_s"][0],
           "cpu_warm_s": cp["warm_samples_s"][0],
           "cpu_rss_mb": cpr.get("rss_peak_mb"),
           "gpu_rss_mb": gpr.get("rss_peak_mb")}
    _stamp(MEASURED_TODAY,
           f"pilot: GPU TE {own['gpu_warm_s']:.2f}s vs CPU TE "
           f"{own['cpu_warm_s']:.1f}s -> {own['cpu_warm_s']/own['gpu_warm_s']:.0f}x "
           f"slower (n=1, bounded pilot, single prompt)")'''),
        md("## 3 · The pipeline view — where the GPU sits idle\n"
           "H1 runs the CPU encoder inside ComfyUI (the integrated arm); H2 "
           "runs four requests sequentially (GPU TE) vs pipelined (CPU TE "
           "overlapping GPU denoise). Consume the qualification-campaign "
           "records below (GPU modes fetch the latest; REFERENCE shows the "
           "historical picture).",
           "## 3 · 流水线视角 —— GPU 在哪里空转\n"
           "H1 在 ComfyUI 内部运行 CPU 编码器（集成臂）；H2 用四种请求比较顺序执行"
           "（GPU TE）与流水线执行（CPU TE 与 GPU 去噪重叠）。下面消费资格认证实验的"
           "记录（GPU 模式取最新；REFERENCE 展示历史图景）。"),
        code('''def latest_record(pattern, fname):
    cands = sorted((ROOT / "runs").glob(f"{pattern}/{fname}"))
    if not cands:
        return None
    import json as _j
    return _j.loads(cands[-1].read_text())

h1 = gpu_cell(lambda: latest_record("*H1-cpu-te", "H1_RESULTS.json"))
if h1 is not None and "integrated" in h1:
    for tag, row in h1["integrated"].items():
        print(f"H1 integrated {tag:9s} total={row.get('total_s')}s "
              f"te={row.get('text_encode_s')}s vram_peak={row.get('vram_peak_mb')}MB")
    _stamp(MEASURED_TODAY, "H1 integrated CPU-TE arm from the current campaign")
else:
    print("no fresh H1 record on this host; the frozen study is section 1")

h2 = gpu_cell(lambda: latest_record("*H2-*", "H2_RESULTS.json"))
if h2 is not None:
    for k, v in h2.items():
        if isinstance(v, (int, float)):
            print(f"H2 {k}: {v}")
    _stamp(MEASURED_TODAY, "H2 sequential-vs-pipelined record from this campaign")
else:
    print("no fresh H2 record; pipelining recovery is taught from the frozen study")'''),
        md("## 4 · The scheduler simulator — clearly labelled\n"
           "The Gantt-style simulation below illustrates *why* overlap helps "
           "once the CPU encoder is fast enough (M5.1's OpenVINO). Its "
           "numbers are **SIMULATED**, never measured.",
           "## 4 · 调度器模拟 —— 明确标注\n"
           "下面的甘特图式模拟演示：当 CPU 编码器足够快（M5.1 的 OpenVINO）时，"
           "重叠为什么有效。其数字是**模拟值**，绝非实测。"),
        code('''_stamp(ESTIMATE, "SIMULATED pipeline schedule - illustrative only")

gpu_t = own["gpu_warm_s"] if own else 0.55     # TE on GPU (s)
dit_t = 12.0                                   # DiT+VAE per request (s, order of M1/B0)
cpu_t = own["cpu_warm_s"] if own else 120.0    # TE on CPU (s)
seq_gpu = 4 * (gpu_t + dit_t)                  # 4 requests, GPU TE
seq_cpu = 4 * (cpu_t + dit_t)                  # 4 requests, naive CPU offload
pipelined = max(4 * cpu_t, gpu_t + 4 * dit_t)  # CPU TE overlapped with GPU work
print(f"SIMULATED 4-request wall: GPU-TE {seq_gpu:.1f}s | naive CPU offload "
      f"{seq_cpu:.1f}s | overlapped {pipelined:.1f}s")
print("naive offload LOSES unless CPU encode overlaps GPU denoise")'''),
        md("## 5 · Your turn\n"
           "1. With your pilot numbers, how many CPU workers would the "
           "overlapped design need to hide the CPU encode fully?\n"
           "2. Why does 'CPU offload' still win on **VRAM** even when it "
           "loses on latency? When is that the right trade?\n"
           "3. Name one reason your host's CPU/GPU ratio may differ from the "
           "frozen campaign's 2 orders of magnitude.",
           "## 5 · 动手练习\n"
           "1. 用你的试点数字计算：重叠设计需要多少 CPU worker 才能完全隐藏 CPU "
           "编码时间？\n"
           "2. 即使在延迟上输了，为什么 CPU 卸载在 **VRAM** 上仍然划算？什么时候"
           "这是正确的权衡？\n"
           "3. 举一个理由：你这台主机的 CPU/GPU 比值为什么会与冻结实验的两个"
           "数量级不同？"),
        md(acceptance_md("en"), acceptance_md("zh")),
        md(repro_md("en", "H1"), repro_md("zh", "H1") + "\n\n"
           "```bash\npython -m lab run --experiment H2 --date <your-tag>  # pipelined arm\n```"),
    ]


def m06() -> list:
    return [
        md("# M5.1 · OpenVINO CPU text encoder — the recovery and the fair A/B\n"
           "\n"
           "**Learning objective:** validate the OpenVINO BF16 CPU encoder "
           "recovery (Part A: local encoder speed), the real end-to-end "
           "architecture benefit (Part B: GPU pipeline with CPU TE "
           "offload), and the **fair factorial** that separates "
           "architecture effects from client-monitoring effects (Part C). "
           "Encoder speedup ≠ image throughput gain — never conflate them.\n\n"
           "Reference lesson: [M5.1 lesson](../course/M05_parallel_heterogeneous/M5_1_OPENVINO_CPU_TE.md) · "
           "[中文](./06_openvino_and_fair_ab.zh-CN.ipynb)",
           "# M5.1 · OpenVINO CPU 文本编码器 —— 恢复与公平 A/B\n"
           "\n"
           "**学习目标：** 验证 OpenVINO BF16 CPU 编码器的恢复（A 部分：本地编码器"
           "速度）、真实的端到端架构收益（B 部分：CPU TE 卸载的 GPU 流水线），以及"
           "区分架构效应与客户端监控效应的**公平因子实验**（C 部分）。编码器加速比 "
           "≠ 图像吞吐收益——永远不要混淆。\n\n"
           "参考讲义：[M5.1 讲义](../course/M05_parallel_heterogeneous/M5_1_OPENVINO_CPU_TE.zh-CN.md) · "
           "[English](./06_openvino_and_fair_ab.ipynb)"),
        code(header("06") + "\n" + MEASURED_TAG + "\n" + GPU_OR_SKIP),
        md("## 0 · Artifact admission — PASS vs PASS_WITH_WARNINGS\n"
           "The OV IRs are verified byte-level against their conversion "
           "manifests (streaming SHA256, source-model digest chain). An "
           "unpinned field keeps the verdict at `PASS_WITH_WARNINGS` — that "
           "distinction is preserved, never rounded up to PASS.",
           "## 0 · 工件准入 —— PASS 与 PASS_WITH_WARNINGS\n"
           "OV IR 会对照其转换清单做字节级校验（流式 SHA256、源模型摘要链）。"
           "存在未锁定字段时结论保持 `PASS_WITH_WARNINGS`——这一区别要保留，"
           "绝不允许四舍五入成 PASS。"),
        code('''def ov_verify():
    import subprocess, sys
    r = subprocess.run([sys.executable, "-c",
        "import os,sys; sys.path.insert(0,'.');"
        "from lab.m51_contract import verify_artifact;"
        "root=os.environ.get('M51_OV_ARTIFACTS', 'runs/m51-ov-artifacts');"
        "print(verify_artifact(root+'/bf16',"
        "['language_model_bf16.xml','language_model_bf16.bin'],"
        "expected_identity={'storage':'FP16','runtime_precision':'bf16'}))"],
        cwd=str(ROOT), capture_output=True, text=True, timeout=300)
    print(r.stdout or r.stderr)
    return r.returncode

rc = gpu_cell(ov_verify)
if rc is None:
    print("REFERENCE mode: artifact verification runs on the measurement host; "
          "campaign status recorded bf16 = PASS_WITH_WARNINGS (xml digest "
          "computed but not pinned by the historical manifest)")'''),
        md("## Part A · Local encoder: GPU PyTorch vs CPU PyTorch vs OpenVINO BF16\n"
           "Frozen reference ratios (encoder only, same CPU): OV BF16 is "
           "~331–480× faster than PyTorch CPU. **Do not blindly assert your "
           "host reproduces this** — check the campaign/current records "
           "below.",
           "## A 部分 · 本地编码器：GPU PyTorch vs CPU PyTorch vs OpenVINO BF16\n"
           "冻结参考比值（仅编码器、同一 CPU）：OV BF16 比 PyTorch CPU 快约 "
           "331–480×。**不要盲目断言你的主机能复现**——请核对下面的当前记录。"),
        code('''import json
enc = ns.read_json("runs/m51-20261105-cleanroom-01/performance/encoder_ab.json")
for arm in ("G_PyTorch_GPU", "C0_PyTorch_CPU", "C1b_OV_BF16"):
    row = enc["arms"][arm]["prompts"]["p1-zh-scene"]
    print(f"{arm:16s} warm median {row['warm_median_s']:8.3f}s  n={row['n_warm']}")
_stamp(HISTORICAL, "frozen encoder A/B (cleanroom campaign, 16-thread cgroup quota)")

def latest_h3():
    cands = sorted((ROOT / "runs").glob("*H3_OV_BF16/H3_RESULTS.json"))
    cands += sorted((ROOT / "runs").glob("*H3_OV_BF16-smoke/H3_RESULTS.json"))
    if not cands:
        return None
    return json.loads(cands[-1].read_text())

h3 = gpu_cell(latest_h3)
if h3 is not None:
    print("gates:", json.dumps(h3.get("gates", {}), indent=1))
    _stamp(MEASURED_TODAY, f"current-campaign H3_OV_BF16 gates above ({h3.get('mode')})")
    print("release_status:", h3.get("release_status"))
else:
    print("no current-campaign OV encoder record on this host; Part A stays "
          "reference-scoped here")'''),
        md("## Part B · End-to-end: GPU-native (A) vs OpenVINO-CPU TE (C)\n"
           "Same GPU, DiT, VAE, scheduler, resolution, prompts, seeds and "
           "client policy; four real requests per arm. The honest metric is "
           "**batch wall time / images per minute**, with service latency "
           "reported separately.",
           "## B 部分 · 端到端：GPU 原生（A）vs OpenVINO CPU TE（C）\n"
           "相同的 GPU、DiT、VAE、调度器、分辨率、提示词、种子与客户端策略；每个臂"
           "四个真实请求。诚实的指标是**批次墙钟时间 / 每分钟图像数**，服务延迟单独"
           "报告。"),
        code('''def latest_pipeline_ab():
    cands = sorted((ROOT / "runs").glob("*m51pipe*/pipeline_ab.json")) + \\
            sorted((ROOT / "runs").glob("m51-*/pipeline/pipeline_ab.json"))
    if not cands:
        return None
    import json as _j
    return _j.loads(cands[-1].read_text())

pab = gpu_cell(latest_pipeline_ab)
if pab is not None:
    for arm, row in pab["arms"].items():
        if isinstance(row, dict) and "batch_wall_s" in json.dumps(row)[:400]:
            print(arm, {k: row.get(k) for k in ("batch_wall_s", "images_per_min",
                                                "gpu_total_s", "cpu_encode_s")
                        if k in row})
    _stamp(MEASURED_TODAY, "pipeline A/C arms from the current campaign")
else:
    frozen = ns.read_json("runs/m51-20261105-cleanroom-01/pipeline/pipeline_ab.json")
    for arm in frozen["arms"]:
        print("frozen", arm, str(frozen["arms"][arm])[:160])
    _stamp(HISTORICAL, "frozen cleanroom pipeline A/B")'''),
        md("## Part C · The fair factorial — architecture × client monitoring\n"
           "Comparing C-fast against A-original overstates the architecture "
           "effect: A-original also pays per-request monitor teardown. The "
           "2×2 factorial (A/C × original/fast) isolates both axes. The old "
           "study found a *small* architecture benefit under matched "
           "clients and a larger client effect — treat as hypotheses.",
           "## C 部分 · 公平因子实验 —— 架构 × 客户端监控\n"
           "拿 C-fast 对比 A-original 会夸大架构效应：A-original 还要支付每请求的"
           "监控器拆除成本。2×2 因子设计（A/C × original/fast）把两个轴分开。旧研究"
           "发现：客户端匹配时架构收益*较小*，而客户端效应更大——把它们当作假设。"),
        code('''ff = ns.fair_factorial(clean_head=False)
cells = ff["cells"]
print(f"{'cell':12s} {'n':>2s} {'median_s':>9s} {'img/min':>8s}")
for k, v in cells.items():
    print(f"{k:12s} {v['n']:2d} {v['median_s']:9.2f} {v['images_per_min']:8.2f}")
arch = cells["A_original"]["median_s"] / cells["C_original"]["median_s"]
arch_fair = cells["A_fast"]["median_s"] / cells["C_fast"]["median_s"]
client_a = cells["A_original"]["median_s"] / cells["A_fast"]["median_s"]
client_c = cells["C_original"]["median_s"] / cells["C_fast"]["median_s"]
_stamp(HISTORICAL,
       f"fair factorial: architecture effect {arch:.3f}x (original clients) "
       f"vs {arch_fair:.3f}x (matched fast clients); client effect "
       f"A {client_a:.3f}x / C {client_c:.3f}x")
print("never report C_fast vs A_original as a pure architecture effect")'''),
        md("## Correctness scope — what stays FAIL and what stays human\n"
           "1. The INT8 encoder arm's numerical **FAIL stays FAIL** — no "
           "automatic quality promotion anywhere in this course.\n"
           "2. The C1b owner acceptance is scoped to the **historically "
           "evaluated deployment**; your new physical run needs its own "
           "correctness evidence.\n\n" + HUMAN_QUALITY_EN,
           "## 正确性范围 —— 什么保持 FAIL，什么留给人工\n"
           "1. INT8 编码器臂的数值 **FAIL 永远是 FAIL**——本课程任何地方都不会自动"
           "提升质量结论。\n"
           "2. C1b 所有者验收只限定于**历史评估过的部署**；你的新物理运行需要自己的"
           "正确性证据。\n\n" + HUMAN_QUALITY_ZH),
        md("## Your turn\n"
           "1. Compute encoder-only speedup vs pipeline-level speedup from "
           "the cells above — why do they differ by an order of magnitude?\n"
           "2. Which Part-C cell would you double-sample to tighten the "
           "architecture estimate, and why?\n"
           "3. Propose the RAM admission you would run before a PyTorch "
           "CPU full arm on a 55 GiB cgroup.",
           "## 动手练习\n"
           "1. 用上面的单元计算“仅编码器加速比”与“流水线加速比”——为什么差一个"
           "数量级？\n"
           "2. Part C 中你会给哪个格子加样以收紧架构效应估计？为什么？\n"
           "3. 在 55 GiB cgroup 上跑 PyTorch CPU 完整臂之前，你会做怎样的 RAM "
           "准入检查？"),
        md(acceptance_md("en"), acceptance_md("zh")),
        md(repro_md("en", "H3_OV_BF16"), repro_md("zh", "H3_OV_BF16")),
    ]


def m07() -> list:
    return [
        md("# M6 · Combined optimization and the quality gate\n"
           "\n"
           "**Learning objective:** measure the combined BEST configuration "
           "(Turbo distilled 8-step + INT8 weights) against B0 on **your** "
           "W7900, produce paired quality evidence, and keep the final "
           "verdict human. The published 6.12× is a reference; your number "
           "is your own.\n\n"
           "Reference lesson: [M06 lesson](../course/M06_combined_quality/LESSON.md) · "
           "[中文](./07_combined_and_quality.zh-CN.ipynb)",
           "# M6 · 组合优化与质量门\n"
           "\n"
           "**学习目标：** 在**你的** W7900 上测量组合 BEST 配置（Turbo 蒸馏 8 步 + "
           "INT8 权重）与 B0 的差距，产生成对质量证据，并让最终判定留给人工。"
           "已发表的 6.12× 只是参考；你自己的数字才是你的。\n\n"
           "参考讲义：[M06 讲义](../course/M06_combined_quality/LESSON.zh-CN.md) · "
           "[English](./07_combined_and_quality.ipynb)"),
        code(header("07") + "\n" + MEASURED_TAG + "\n" + GPU_OR_SKIP),
        md("## 1 · Run B0 and BEST physically\n"
           "Both arms with the profile matching your mode; reuse of your "
           "M1 B0 run is explicit (shown run_id).",
           "## 1 · 物理运行 B0 与 BEST\n"
           "两个臂都按你的模式运行；复用 M1 的 B0 时会显式标明 run_id。"),
        code('''def run_m6_pair():
    prof = "classroom" if session.mode == "GPU_CLASSROOM" else "research"
    b0 = session.run("B0", profile=prof)
    best = session.run("BEST", profile=prof)
    return b0, best

m6 = gpu_cell(run_m6_pair)
if m6 is None:
    ids = [session.latest_campaign_run("B0"), session.latest_campaign_run("BEST")]
    if all(ids):
        m6 = (session.consume(ids[0], why="campaign B0"),
              session.consume(ids[1], why="campaign BEST"))
if m6 is not None:
    ab = rt.ab_compare(m6[0], m6[1], label="B0 vs BEST (turbo+int8)")
    print({k: ab[k] for k in ("classification", "speedup_median",
                              "latency_delta_s", "vram_delta_mb")})
    _stamp(MEASURED_TODAY, f"B0/BEST speedup {ab['speedup_median']}x "
          f"[{ab['classification']}] vs frozen reference 6.12x")
    fig = rt.plot_ab(ab) or rt.plot_ab(ab, metric="vram")
    if fig: fig.show()'''),
        md("## 2 · The quality axis — paired evidence, human verdict\n"
           "Combined changes (model variant × schedule × precision) get "
           "**distance semantics**. Produce or inspect the paired grid + "
           "native-resolution poster crops; score with the three-tier "
           "rubric. LPIPS here is a perceptual distance that *triages* "
           "review — it never issues the verdict.\n\n" + HUMAN_QUALITY_EN,
           "## 2 · 质量维度 —— 成对证据，人工判定\n"
           "组合变更（模型变体 × 调度 × 精度）采用**距离语义**。生成或检查成对网格"
           "与原生分辨率海报裁剪；用三级评分标准打分。这里的 LPIPS 是用于*分流*评审"
           "的感知距离——它永远不下结论。\n\n" + HUMAN_QUALITY_ZH),
        code('''print("three-tier rubric:  (1) no perceptible change  "
      "(2) slight degradation, still deployable  (3) unacceptable")
print()
def run_own_quality():
    session.run_quality("BEST", reference="B0")
    qdir = ROOT / "runs/quality/BEST"
    for f in sorted(qdir.glob("*.png")):
        print("  your artifact:", f.name)
    import json as _j
    lp = qdir / "lpips_table.json"
    if lp.is_file():
        print(_j.dumps(_j.loads(lp.read_text()), indent=1)[:800])
    return qdir

q = gpu_cell(run_own_quality)
if q is None:
    print("frozen campaign artifacts (reference):")
    for f in sorted((ROOT / "evidence/quality/BEST").glob("*")):
        print("  ", f.name)
    print("historical M6 BEST independent human review: PENDING_HUMAN_REVIEW "
          "(owner decision boundary preserved)")
_stamp("GATE STATUS", "your verdict on YOUR images is PENDING_HUMAN_REVIEW "
      "until you complete the rubric on the grid above")'''),
        md("## 3 · Read your own combination honestly\n"
           "1. Is your measured B0/BEST ratio close to 6.12×? If not, which "
           "arm differs from the reference (compare B0 and BEST medians "
           "separately against 55.206 s / 9.023 s)?\n"
           "2. The combined ratio is NOT F1×F2×L1 multiplied — why?\n"
           "3. BEST's VRAM peak is ~6.6 GB below B0's — what does that "
           "enable?",
           "## 3 · 诚实解读你的组合结果\n"
           "1. 你测得的 B0/BEST 比值接近 6.12× 吗？若不接近，是哪个臂与参考不同"
           "（分别对比 B0 与 BEST 中位数和 55.206 秒 / 9.023 秒）？\n"
           "2. 组合比值不等于 F1×F2×L1 相乘——为什么？\n"
           "3. BEST 的 VRAM 峰值比 B0 低约 6.6 GB——这能带来什么？"),
        code('''if m6 is not None:
    b0s, bests = m6[0].stats(), m6[1].stats()
    print(f"B0   {b0s['median_s']:7.2f}s (ref 55.206)  VRAM {b0s['vram_device_peak_mb']:.0f}MB (ref 33153)")
    print(f"BEST {bests['median_s']:7.2f}s (ref 9.023)  VRAM {bests['vram_device_peak_mb']:.0f}MB (ref 26514)")
    print("report the delta per arm, not only the ratio")'''),
        md(acceptance_md("en"), acceptance_md("zh")),
        md(repro_md("en", "BEST"), repro_md("zh", "BEST")),
    ]


def m08() -> list:
    return [
        md("# M7 · Capstone — your reproducible optimization PR\n"
           "\n"
           "**Learning objective:** assemble a genuine experimental report "
           "from **your own** W7900 records — measured baseline, measured "
           "candidate, declared protocol, quality scope, limitations, and a "
           "reproduction command — and draft the optimization PR. If your "
           "physical evidence is smoke/classroom-only, the report is "
           "labelled a **PILOT**, never a proven general improvement.\n\n"
           "Reference lesson: [M07 challenge](../course/M07_student_challenge/CHALLENGES.md) · "
           "[中文](./08_capstone_and_pr.zh-CN.ipynb)",
           "# M7 · 毕业课题 —— 你的可复现优化 PR\n"
           "\n"
           "**学习目标：** 用**你自己的** W7900 记录组装一份真实的实验报告——实测"
           "基线、实测候选、声明的协议、质量范围、局限与复现命令——并起草优化 PR。"
           "如果你的物理证据只有 smoke/课堂级，报告必须标注 **PILOT**，绝不能声称"
           "已证明的普遍性能提升。\n\n"
           "参考讲义：[M07 挑战](../course/M07_student_challenge/CHALLENGES.zh-CN.md) · "
           "[English](./08_capstone_and_pr.ipynb)"),
        code(header("08") + "\n" + MEASURED_TAG),
        md("## 1 · Declare your experiment\n"
           "Fill in your identity and pick the eligible module whose records "
           "you will report (a module with a genuine physical measurement "
           "from this course: M1, M3, M4, M5, M6).",
           "## 1 · 声明你的实验\n"
           "填写你的身份信息，并选择你要报告的合格模块（本课程中拥有真实物理测量的"
           "模块：M1、M3、M4、M5、M6）。"),
        code('''STUDENT_ID = "your-name-here"          # <- edit
STUDENT_NOTE = "what you claim and why it matters"
ELIGIBLE_MODULE = "M4"                 # M1|M3|M4|M5|M6 (module with YOUR physical run)
CANDIDATE_EXPERIMENT = {"M1": ("B0", "B0"), "M3": ("B0", "L1"),
                        "M4": ("B0", "F1"), "M5": ("A", "C"),
                        "M6": ("B0", "BEST")}.get(ELIGIBLE_MODULE, ("B0", "BEST"))
BASE_EXPERIMENT, CAND_EXPERIMENT = CANDIDATE_EXPERIMENT
print(f"capstone: {STUDENT_ID} reports {ELIGIBLE_MODULE}: "
      f"{BASE_EXPERIMENT} vs {CAND_EXPERIMENT}")'''),
        md("## 2 · Collect the evidence (own run first, campaign fallback)\n"
           "The collector prefers **your own** physical runs from this "
           "session, then your earlier runs under `runs/`, and only then "
           "the shared qualification-campaign records (explicitly labelled "
           "as shared). Reference-only capstones are labelled as such.",
           "## 2 · 收集证据（优先自己的运行，其次实验记录）\n"
           "收集顺序：**你自己**本会话的物理运行 → 你在 `runs/` 下的早期运行 → "
           "共享的资格认证实验记录（显式标注为共享）。仅有参考数据的毕业课题会被"
           "明确标注。"),
        code('''def collect_evidence(exp):
    own = [e for e in session.executions if e.get("experiment") == exp]
    if own:
        b = rt.load_run(own[-1]["run_id"])
        return b, "own physical run (this session)"
    rid = session.latest_campaign_run(exp)
    if rid and session.mode != "REFERENCE":
        return rt.load_run(rid), f"shared campaign record {rid}"
    rid = rt.list_runs(exp)
    if rid:
        return rt.load_run(rid[-1]), f"existing run {rid[-1]}"
    return None, "no physical record"

base_b, base_src = collect_evidence(BASE_EXPERIMENT)
cand_b, cand_src = collect_evidence(CAND_EXPERIMENT)
for nm, b, src in ((BASE_EXPERIMENT, base_b, base_src),
                   (CAND_EXPERIMENT, cand_b, cand_src)):
    print(f"{nm:5s} {src}" + (f" run_id={b.run_id}" if b else ""))'''),
        md("## 3 · Compute the claim and its correct scope label\n",
           "## 3 · 计算结论及其正确的范围标签\n"),
        code('''import subprocess

def git_head():
    r = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                       capture_output=True, text=True, timeout=10)
    return r.stdout.strip() or "unknown"

report = {
    "student_id": STUDENT_ID, "module": ELIGIBLE_MODULE,
    "claim": STUDENT_NOTE,
    "git_head": git_head(),
    "gpu": session.admission.detail if session.admission else None,
    "baseline": {"experiment": BASE_EXPERIMENT, "source": base_src,
                 "stats": base_b.stats()} if base_b else None,
    "candidate": {"experiment": CAND_EXPERIMENT, "source": cand_src,
                  "stats": cand_b.stats()} if cand_b else None,
}
if base_b and cand_b:
    ab = rt.ab_compare(base_b, cand_b)
    report["comparison"] = {k: ab[k] for k in
                            ("classification", "speedup_median",
                             "vram_delta_mb", "paired_per_prompt")}
    if ab["classification"] == "FORMAL_PROTOCOL_MEASURED":
        report["evidence_class"] = "PHYSICAL_FORMAL"
    elif "own physical run" in base_src:
        report["evidence_class"] = "PHYSICAL_PILOT"
    else:
        report["evidence_class"] = "SHARED_CAMPAIGN_PHYSICAL"
else:
    report["evidence_class"] = "REFERENCE_ONLY"
print("evidence_class:", report["evidence_class"])
print(json.dumps(report.get("comparison", {}), indent=1)[:600])'''),
        md("## 4 · Write the report (append-only, under `runs/`)\n"
           "Never into frozen `evidence/`. The file name is unique; "
           "re-running creates a new record instead of overwriting.",
           "## 4 · 写入报告（只追加，位于 `runs/`）\n"
           "绝不写入冻结的 `evidence/`。文件名唯一；重复运行会创建新记录而不是覆盖。"),
        code('''report["limitations"] = [
    "speedup is a ratio of the two declared populations only",
    "single-host, single-GPU (Radeon Cloud W7900 instance)",
    "quality verdict stays PENDING_HUMAN_REVIEW unless you completed the M6 rubric",
]
report["reproduce"] = (
    f"python -m lab run --experiment {CAND_EXPERIMENT} --date <your-tag>")
from datetime import datetime
_cap_ts = datetime.now().strftime("%H%M%S")
out = ns.student_record(report, f"capstone-{_cap_ts}", confirmed=True)
print("capstone report:", out)'''),
        md("## 5 · Your PR draft\n"
           "Copy the template below into your PR description; every bracket "
           "must be replaced by YOUR values, and the title must carry the "
           "evidence class from section 3.",
           "## 5 · 你的 PR 草稿\n"
           "把下面的模板复制进你的 PR 描述；每个方括号都必须替换为**你的**数值，"
           "标题必须带上第 3 节得到的证据等级。"),
        code('''pr_draft = f"""# optimization: [{report['evidence_class']}] {ELIGIBLE_MODULE} {BASE_EXPERIMENT} vs {CAND_EXPERIMENT}

## Hypothesis
[what you predicted and why]

## Protocol
- GPU: {(report['gpu'] or {}).get('device_name', 'n/a')} @ ROCm {(report['gpu'] or {}).get('hip', '')}
- populations: baseline {(report['baseline'] or {}).get('stats', {}).get('population', 'n/a')} | candidate {(report['candidate'] or {}).get('stats', {}).get('population', 'n/a')}
- run records: {(report['baseline'] or {}).get('stats', {}).get('run_id', '')} , {(report['candidate'] or {}).get('stats', {}).get('run_id', '')}

## Results
- speedup median: {report.get('comparison', {}).get('speedup_median')}
- VRAM delta: {report.get('comparison', {}).get('vram_delta_mb')} MB

## Quality
- [rubric verdict or PENDING_HUMAN_REVIEW]

## Limitations
{chr(10).join('- ' + l for l in report['limitations'])}

## Reproduce
{report['reproduce']}
"""
print(pr_draft)'''),
        md(acceptance_md("en"), acceptance_md("zh")),
    ]


MODULES = {
    "00": ("00_experimental_discipline.ipynb", "00_experimental_discipline.zh-CN.ipynb",
           "course/M00_setup_and_discipline/LESSON.md", m00),
    "01": ("01_baseline_and_uncertainty.ipynb", "01_baseline_and_uncertainty.zh-CN.ipynb",
           "course/M01_baseline/LESSON.md", m01),
    "02": ("02_profile_and_bottlenecks.ipynb", "02_profile_and_bottlenecks.zh-CN.ipynb",
           "course/M02_profiling/LESSON.md", m02),
    "03": ("03_less_computation.ipynb", "03_less_computation.zh-CN.ipynb",
           "course/M03_less_computation/LESSON.md", m03),
    "04": ("04_faster_computation.ipynb", "04_faster_computation.zh-CN.ipynb",
           "course/M04_faster_computation/LESSON.md", m04),
    "05": ("05_cpu_offload_negative.ipynb", "05_cpu_offload_negative.zh-CN.ipynb",
           "course/M05_parallel_heterogeneous/LESSON.md", m05),
    "06": ("06_openvino_and_fair_ab.ipynb", "06_openvino_and_fair_ab.zh-CN.ipynb",
           "course/M05_parallel_heterogeneous/M5_1_OPENVINO_CPU_TE.md", m06),
    "07": ("07_combined_and_quality.ipynb", "07_combined_and_quality.zh-CN.ipynb",
           "course/M06_combined_quality/LESSON.md", m07),
    "08": ("08_capstone_and_pr.ipynb", "08_capstone_and_pr.zh-CN.ipynb",
           "course/M07_student_challenge/CHALLENGES.md", m08),
}


def main() -> int:
    check = "--check" in sys.argv
    for key, (en, zh, _lesson, builder) in MODULES.items():
        spec = builder()
        for fname, lang in ((en, "en"), (zh, "zh")):
            path = NB / fname
            if check:
                old = path.read_text(encoding="utf-8") if path.is_file() else ""
                import tempfile
                with tempfile.NamedTemporaryFile("w", suffix=".ipynb",
                                                 delete=False) as tf:
                    write_notebook(Path(tf.name), lang, "", spec)
                    new = Path(tf.name).read_text(encoding="utf-8")
                if new != old:
                    print(f"STALE: {fname} (rebuild required)")
                    return 1
                continue
            write_notebook(path, lang, "", spec)
            print(f"wrote {fname} ({len(spec)} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
