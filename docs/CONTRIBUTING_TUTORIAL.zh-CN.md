# 贡献教程 —— Fork → 编辑 → 测试 → PR

*面向首次贡献者的完整走查。预计时间：30–60 分钟（主要在测试循环；PR 本身只要
5 分钟）。* English version: [CONTRIBUTING_TUTORIAL.md](CONTRIBUTING_TUTORIAL.md)

本教程贡献一个真实的小改进：给质量工具加一个回归测试（M7 任务 3）。同一流程适用
于[挑战任务](../course/M07_student_challenge/CHALLENGES.zh-CN.md)中的任何题目。

## 0. 你需要什么

- 一个 GitHub 账号。
- 一台有 Python 3.12 的机器（你的 Radeon Cloud 实例可以；任何笔记本也行——CPU
  测试不需要 GPU）。
- 30–60 分钟。

## 1. Fork（1 分钟，浏览器里完成）

1. 打开 <https://github.com/AIwork4me/qwen-image-inference-tuning-course>。
2. 点右上角 **Fork** → 创建 `YOURNAME/qwen-image-inference-tuning-course`。
3. 你现在拥有一份副本。所有工作在你的 fork 里进行；上游仓库只通过由所有者评审的
   Pull Request 接收你的工作。

## 2. 克隆你的 fork 并建分支（2 分钟）

```bash
cd /workspace   # 或你机器上的任意目录
git clone https://github.com/YOURNAME/qwen-image-inference-tuning-course.git
cd qwen-image-inference-tuning-course
git remote add upstream https://github.com/AIwork4me/qwen-image-inference-tuning-course.git
git checkout -b test/quality-collect-measured     # 有描述性的分支名
```

## 3. 搭建测试环境（5 分钟，纯 CPU）

```bash
python3 -m venv .venv-contrib
.venv-contrib/bin/pip install -r requirements-dev.txt
.venv-contrib/bin/python -m pytest tests/ -q
```

改任何东西之前所有测试必须通过——干净检出都不通过的话，停下来报 issue，不要在
坏基线上叠代码。（Radeon Cloud 实例上可复用课程 venv：
`.venv-lab/bin/python -m pytest tests/ -q`。）

## 4. 编辑（真正的贡献）

先研究你要测的代码。本例中阅读 `lab/quality_gate.py` → `collect_measured()`：

```python
def collect_measured(raw_dir):
    """Map (prompt_id, run_idx) -> png for measured runs (-r<idx>.png files)."""
    out = {}
    for f in sorted(Path(raw_dir).glob("*.png")):
        parts = f.stem.split("-")
        if len(parts) >= 3 and parts[-1].startswith("r") and parts[-1][1:].isdigit():
            out[("-".join(parts[:-2]), int(parts[-1][1:]))] = f
    return out
```

写测试——追加到 `tests/test_lab.py`（沿用现有风格：普通函数、docstring 写明钉住的
是什么性质）：

```python
def test_quality_collect_measured_parsing(tmp_path):
    """collect_measured maps filenames to (prompt_id, run_idx) correctly."""
    from lab.quality_gate import collect_measured

    names = ["p1-zh-scene-42-r0.png", "p2-en-scene-1337-r1.png",
             "p1-zh-scene-1000042-cold.png", "p1-zh-scene-2000042-warmup.png",
             "notes.txt", "p3-poster-text-20261030-r2.png"]
    for n in names:
        (tmp_path / n).write_bytes(b"png")
    m = collect_measured(tmp_path)
    assert m[("p1-zh-scene", 0)] == tmp_path / "p1-zh-scene-42-r0.png"
    assert m[("p2-en-scene", 1)] == tmp_path / "p2-en-scene-1337-r1.png"
    assert ("p3-poster-text", 2) in m
    assert len(m) == 3          # cold/warmup/txt 文件被排除
```

## 5. 测试（循环）

```bash
.venv-contrib/bin/python -m pytest tests/ -q          # 全绿？
.venv-contrib/bin/python -m pytest tests/test_lab.py::test_quality_collect_measured_parsing -v
```

循环规则：
- 尽可能先看到红（把函数改坏，看测试失败——不会失败的测试什么也保护不了）。
- 绝不为了让自己通过而削弱既有断言。
- 测试保持纯 CPU 且快速（CI runner 没有 GPU；GPU 证据放在 `runs/` 里附到 PR，
  不进单元测试）。

## 6. 提交（原子、约定式）

```bash
git add tests/test_lab.py
git commit -m "test(quality): pin collect_measured filename parsing

cold/warmup/txt artifacts must never appear as measured pairs;
(prompt_id, run_idx) mapping pinned for multi-seed names."
```

每次提交一个逻辑改动；主题行说"做了什么"，正文说"为什么"。

## 7. 推送并开 PR

```bash
git push -u origin test/quality-collect-measured
```

然后在 GitHub 上：**Compare & pull request**。好的 PR 描述包含：

1. **What** —— 一句话。
2. **Why** —— 对应哪个挑战任务，或修复什么 bug。
3. **Evidence** —— 粘贴测试输出；GPU 工作附 `runs/` 下的运行记录目录和你测到的
   数字（绝不只是结论）。
4. **Scope honesty** —— 你*没有*测 / 不能主张什么。

工作进行中标记 **Draft**；所有者评审每一个 PR，并亲自评审所有质量判定。
性能优化类 PR 请使用仓库的优化模板
（`.github/PULL_REQUEST_TEMPLATE/optimization.md`）：瓶颈、改动、不变量、前后数字、
质量影响、资源成本、可复现性、未验证项。

## 8. 评审礼仪

- CI 必须绿（CPU 门在每个 PR 上自动运行）。
- 对评审意见用提交回应，不用争论；有分歧时用证据分歧。
- 评审开始后绝不 force-push（评审者会失去上下文）。
- 若你的 PR 触及已发表的数字或证据，预期会被额外严格审查：`evidence/` 下的冻结
  战役不可变——修改它的 PR 会被拒绝，除非它*本身就是*一个经明确批准、带完整
  理由的证据更正。

## 首次 PR 常见错误（每一种都真发生过）

| 错误 | 为什么被拒 |
|---|---|
| 提交 `runs/raw/` 的 PNG 或权重 | 仓库卫生：记录可以，大批工件不行（.gitignore 强制） |
| 改冻结 manifest（B0.json 等）来"修"结果 | manifest 就是协议；要改 = 新实验 ID |
| 只有新基准数字、没有运行记录 | 主张 ≤ 证据是课程第一定律 |
| "在我机器上能跑"却不给机器事实 | 记录必须捕获 GPU/技术栈——那才是它们成为证据的原因 |
| mock 太多以致于在测 mock | 用真实（微小的）输入钉住真实行为 |
