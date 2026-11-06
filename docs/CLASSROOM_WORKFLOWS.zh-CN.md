# 课堂工作流 —— 冒烟 vs 正式，具体化

*一页纸，无歧义：课堂上跑什么（有界、便宜、可安全中断），要出可引用结果时跑什么
（完整协议、有预算）。这里的一切都与 CLI 完全一致（`python -m lab <cmd> --help`）。*
English version: [CLASSROOM_WORKFLOWS.md](CLASSROOM_WORKFLOWS.md)

---

## 课堂安全的冒烟工作流（设计上有界）

**成本：每条命令 1–3 GPU 分钟。可安全中断——每次运行都可恢复，绝不重复下载。**

```bash
# 一次性安装（约 15 分钟，只下载冒烟所需）
cd /workspace/qwen-image-inference-tuning-course
bash scripts/cloud/start.sh

# 课程的冒烟命令（任何实验运行都可加 --smoke）：
.venv-lab/bin/python -m lab run --experiment B0   --smoke --date m1-smoke
.venv-lab/bin/python -m lab run --experiment L1   --smoke --date m3-smoke
.venv-lab/bin/python -m lab run --experiment F1   --smoke --date m4-smoke
.venv-lab/bin/python -m lab run --experiment F2   --smoke --date m4-smoke
.venv-lab/bin/python -m lab run --experiment BEST --smoke --date m6-smoke
```

`--smoke` 做了什么（由 runner 强制，不靠约定）：
- 1 个提示词 × 1 次测量运行（外加一次冷启动），代替 3×3 + 新提示词臂；
- 只写入 `runs/<tag>-<ID>-smoke/`（`-smoke` 后缀正是 report 用来把它们排除在
  正式中位数之外的东西）；
- 绝不触碰 `evidence/`（冻结战役）——由 CI 测试保证。

查看冒烟结果而不污染任何东西：

```bash
.venv-lab/bin/python -m lab report --scope student --kind smoke   # 只读
.venv-lab/bin/python -m lab report --scope student --kind formal  # 只读
```

**中断恢复：** 被杀掉的运行在 `runs/` 留下部分目录；用*新* `--date` 重跑即干净
开始（证据只增不改）。模型/venv/ComfyUI 由重跑 `start.sh` 校验而非重新下载
（幂等；实例重启后也安全）。

## 正式基准工作流（可引用中位数）

**成本：每个实验 4–16 GPU 分钟（预算表见下）。当你需要一个要引用、要与参考对比、
或要附到 PR 上的数字时再跑。**

```bash
bash scripts/cloud/start.sh --all-models    # 一次性：解锁完整实验矩阵
.venv-lab/bin/python -m lab doctor          # 严格预检必须 PASS

# 完整协议会话（全新服务端，冷+预热+3x3 测量+新提示词臂）：
.venv-lab/bin/python -m lab run --experiment B0   --date <your-tag>
.venv-lab/bin/python -m lab run --experiment L1   --date <your-tag>
#   ... F1 F2 L1O LF1 LF2 LF3 BEST

# 用你自己的原图做质量门（manifest 声明的协议）：
.venv-lab/bin/python -m lab quality --experiment F2
.venv-lab/bin/python -m lab quality --experiment BEST

# 聚合*你的*正式会话（绝不混入你的冒烟、绝不混入参考）：
.venv-lab/bin/python -m lab report --scope student --write
```

**GPU 时间预算（W7900 级、完整会话）：**

| 实验 | 完整 | 冒烟 | 备注 |
|---|---|---|---|
| B0 | 约 16 分钟 | 约 3 分钟 | 一切配对比较都需要它作为参考 |
| L1 / L1O | 约 5 分钟 | 约 2 分钟 | 需要 turbo-bf16 模型 |
| F1 | 约 13 分钟 | 约 3 分钟 | ck 注意力；对 B0 的保真度门 |
| F2 | 约 15 分钟 | 约 3 分钟 | int8；对 B0 的保真度门 |
| LF1–LF3、BEST | 约 4 分钟 | 约 2 分钟 | turbo-int8（默认已暂存） |
| H1 / H2 | 各约 15 分钟 | — | **仅自主研究**（缓慢的 CPU 负载）；默认从已提交证据讲授 |

**保持正式数字可引用的规则：**
1. 会话前严格 doctor PASS（错误的技术栈 = 不可比）。
2. 每个服务端生命周期只跑一个实验（runner 强制全新服务端）。
3. 只要完整协议——`--smoke` 输出是课堂证据，绝不是我方引用的中位数。
4. 用 `report --scope student --write` 出*你的*数字；参考范围默认只读
   （对它 `--write` 会重写冻结 SUMMARY——那是维护动作，不是学生动作）。
5. H1/H2 重跑是研究任务：每个预算约 15 分钟，默认教学材料仍是已提交的负结果。
