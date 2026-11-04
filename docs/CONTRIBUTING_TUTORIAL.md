# Contributing Tutorial — Fork → Edit → Test → PR

*A complete walkthrough for first-time contributors. Estimated time:
30–60 minutes (mostly the test cycle; the PR itself takes 5 minutes).*

This tutorial contributes a real, small improvement: adding one regression
test to the quality tooling (M7 Task 3). The same workflow applies to any
task in [the challenges](../course/M07_student_challenge/CHALLENGES.md).

## 0. What you need

- A GitHub account.
- A machine with Python 3.12 (your Radeon Cloud instance works; so does
  any laptop — the CPU tests need no GPU).
- 30–60 minutes.

## 1. Fork (1 minute, in the browser)

1. Open <https://github.com/AIwork4me/qwen-image-inference-tuning-course>.
2. Click **Fork** (top right) → create `YOURNAME/qwen-image-inference-tuning-course`.
3. You now own a copy. All work happens in your fork; the upstream repo
   receives your work only through a Pull Request that the owner reviews.

## 2. Clone your fork and create a branch (2 minutes)

```bash
cd /workspace   # or any directory on your machine
git clone https://github.com/YOURNAME/qwen-image-inference-tuning-course.git
cd qwen-image-inference-tuning-course
git remote add upstream https://github.com/AIwork4me/qwen-image-inference-tuning-course.git
git checkout -b test/quality-collect-measured     # descriptive branch name
```

## 3. Set up the test environment (5 minutes, CPU-only)

```bash
python3 -m venv .venv-contrib
.venv-contrib/bin/pip install -r requirements-dev.txt
.venv-contrib/bin/python -m pytest tests/ -q
```

All tests must pass before you change anything — if they don't on a clean
checkout, stop and report an issue instead of building on a broken base.
(On a Radeon Cloud instance you can reuse the course venv:
`.venv-lab/bin/python -m pytest tests/ -q`.)

## 4. Edit (the actual contribution)

Study the code you're about to test. For our example, read
`lab/quality_gate.py` → `collect_measured()`:

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

Write the test — append to `tests/test_lab.py` (follow the existing
style: plain functions, docstring stating WHAT property is pinned):

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
    assert len(m) == 3          # cold/warmup/txt files are excluded
```

## 5. Test (the loop)

```bash
.venv-contrib/bin/python -m pytest tests/ -q          # everything green?
.venv-contrib/bin/python -m pytest tests/test_lab.py::test_quality_collect_measured_parsing -v
```

Rules of the loop:
- Red first if practical (break the function, watch the test fail — a
  test that can't fail protects nothing).
- Never weaken an existing assertion to make yours pass.
- Keep tests CPU-only and fast (the CI runner has no GPU; GPU evidence
  belongs in `runs/`, attached to the PR, not in unit tests).

## 6. Commit (atomic, conventional)

```bash
git add tests/test_lab.py
git commit -m "test(quality): pin collect_measured filename parsing

cold/warmup/txt artifacts must never appear as measured pairs;
(prompt_id, run_idx) mapping pinned for multi-seed names."
```

One logical change per commit; the subject line states what, the body
states why.

## 7. Push and open the PR

```bash
git push -u origin test/quality-collect-measured
```

Then on GitHub: **Compare & pull request**. A good PR description
contains:

1. **What** — one sentence.
2. **Why** — which challenge/task it addresses, or the bug it fixes.
3. **Evidence** — test output pasted; for GPU work, run-record dirs under
   `runs/` and the numbers you measured (never just conclusions).
4. **Scope honesty** — what you did NOT test / cannot claim.

Mark it **Draft** while work is in progress; the owner reviews every PR,
and all quality verdicts personally.

## 8. Review etiquette

- CI must be green (the CPU gate runs on every PR automatically).
- Respond to review comments with commits, not arguments; if you disagree,
  disagree with evidence.
- Never force-push after review starts (reviewers lose context).
- If your PR touches published numbers or evidence, expect extra
  scrutiny: the frozen campaign under `evidence/` is immutable — a PR
  that modifies it will be rejected unless it *is* an explicitly approved
  evidence-correction with full justification.

## Common first-PR mistakes (each has happened)

| Mistake | Why it's rejected |
|---|---|
| Committing `runs/raw/` PNGs or weights | repo hygiene: records yes, bulk artifacts no (.gitignore enforces) |
| Editing a frozen manifest (B0.json etc.) to "fix" results | manifests are the protocol; changes = new experiment id |
| A new benchmark number without its run record | claims ≤ evidence is the course's first law |
| "It works on my machine" without the machine's facts | records must capture GPU/stack — that's what makes them evidence |
| Tests that mock so much they test the mock | pin real behavior with real (tiny) inputs |
