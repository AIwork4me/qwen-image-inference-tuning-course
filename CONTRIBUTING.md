# Contributing

Students and community members welcome — this repo's whole point is
reproducible practice. The bar is evidence, not eloquence.

## Ground rules

1. **Reproduce the baseline before proposing changes**
   (`python -m lab doctor` then `python -m lab baseline`).
2. **One variable per experiment**; never multiply independent speedups;
   combinations are measured end-to-end.
3. **A run record or it didn't happen**: schema-valid JSON, attention-path
   evidence, image hashes. Negative results are welcome — with measurements.
4. **Quality semantics are declared** (fidelity vs distance); the human
   three-tier verdict (无感/略降/不可接受) is never auto-generated.
5. Don't touch frozen artifacts: `lab/prompts.json`, `lab/schema/`,
   `lab/experiments/*.json` (B0 especially), and anything under `evidence/`
   (append new runs; corrections land as new runs + errata, never edits).

## Workflow

```bash
python3 -m venv .venv-lab && .venv-lab/bin/pip install -r requirements-dev.txt
.venv-lab/bin/python -m pytest tests/ -v      # CPU-only gate; must be green
```

- Branch from `main`, conventional commits, small atomic diffs.
- GPU work: attach your run-record directory + SUMMARY row to the PR.
- Exercise solutions: keep them in your own fork (not in PRs).

## Student tasks

See [course/M07_student_challenge/CHALLENGES.md](course/M07_student_challenge/CHALLENGES.md)
— each task lists difficulty, reproduction commands, expected outputs,
acceptance criteria and evidence requirements. Difficulty labels are honest:
GPU kernel work is never marked "good first issue".

## Reporting bugs in the lab

Attach `python -m lab doctor` output and the relevant record JSON.
