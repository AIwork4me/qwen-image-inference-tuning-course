# P6 — Clean-HEAD Physical Validation (Bounded Post-Commit Regression)

**Session:** final reproducibility & publication polish, 2026-11-06.
**Run ID:** `p4-fairab-cleanhead-003750` (new, unique; original records untouched)
**Raw record:** `runs/p4-fairab-cleanhead-003750/pipeline_profile/pipeline_profile.json`

## 1. Purpose and preconditions

The original P4 campaign executed with a dirty worktree whose exact runtime
bytes are not independently recoverable
(`SOURCE_RECONSTRUCTED_WITH_EVIDENCE` — see
[`P6_SOURCE_PROVENANCE_AUDIT.md`](P6_SOURCE_PROVENANCE_AUDIT.md)). This phase
re-measures the factorial's four cells from a **fully committed, verified
source state** to establish source-to-execution reproducibility. It is NOT a
new speedup claim and does NOT alter the original dataset or its labels.

Preconditions met: physical gfx1100 W7900-class host free; OV artifacts
byte-verified this session (`provision_ov.sh --check-only` →
`ARTIFACTS_VERIFIED`); RAM/disk sufficient; no concurrent load; all
provenance work committed **before** the run (`4f2b7d7`→`edc8950`).

## 2. Source state of THIS run (recorded by the new provenance machinery)

From the run's own `environment.source_provenance`:

- `provenance_class`: **`EXACT_SOURCE_VERIFIED`**
- git HEAD `edc89508…`, branch `feat/world-class-bilingual-inference-course`,
  `dirty: false`, `dirty_files: []`, `tracked_diff_sha256` = sha256 of the
  empty diff (`e3b0c442…`)
- all 9 `EXPERIMENT_MODULES` (runner, helpers, pipeline plumbing, worker,
  custom node) classified `COMMITTED_CLEAN` — the executed bytes are exactly
  the committed blobs, recoverable by digest from git
- dependencies probed: python 3.12.3; worker venv OpenVINO
  `2026.4.1-22982-e213a147257`, torch `2.9.1+gitff65f5b` (both venvs)
- argv, cgroup limits (`1600000/100000`, 55 GiB), bf16 manifest recorded
  in-run; conditioning SHA256 per prompt byte-identical to BOTH prior studies
  (2026-11-05 P2 and the original P4) — third independent cross-day
  confirmation of worker determinism.

## 3. Design (bounded, pre-declared)

`--pairs 1 --blocks 2` → 2 counterbalanced blocks × (A session + C session) ×
(poll + fast) = **2 batches per cell, 8 measured batches**; identical frozen
prompts/token IDs/seed/latent/DiT/VAE/scheduler/8 steps/cfg=1/1024²; same
completion semantics, warmups and reload policy as the original (2 warmups in
A sessions; measured batches scanned for loader re-execution — **0 reloads**).
GPU budget used: **671 s ≈ 11.2 min** of the 30-min ceiling.

## 4. Result vs original campaign

| Cell | Clean-HEAD mean (n=2) | Original mean (n=6) | Δ | Original range |
|---|---:|---:|---:|---|
| A-original | 53.297 s / 4.503 img/min | 53.996 s / 4.445 | −1.3% | 53.620–54.994 |
| A-fast | 50.954 s / 4.710 | 50.894 s / 4.716 | +0.1% | 50.668–51.200 |
| C-original | 52.611 s / 4.562 | 52.643 s / 4.559 | −0.1% | 52.427–52.959 |
| C-fast | 50.079 s / 4.792 | 49.967 s / 4.803 | +0.2% | 49.705–50.226 |

| Effect | Clean-HEAD | Original | Direction |
|---|---:|---:|---|
| Architecture gain, original clients (A/C) | 1.0130 | 1.0257 | reproduced (+1.3% vs +2.6%) |
| Architecture gain, fast clients (A/C) | 1.0175 | 1.0186 | reproduced (+1.8% vs +1.9%) |
| Monitor gain within A | 1.0460 | 1.0609 | reproduced (+4.6% vs +6.1%) |
| Monitor gain within C | 1.0506 | 1.0536 | reproduced (+5.1% vs +5.4%) |
| Interaction | +0.19 s | −0.43 s | small, sign-unstable at n=2 — as expected for noise |

Three of four cells land within 0.2% of the original means; A-original is
1.3% faster here (the original cell contained one 54.99 s block-2 batch;
both clean-HEAD A-poll batches sit at the original's low edge, 53.29/53.31).
**Verdict: `CLEAN_HEAD_PHYSICAL_REGRESSION_PASS`** — direction and approximate
scale of every published effect reproduce from a verified committed source
state.

## 5. What this does and does not establish

- Establishes: the committed implementation, executed from a clean HEAD with
  verified artifacts on the same hardware/budget class, reproduces the P4
  factorial's effects within ~1.5% (three cells) to ~1.3% (one cell); the
  conditioning pipeline is deterministic across three studies and two days.
- Does NOT establish: statistical equivalence (n=2 per cell is a smoke-scale
  sample — stated, not hidden); any claim about the ORIGINAL campaign's exact
  runtime bytes (that dataset keeps
  `SOURCE_RECONSTRUCTED_WITH_EVIDENCE` forever); universality beyond this
  host/workload scope.
- Both datasets remain separately labeled and separately committable; the
  original `runs/p4-fairab-225326` JSON is untouched (blob `7d100108…`).

## 6. Reproduction

Same commands as the P4 guide (`docs/publication-closure/P4_FAIR_CLIENT_AB.md`
§6) with the bounded parameters:

```bash
export M51_OV_ARTIFACTS=/opt/m51-artifacts   # example root
bash scripts/m51/provision_ov.sh --check-only
git status --porcelain                        # must be empty
M51_OV_ARTIFACTS=$M51_OV_ARTIFACTS .venv-lab/bin/python -m lab.m51_pipeline_fair_ab \
  --run runs/<new-unique-run-id> --port 8214 --pairs 1 --blocks 2
```

Verify `environment.source_provenance.provenance_class == EXACT_SOURCE_VERIFIED`
in the output JSON before trusting the run as a clean-HEAD reproduction.
