# P4 — Fair Physical A/B: Architecture × Client 2×2 Factorial (W7900)

**Session:** publication-closure Phase P4, 2026-11-06.
**Run ID:** `p4-fairab-225326` (new, unique; historical runs untouched)
**Raw record:** `runs/p4-fairab-225326/pipeline_profile/pipeline_profile.json`
**Module:** `lab/m51_pipeline_fair_ab.py` (experiment-mode; production harness untouched)
**Source HEAD at run:** `4ec5905a` (worktree carrying the pre-commit P4 module —
recorded in the run JSON `environment.git`, dirty-files list included)

## 1. Research question (pre-registered)

When BOTH arms use the same optimized client, how much end-to-end four-request
batch-throughput improvement remains attributable to OpenVINO CPU offloading?

The P2 study measured C-original 52.761 s vs C-fast 50.320 s (+4.85%, n=2 per
mode) and noted that quoting C-fast vs A-original ("1.079×") is not a fair
architecture estimate because A-original pays the same per-request monitor
teardown. This experiment completes the 2×2 factorial:

| | original client (per-request RunMonitor) | fast client (persistent RunMonitor) |
|---|---|---|
| **A: GPU-native TE** | A-original | **A-fast (first time measured)** |
| **C: OpenVINO BF16 CPU TE** | C-original | C-fast (P2's C-opt) |

## 2. Protocol (as pre-registered in the module docstring, before running)

- Frozen: 4 course prompts/token IDs/seed 42/latent/DiT/VAE/scheduler/8 steps/
  cfg=1/1024²; workflows byte-identical to the P2 study (same helpers).
- 2 blocks × (warm A session + warm C session); block 2 order reversed
  (C→A) to control session-order drift. Within every session, batches
  alternate `poll, fast, poll, fast, …` (3 pairs per session) — 6 batches per
  arm×mode cell, 24 measured batches, 96 requests total.
- A sessions run TWO warmups so the documented post-warmup CLIP-eviction
  reload lands outside measured batches; every measured batch is scanned for
  loader re-execution via node timings.
- Exclusion rule (pre-declared): fatal technical failure only; none occurred —
  **0 excluded batches, 0 in-batch model reloads** (the `primary_all_batches`
  and `warm_resident` views are identical for that reason).
- Admission: OV artifacts byte-verified against their conversion manifests
  before the run (streaming SHA256 vs pinned digests — the same check
  `bash scripts/m51/provision_ov.sh --check-only` performs); cgroup
  `1600000/100000` (16 CPUs) / 55 GiB — same budget class as every M5.1
  campaign run.
- GPU budget: **1,522 s ≈ 25.4 min** of the 60-min ceiling (including both
  blocks' server boots and warmups).

## 3. Results (actual four-request batch walls; img/min = 240/wall)

| Cell | n | mean s | median s | min–max s | img/min | throughput ranges |
|---|---:|---:|---:|---|---:|---|
| A-original | 6 | 53.996 | 53.765 | 53.620–54.994 | **4.445** | non-overlapping vs A-fast |
| A-fast | 6 | 50.894 | 50.899 | 50.668–51.200 | **4.716** | |
| C-original | 6 | 52.643 | 52.600 | 52.427–52.959 | **4.559** | non-overlapping vs C-fast |
| C-fast | 6 | 49.967 | 49.957 | 49.705–50.226 | **4.803** | |

Per-block stability (no drift): A-poll block1 mean 53.836 vs block2 54.156;
C-poll 52.566 vs 52.720; C-fast 49.937 vs 49.997; A-fast 50.882 vs 50.907.

### Factorial effects (the answer to §1)

| Effect | Estimate | Distribution evidence |
|---|---|---|
| **Architecture gain under equivalent ORIGINAL clients** (A-orig vs C-orig) | 53.996/52.643 = **1.026× (+2.6%)** | batch ranges non-overlapping (A ≥ 53.62 > C ≤ 52.96) |
| **Architecture gain under equivalent FAST clients** (A-fast vs C-fast) | 50.894/49.967 = **1.019× (+1.9%)** | batch ranges non-overlapping (A ≥ 50.67 > C ≤ 50.23) |
| **Client/monitor gain within A** | 53.996/50.894 = **1.061× (+6.1%)** | non-overlapping |
| **Client/monitor gain within C** | 52.643/49.967 = **1.054× (+5.4%)** | non-overlapping |
| **Interaction** (monitor effect A minus monitor effect C) | **−0.43 s** (A benefits slightly more) | consistent sign in both blocks |

**Conclusion:** with symmetric clients the architecture benefit of OpenVINO
CPU offloading on actual four-request batch throughput is **≈ +1.9 to +2.6%**
in this campaign's warm steady state — the earlier cross-population quote of
"1.079×" (C-fast vs A-original) overstated the architecture effect by mixing
client generations: most of that difference was the baseline's own monitor
overhead. The registr entry `RATIO-CFAST-VS-AWARM-1.079` stays in the
provenance registry only as declared cross-population motivation for this
factorial, exactly as labeled.

### Secondary observations (all from raw records)

- **The client distorted per-request service latency too:** A's service wall
  median falls 13.034 → 12.484 s under the fast client (range 12.307–12.712):
  the production client's 1.0 s history-poll quantization adds ~0.5 s to what
  we call "service latency". C's service wall is unchanged (12.066 → 12.096 s)
  because its completion detection already returned quickly. Lesson: even a
  per-request metric can carry harness artifact, not just batch gaps.
- Batch-gap (`loop_head`) medians: A-poll 0.53 s, C-poll 0.93 s, both fast
  modes ~0.001 s — the teardown mechanism is exactly as profiled in P2.
- One transient slow request (13.031 s inside block-1 C1_poll req2, others
  12.03–12.07): retained in the primary view per the pre-declared rule; it
  does not change any conclusion (C-poll mean would be 52.53 s without it —
  reported for transparency, not excluded).
- Resources: full-pipeline VRAM peak ≈ **34.0 GB (A) vs 17.0 GB (C)** — the
  ~17 GB VRAM saving reproduces; CPU cost: the OV worker adds **≈ 23.3 GB
  resident RSS** while the A-arm server carries ≈ 2.5 GB RSS (C server
  ≈ 1.9 GB) — CPU-RAM-for-VRAM is the real trade, as taught.
- Correctness contract: conditioning-file SHA256 per prompt identical across
  ALL C batches and modes, and **byte-identical to the 2026-11-05 P2 study's
  files** (cross-day worker determinism); identical seeds/workflows; image
  bytes differ only by the documented run-to-run kernel nondeterminism (all
  96 image SHA256s retained in the record).

### The P2 §7.1 prediction, checked honestly

P2 predicted A-fast ≈ 4×13.04 + 0.5 ≈ 52.7 s (4.56 img/min) and an A/C batch
ratio ≈ 1.05×. Measured: A-fast = 50.894 s (4.716 img/min), ratio 1.019×.
The prediction was **wrong in both directions** — instructively: it assumed
A's service wall would stay 13.04 s, but removing the per-request monitor
also removes the ~0.55 s poll-quantization inside that wall (§Secondary), so
A gained more than C from the same client fix (negative interaction), pulling
the architecture ratio down, not up. Predictions are hypotheses; this is why
we run factorials.

## 4. Statistical interpretation (small-n discipline)

- n = 6 batches per cell (12 per arm), batch-level distributions
  **non-overlapping** for all four reported contrasts; effects 1.9–6.1%.
- These are **descriptive results for one host, one campaign, turbo 8-step,
  cfg=1, 1024², 16-CPU cgroup**. No p-values are claimed; no claim of
  universality across hardware, sessions, workloads or step counts.
- The P2 C-only result (+4.85%, n=2/mode) stands unedited as prior evidence;
  this experiment strengthens and generalizes the mechanism (monitor teardown)
  with 6× the sample and both arms.

## 5. Adoption decision (explicitly not a retroactive success claim)

The P2 pre-registered 5% exploratory target applied to that n=2 C-only study,
which honestly reported a near-miss (+4.85%) and kept the production harness
unchanged. The P4 measurements (+6.1% within A, +5.4% within C, n=6/mode,
non-overlapping, mechanism-proven, zero server-side changes, no memory cost)
exceed that 5% figure **in a new, differently-sized experiment** — that is
new evidence, NOT a retroactive pass of the original pre-registration, and it
is labeled as such here. The production harness remains unchanged in this
closure (scope discipline); with the strengthened evidence the owner may now
reasonably adopt a persistent-monitor client as a separate, forward-looking
engineering decision (recommended follow-up: correctness tests + regression
coverage for the persistent monitor path before it serves production).

## 6. Reproduction

> ⚠ Do **not** feed `conversion_manifest.json` to `sha256sum -c` — it is a
> structured JSON manifest, not a GNU checksum list (an earlier revision of
> this section showed that invalid form). The supported workflow is the
> provisioning script's check mode, which streams every artifact against the
> manifest's pinned digests, sizes and conversion identity:

```bash
# 1) verified artifact admission (no download, no modification; /opt/m51-artifacts
#    is an EXAMPLE root — use your own $M51_OV_ARTIFACTS location):
export M51_OV_ARTIFACTS=/opt/m51-artifacts
bash scripts/m51/provision_ov.sh --check-only        # -> ARTIFACTS_VERIFIED

# 2) clean-worktree check (the run record carries source provenance):
git status --porcelain                              # should be empty

# 3) the factorial (physical W7900-class host, campaign cgroup budget):
M51_OV_ARTIFACTS=$M51_OV_ARTIFACTS \
  .venv-lab/bin/python -m lab.m51_pipeline_fair_ab \
  --run runs/<new-unique-run-id> --port 8213 --pairs 3 --blocks 2
```

- Expected output: `<run>/pipeline_profile/pipeline_profile.json` (per-request
  timelines, per-batch walls, conditioning SHA256, source manifest) plus
  server logs; raw images stay host-side under `raw/` (hashes in the record).
- Budget: ~25 min GPU for the full 96-request design; `--pairs 1 --blocks 2`
  gives a ~15 min bounded regression (2 batches/cell).
- Verify a rerun against this study by comparing the `analysis.views` cells
  (see `P6_P4_INDEPENDENT_REDERIVATION.md` for the recomputation script-free
  method) and the per-prompt `cond_sha256_stable_across_all_C_batches`
  entries, which must match this study's digests byte-for-byte.
- Recovery: on admission BLOCK, re-provision per the script's message; on
  port conflicts, change `--port`; failed batches are retained and marked.

Environment: gfx1100 48 GiB, EPYC 9334 (16-CPU/55 GiB cgroup), ROCm system
stack, PyTorch 2.9.1+rocm, ComfyUI pinned checkout, OpenVINO 2026.4.1
(`.venv-ov`). Analysis cells recomputed independently in
`docs/publication-closure/P1_METRIC_PROVENANCE.json` (population `P4-FAIRAB`).
Result limits: see §7.

## 7. Limitations

1. Single host, single day, two blocks; session-level variance is partially
   controlled (counterbalanced blocks) but not eliminated.
2. The architecture contrast necessarily crosses server sessions (A sessions
   instantiate the GPU CLIP; C sessions never load it) — inherent to the
   deployment difference being measured; both sessions are warm and
   reload-free by construction (verified per batch).
3. cfg=1 turbo scope; the INT8 arm (C2) is excluded from production
   consideration by its immutable numerical FAIL and was not part of this
   factorial.
4. The fast client is still experiment-mode code (~90 lines) living beside
   the frozen production path.
