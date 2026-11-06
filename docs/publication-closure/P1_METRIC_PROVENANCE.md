# P1 — Metric Provenance Audit & Cross-Run Mixing Repair

**Session:** publication-closure Phase P1, 2026-11-06.
**Machine-readable registry:** [`P1_METRIC_PROVENANCE.json`](P1_METRIC_PROVENANCE.json)
**CI gate:** [`scripts/course-quality/check_metric_pairs.py`](../../scripts/course-quality/check_metric_pairs.py)
(wired into CI; negative tests in `tests/test_course_quality.py`).

## 1. Findings (before repair)

| # | Location | Defect | Why wrong |
|---|---|---|---|
| F1 | `README.md` Mermaid, M5.0 node | `4 reqs: 737 s` paired with `0.35 img/min` | 737.126 s is the **H2 campaign** B arm; its only consistent partner is 240/737.126 = **0.33 img/min**. 0.35 belongs to the **M5.1 rebaseline** B arm (680.004 s → 0.3529). Pairing them quotes one experiment's time with a different experiment's throughput. |
| F2 | `README.md` Mermaid, M5.1 node | `4 reqs: 50 s` paired with `4.55 img/min + 17.26 GB` | 50.046 s is the **cleanroom campaign cold-inclusive** C1b batch → **4.80 img/min** (VRAM 17.26 GB *is* the campaign definition). 4.55 img/min is the **balanced warm audit** (52.7415 s mean; its VRAM figure is 16.18 GB). Triple-mixing in one node. |
| F3 | `README.zh-CN.md` Mermaid | identical defects (F1+F2) in Chinese | same |
| F4 | `docs/M51_RESULTS.md` §3.4 | `1.04× steady-state` stated without the estimate label in that line | 1.04× was the campaign's *estimate*; the directly measured warm results are 1.081× (service) / 1.029× (actual batch). Labeled at the top table but not at §3.4. |

**Audit result for the wider publication set** (SYLLABUS EN/ZH, METHODOLOGY EN/ZH,
LECTURE_DEMO EN/ZH, both M5.0 lessons, both M5.1 lessons, M51_RESULTS,
M51_E2E_AND_PIPELINE, PIPELINE_OVERHEAD_STUDY, verified_claims.json): no further
mixed-population pairs found. The M5.0 lessons correctly pair 737.1 s ↔ 0.33
(L106/62); the M5.1 lessons correctly pair 680.00 s ↔ 0.35 as the rebaseline
(L108/116) and 50.05 ↔ 4.80 (L109); the lecture pairs 737.1 ↔ 0.33 (L120/103);
`docs/M51_RESULTS.md` L22 explicitly labels every population. The defects were
concentrated in the two README diagrams added by the course-v2 redesign.

## 2. Root cause

The README redesign compressed two campaigns into one diagram for visual
impact: the M5.0 node took the *time* from the historical H2 campaign but the
*throughput* from the newer M5.1 rebaseline (0.35 looked "less embarrassing"
next to the recovery numbers); the M5.1 node took the campaign's cold-inclusive
time and VRAM but the warm audit's throughput. No validator existed: CI checked
bilingual *presence* of values, never their arithmetic identity or population.

## 3. Repair (Pattern A — one population per node)

Both diagrams now use strictly same-campaign pairs, with the population named
in the subgraph title and a caption that (a) shows the arithmetic
(`240/737.1 = 0.33`, `50.05 → 4.80`), (b) labels cold-inclusive vs warm, and
(c) points to the warm numbers quoted separately in the arc text
(service 1.081×, batch 1.029× = 4.42 → 4.55 img/min). `M51_RESULTS.md` §3.4
now labels 1.04× as the campaign estimate and names the measured successors.

Every corrected pair (verified by recomputation from raw records):

| Published pair | Population | Recomputation |
|---|---|---|
| 737 s / 0.33 img/min | H2 campaign (`evidence/results/p5-H2-H2-pipeline/H2_RESULTS.json`) | 4·60/737.126 = 0.3256 → 0.33 |
| 680.00 s / 0.35 img/min | M5.1 rebaseline B (cleanroom campaign) | 4·60/680.004 = 0.3529 → 0.35 |
| 50.05 s / 4.80 img/min (cold-incl.) | cleanroom campaign C1b | 4·60/50.046 = 4.7956 → 4.80 |
| 54.28 s / 4.42 img/min (warm) | balanced warm audit A (mean of 54.421, 54.136) | 4·60/54.2785 = 4.4216 |
| 52.74 s / 4.55 img/min (warm) | balanced warm audit C (mean of 52.773, 52.710) | 4·60/52.7415 = 4.5505 |
| 52.761 s / 4.549 img/min | course-v2 P2 C-original | 4·60/52.761 = 4.5488 |
| 50.320 s / 4.770 img/min | course-v2 P2 C-fast | 4·60/50.320 = 4.7695 |
| 1.081× service | balanced warm audit | 13.031/12.060 = 1.0805 |
| 1.029× batch | balanced warm audit | 4.5505/4.4216 = 1.0291 (also 54.2785/52.7415) |
| 1.33× cold-inclusive | cleanroom campaign | 66.615/50.046 = 1.3311 |
| 1.0485× monitor gain | course-v2 P2 | 52.761/50.320 = 1.0485 |
| 1.079× C-fast vs A-original | **declared cross-population** (see §4) | 4.7695/4.4216 = 1.0787 |
| 6.12× B0→BEST | course benchmark campaign | 55.206/9.023 = 6.1184 |
| 17.26 GB vs 16.18 GB VRAM | campaign vs balanced-audit definitions | taught as non-interchangeable |

## 4. Cross-population claims policy (Pattern B)

Exactly one published ratio deliberately spans populations: **1.079×** — C-fast
(course-v2 P2 session) vs A-original (balanced warm audit). It is retained
*only* as the motivating observation for the P4 fair factorial and is registered
in the provenance JSON as `derived_cross_population` with an explicit note. The
clean architecture estimate is the P4 symmetric-client result
(`P4_FAIR_CLIENT_AB.md`).

## 5. Enforcement (fail-closed)

`check_metric_pairs.py` scans the publication set; every co-located
(time, throughput) pair in one Mermaid node / table cell / slash-pair must be

1. **arithmetically consistent**: tput ≈ n·60/wall within 2% (a mixed pair like
   (737 s, 0.35) fails: 240/737 = 0.326 ≠ 0.35), **and**
2. **registered**: matching a `P1_METRIC_PROVENANCE.json` entry's numerator and
   throughput within 1.5%/2% (catches fabricated-but-arithmetic-consistent pairs).

Negative tests inject the historical defects — `(737 s, 0.35 img/min)`,
`(50 s, 4.55 img/min)` — and assert the gate fails; positive tests assert the
repaired pairs and all honest documents pass. The gate runs in CI next to the
other four course-quality gates.

## 6. Verification

- `python scripts/course-quality/check_metric_pairs.py` → **OK** on the repaired
  tree; on the pre-repair READMEs it reports exactly the four defects of §1
  (F1/F2 EN + F1/F2 ZH), no false positives elsewhere.
- `check_claims.py`, `check_links.py`, `check_bilingual.py`, `check_examples.py`
  all still pass after the repair.
