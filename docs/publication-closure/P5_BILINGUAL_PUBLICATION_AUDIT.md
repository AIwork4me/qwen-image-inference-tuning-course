# P5 — Bilingual Publication Audit (README / Curriculum / Lecture)

**Session:** publication-closure Phase P5, 2026-11-06.
**Scope:** focused scientific and instructional editing of the existing
bilingual curriculum — no rewrite. Both language tracks verified to carry
identical scientific conclusions.

## 1. Repairs made

### 1.1 README Mermaid diagrams (both languages) — cross-run mixing eliminated

See [`P1_METRIC_PROVENANCE.md`](P1_METRIC_PROVENANCE.md) §1–3. Before: `737 s`
paired with `0.35 img/min`, `50 s` paired with `4.55 img/min` + `17.26 GB` in
EN and ZH alike. After: strictly same-population pairs
(`737 s / 0.33 张-per-min`, `50.05 s cold-incl. / 4.80 + 17.26 GB campaign
definition`), population named in each subgraph title, arithmetic shown in the
caption, cold-inclusive vs warm labeled, and the mixed-population defect
documented as a teaching moment with its CI enforcement. The zh-CN track
received the identical repair with identical numbers (bilingual numeric parity
gate + metric-pair gate both green).

### 1.2 The "measure the measurer" lesson made explicit (both READMEs)

New item 4 in the instructive arc: the benchmark client itself was part of the
bottleneck (~0.5–1.0 s/request monitor teardown in BOTH arms); fixing only C's
client (+4.85%) made a C-vs-A quote measure the harness; the 2×2 factorial
{A,C}×{original,fast} is the fairness experiment (links to
[`P4_FAIR_CLIENT_AB.md`](P4_FAIR_CLIENT_AB.md)). Identical reasoning and
numbers in 中文.

### 1.3 New scientific-fairness student exercise (both M5.1 lessons)

Exactly the mission's exercise — "a developer improves C from 52.76 s to
50.32 s and compares C-fast against A-original; why is this not fair; design
the separating experiment" — added to `M5_1_OPENVINO_CPU_TE.md` (Advanced
tier) and `M5_1_OPENVINO_CPU_TE.zh-CN.md`, each with the full answer
(unbound-teardown confound; 2×2 factorial decomposition: architecture effect
under equivalent clients, client effect within each arm, interaction) and the
evidence pointers (run `overnight-course-v2-a7fa69a-p2ab-152000`, the P4
study, registry entry `RATIO-CFAST-VS-AWARM-1.079` declared
cross-population).

### 1.4 Stale 1.04× estimate de-staled

`docs/M51_RESULTS.md` §3.4 now labels `1.04×` as the campaign *estimate* and
names the measured warm successors (1.081× service / 1.029× batch) — matching
the M5.1 lessons (EN L122/142, ZH L132/149), which already taught the
distinction.

## 2. Bilingual consistency verification (both tracks, same conclusions)

| Property | EN carrier | ZH carrier | Gate |
|---|---|---|---|
| Claim IDs / experiment identities | preserved | preserved | `check_bilingual` structure + manual |
| Critical values (55.21, 9.02, 6.12, 1.081, 1.029, 13.031, 12.060, 4.42, 4.55, 331, 480, 28.25, 1.33, 16.18, 17.26) | present | present | `check_claims` PARITY_VALUES |
| Time/throughput population consistency | enforced | enforced | `check_metric_pairs` (new) |
| Cold/warm semantics | labeled in diagram caption + arc item 3/4 | identical | manual + P1 audit |
| Quality-verdict wording (C1b owner-accepted scope-only; INT8 FAIL; uncompressed FAIL; M6 BEST pending; 40-step NOT_RUN) | unchanged | unchanged | `check_claims` strict enums |
| Fairness exercise + answer | M5.1 lesson | M5.1 lesson (中文) | manual review, structure gate |
| Commands runnable | yes | yes | `check_examples` |

All five gates pass on the repaired tree:
`check_links`, `check_bilingual`, `check_claims`, `check_examples`,
`check_metric_pairs`.

## 3. README quality requirements check

- Opening screen: what the course teaches, measured 6.12× with its
  **independent pending quality gate** stated in the caveat block, M5.0
  failure → M5.1 recovery arc, the three optimization strategies
  (§"What you will learn"), three-step start path, EN/中文 navigation — all
  present in both languages (unchanged from course-v2 redesign, verified).
- Mobile-readable: no wide tables added by this pass; the diagram caption is
  prose.
- No new low-value decoration; no rewrite — the diff is surgical (diagram
  nodes, captions, one arc item, one exercise, one stale-estimate label,
  per language).

## 4. Deliberately NOT changed

- The honest +4.85% near-miss narrative and the decision to keep the
  production harness unchanged (pre-registration discipline).
- Any frozen measurement, any FAIL/NOT_RUN verdict, the owner-verdict scope.
- The lecture documents (they already carry Q5's service-vs-batch lesson;
  the new exercise links to the same study rather than duplicating it).
