# PR #2 — Release Gate Matrix and Closure Summary

Audit session: 2026-11-05 (independent). Baseline `6aacd6f` → fixes layered on top (P8 commits).
Full reports in `docs/release-closure/`.

## Gate matrix

| Gate | Before | After | Evidence | Status | Blocking? |
|---|---|---|---|---|---|
| P0 baseline integrity | unverified | git/PR/CI reconciled; 493 blob IDs + 138 raw SHA256 frozen | `PR2_P0_BASELINE.md`, `PR2_BASELINE_FREEZE_MANIFEST.txt` | **PASS** | yes |
| P0 artifact integrity (bytes) | presence + field-name check only | streaming SHA256 + size + identity + path safety at admission; 45.5 GB physically verified | `PR2_P0_CORRECTNESS.md` §P0-B; `provision_ov.sh --check-only` output | **PASS** (bf16 xml pin = warning, digest recorded `28bd053a…`) | yes |
| P0 token semantics | `token_ids_self_consistent=false` unexplained | production contract pinned (`lab/m51_token_contract.json`), live-verified incl. empty/whitespace/nonempty; diagnostic fixed; OV semantics unchanged (were correct) | `PR2_P0_CORRECTNESS.md` §P0-C; `probe_cpu.json` of audit sessions (`ids_equal=true`) | **PASS** | yes |
| P0 scientific status propagation | finite⇒PASS | gates + `release_status`; INT8 negative FAIL structurally unfakeable; BLOCKED never upgraded; human verdict only via explicit record | `PR2_P0_CORRECTNESS.md` §P0-A; `runs/pr2audit-114153-H3_OV_INT8-smoke` (execution PASS → RELEASE_FAIL) | **PASS** | yes |
| P1 resource safety (RAM) | cgroup ceiling = "free" | conservative min(host avail, limit−anon) − headroom − reserve; unlimited fallback | `PR2_PORTABILITY_AND_RESOURCE_SAFETY.md` §P1-A; 6 tests | **PASS** | yes |
| P1 portability (paths) | `/opt/m51-artifacts` hardcoded in pipeline; profile.py shadow crash; frozen env overwriteable | one `$M51_OV_ARTIFACTS` policy everywhere; sys.path guard; write-guard | `PR2_PORTABILITY_AND_RESOURCE_SAFETY.md` §P1-B | **PASS** | yes |
| P1 provisioning/recovery | check-only = presence check | byte-verify only, no side effects; atomic manifests; worker quit/kill; portable TMPDIR | same, §P1-C; live `--check-only` | **PASS** | no |
| P1 encoder A/B statistics | n=1 C0 warm/prompt | +3 paired sessions: C0 within 0.5%, p2 speedup 480.7× (n=3/6) | `PR2_STATISTICAL_…AUDIT.md` §P1-D; `runs/pr2-p4-cpuab-v2-114711` | **PASS_WITH_LIMITATIONS** (p1/p3 still single-session; bounded by design) | no |
| P1 pipeline fairness (steady attribution) | 1.04× steady = estimate | measured balanced warm A/C: **1.081×** (n=8/8, non-overlapping) + recurring 26–30 s session-start penalty characterized | same §P1-E; `runs/pr2-p4-pipewarm-120822` | **PASS** | no |
| P1 precision integrity | at risk of conflation | FP32 / FP32-IR+BF16-hint / FP16-IR+BF16-exec / INT8 kept distinct; identity checked at admission; verdicts preserved (C1b-dep PASS, C1b-uncomp FAIL, C2 FAIL) | alignment JSON verdicts + new identity gates; docs language verified | **PASS** | yes |
| P1 quality package | PENDING (implicit) | explicit review package with exact paths, automated metrics separated, verdict protocol | `PR2_HUMAN_QUALITY_REVIEW_PACKAGE.md` | **PENDING_HUMAN_REVIEW** | yes (for final merge) |
| P1 course consistency | SYLLABUS/LECTURE_DEMO stale (NOT_RUN) | reconciled with measured scope + limitations; M0–M4 path untouched | `PR2_COURSE_READINESS.md` | **PASS** | no |
| CPU CI | 53 tests green at `6aacd6f` | **82 passed** locally (new negative/gate coverage); CI to confirm on push | pytest output; Actions run on push | **PASS** (local; CI pending push) | yes |
| W7900 physical regression | campaign only | doctor (GPU/ROCm/ComfyUI PASS; 2 absent non-turbo weights = known deviation), digest verify, gated smokes BF16/INT8, formal BF16 run, balanced warm A/C, VRAM/RSS — all at audit HEAD with gates | `runs/pr2audit-*`, `runs/pr2-p4-*` | **PASS** | yes |
| Historical evidence immutability | assumed | 138/138 raw files byte-identical post-audit; `evidence/` and frozen runs untouched (git + SHA256) | freeze manifest re-comparison | **PASS** | yes |
| Human image-quality verdict | PENDING_HUMAN_REVIEW | unchanged by design — owner action required | `PR2_HUMAN_QUALITY_REVIEW_PACKAGE.md` | **PENDING_HUMAN_REVIEW** | yes (pre-merge) |
| 40-step BF16 DiT / CFG sweep | NOT_RUN | NOT_RUN (scoped out; VRAM claim explicitly not extrapolated) | `M51_LIMITATIONS.md` | **NOT_RUN** | no |

## Three-conclusion framework

- **A. Engineering correctness: PASS** — reproducible execution, byte-verified artifacts,
  pinned token contract, byte-identical conditioning reproduction, honest gating.
- **B. Performance claim readiness: PASS_WITH_LIMITATIONS** — encoder speedup and VRAM claims
  reproduced and correctly attributed (steady ≈8% now directly measured; cold-inclusive
  1.30–1.33× labeled as session-start effect); limitations: single host, p1/p3 C0 single-session,
  bounded sample counts stated everywhere.
- **C. Repository release/merge readiness: READY_FOR_OWNER_REVIEW** — every automated,
  technical hard gate passes; the only pre-merge item is the owner's human image-quality
  verdict (PENDING by design, cannot be automated).

PR #2 remains **Draft**; no merge performed; no Approve review submitted (independent audit).
