# Course-v2 Scientific Claims Audit (P0-B/P0-C)

Independent re-verification of every load-bearing claim against committed
evidence, 2026-11-05/06. Verdicts: **VERIFIED** (number traced to evidence),
**DRIFT** (stale/incorrect presentation — fixed in this PR), **HONEST**
(negative/pending correctly labeled).

## 1. Course-wide core claims (M0–M6)

| # | Claim | Verified value & source | Verdict |
|---|---|---|---|
| C1 | B0 frozen baseline warm latency 55.21 s | `evidence/results/SUMMARY.md` row B0: 55.206 s, n=18 (two 9-run sessions) | VERIFIED |
| C2 | Steady step 1.345 s | SUMMARY B0 `step med` 1.345 | VERIFIED |
| C3 | Turbo 8-step L1 = 12.13 s = 4.55× | SUMMARY L1 12.132; 55.206/12.132 = 4.55 | VERIFIED |
| C4 | ck attention F1 = 45.09 s = 1.22× | SUMMARY F1 45.086 | VERIFIED |
| C5 | INT8 F2 = 51.09 s = 1.08×, VRAM 37.7→31.0 GB | SUMMARY F2 51.093, 37688.6→31024.7 MB | VERIFIED |
| C6 | BEST = 9.02 s = 6.12× | SUMMARY BEST 9.023; 55.206/9.023 = 6.116 → 6.12 | VERIFIED |
| C7 | Amdahl cap ~4.6× explains "not 5×" | DiT fraction 97.6% ⇒ 1/(0.024+0.976/5)=4.63 | VERIFIED (derived, labeled as such) |
| C8 | LF2=LF3 (ck adds nothing on int8-turbo) | 9.023 vs 9.022 s, n=9/9 | VERIFIED |
| C9 | M5.0 CPU TE 152–391× slower; VRAM −16.8 GB | H1_RESULTS.json (frozen) | VERIFIED |
| C10 | M5.0 pipeline 14× throughput loss; 50.4 s overlap | H2_RESULTS.json (frozen) | VERIFIED |
| C11 | M6 BEST quality verdict | Still independently `PENDING_HUMAN_REVIEW` (`runs/quality/BEST/quality.md`); M5.1 acceptance does NOT cover it | HONEST (must stay scoped) |

## 2. M5.1 claims

| # | Claim | Verified value & source | Verdict |
|---|---|---|---|
| M1 | Encoder C0→C1b 331–480× on positive prompts | M51_RESULTS §3.1: 331/480/374× (p1/p2/p3); **328×** on the empty negative. Scope label "on positive prompts" is required wherever 331–480 is quoted | VERIFIED with scope condition |
| M2 | C1a FP32 alignment PASS ≤5.4e-6 | alignment JSON verdict `C1a: PASS` | VERIFIED |
| M3 | C1b deployment PASS, max 1.91% (empty negative) | alignment JSON verdict `C1b_compressed: PASS` | VERIFIED |
| M4 | C2 INT8 empty-negative FAIL 28.25% | alignment JSON verdict `C2: FAIL` — immutable; owner acceptance cannot promote it | VERIFIED, must stay FAIL |
| M5 | C1b-uncompressed negative FAIL 6.63% | alignment JSON | VERIFIED, must stay FAIL |
| M6 | Warm service latency 13.031→12.060 s = 1.081× | `PR2_LATENCY_VS_THROUGHPUT.json` audit_warm_sessions A/C (n=8/8, non-overlapping ranges 13.029–13.033 vs 12.027–12.198) | VERIFIED |
| M7 | Actual 4-request batch throughput 4.42→4.55 img/min = 1.029× | same JSON corrected_summary | VERIFIED |
| M8 | Cold-inclusive pipeline 66.62→50.05 s = 1.33× | campaign_cold_inclusive A/C | VERIFIED (distinct metric) |
| M9 | VRAM peak saving 17.26 GB (campaign definition: 34.15→16.77 GB warm-session peak) vs 16.18 GB (balanced audit session) | Two different sessions/definitions; `PR2_STATISTICAL_…AUDIT.md` §P1-E keeps them separate | VERIFIED — definitions must never be interchanged |
| M10 | Human quality verdict | Owner acceptance 2026-11-05, scope C1b-vs-G; recorded `PR2_OWNER_QUALITY_VERDICT.json` (+ provenance page) via `settle_human_quality` | RECORDED with explicit scope |

## 3. Documentation drift found and repaired (P0-C)

1. **M5.1 lesson §3.4/§3.6** taught the retired "**1.04× steady-state**"
   estimate as the measured result. → Repaired: the lesson now teaches the
   three distinct notions (cold-inclusive 1.33× / warm service 1.081× / actual
   batch throughput 1.029×) with the mechanism behind each (CLIP reload
   avoidance / encoder share / orchestration overhead), and links the raw JSON.
2. **"331–480×" scope**: M5.0 lesson and lecture quote 331–480× without the
   positive-prompt scope; the empty negative is 328×. → Scope label added
   where the range is quoted (M5.1 lesson now states both).
3. **README bilingual asymmetry**: English README carried the current Radeon
   Cloud application link and Chinese guide link; the Chinese README used
   generic `developer.amd.com.cn/radeon` + docs link and contains **zero**
   M5.1 numbers, and neither README foregrounds the M5.0-negative →
   M5.1-recovery arc. → Repaired in the P4 README redesign (both languages,
   full parity, verified links).
4. **Owner-verdict propagation**: `lab/m51_run.py` previously always settled
   `human_quality=PENDING`; the committed owner-verdict record is now consumed
   strictly scope-matched (H3_OV_BF16 only; INT8/FP32 keep PENDING), with
   negative tests.
5. **Link audit**: all 27 relative markdown links from README.zh-CN and all 13
   from course/ re-verified — zero broken (the audited pre-existing state was
   already clean; new content must keep it that way, enforced by the new
   `scripts/course-quality/check_links.py`).

## 4. Forbidden inferences (re-asserted, unchanged)

LPIPS low ⇒ high quality; distance high ⇒ bad; human verdict on 9 pairs ⇒ all
users; any claim about prompts never run; any propagation of the M5.1 human
acceptance to C2/C1b-uncompressed/M6-BEST/40-step-scope.

## 5. Verdict

After the repairs in this commit, **no unresolved P0 scientific contradiction
remains**. Every number taught by the course traces to a committed run record
with its correct scope, and every negative/pending verdict keeps its honest
status.
