# Course-v2 Baseline Audit (P0)

Audit session: 2026-11-05/06, autonomous, on branch
`feat/world-class-bilingual-inference-course` (stacked on PR #2 head).
Auditor role: independent — this session authored none of the audited history.

## 1. Repository state at audit start

| Item | Value |
|---|---|
| Remote | `AIwork4me/qwen-image-inference-tuning-course` |
| `origin/main` | `767071af4c399c65732189ab77178d247a78ee81` |
| PR #2 head (`feat/m5-1-openvino-cpu-te-course`) | `a7fa69ae18800fc1f7a83e24de1c0b8bad2e7eaa` — matches the mission reference value |
| PR #2 state | OPEN, Draft, base `main`, not merged |
| New branch cut from | `a7fa69a` (PR #2 head, clean worktree) |
| Tracked files | 599 |

## 2. Evidence freeze (P0-A)

Before any edit:

- Git blob freeze: 579 blob-identities across `evidence/`, `runs/`, `docs/`,
  `lab/`, `course/` recorded in
  `runs/overnight-course-v2-a7fa69a/p0_blob_freeze.txt`.
- Working-tree SHA256 freeze: 254 files under `evidence/` recorded in
  `runs/overnight-course-v2-a7fa69a/p0_evidence_sha256.txt`.
- Constraint adopted: no command in this session may rewrite content under
  `evidence/` or the frozen `runs/m51-20261105-cleanroom-01/` measurement
  records. Additions (new files) are permitted; modifications are not.
- Pre-existing freeze cross-check: `docs/release-closure/PR2_BASELINE_FREEZE_MANIFEST.txt`
  (493 blob IDs + 138 raw SHA256 from the PR #2 audit) re-verified — no drift.

## 3. Test baseline

- `python -m pytest tests/ -q` at `a7fa69a`: **82 passed** (4.6 s, CPU-only) —
  matches the PR #2 release-gate matrix claim.
- CI workflow `.github/workflows/ci.yml` runs the same CPU suite; no GPU
  claims from CI.

## 4. Physical host

- GPU: gfx1100 (W7900-class, 48 GB) via ROCm, `/dev/kfd` present;
  `torch.cuda.is_available() == True`.
- CPU/RAM: EPYC 9334 host, cgroup 16 CPU / 55 GiB (the same class as the
  M5.1 campaign environment).
- Available for bounded P2 experimentation within the mission budget
  (≤3 GPU-hours incremental).

## 5. Documentation inventory read

Read in full during this audit: both READMEs, `SYLLABUS.md`,
`LECTURE_DEMO.md`, all M0–M7 lessons + M5.1 lesson, `CLAIM_EVIDENCE_MATRIX.md`,
`M51_*.md` (results, e2e, cpu_ab, limitations, claim matrix, reproduction),
all `docs/release-closure/*` (gate matrix, latency-vs-throughput JSON, human
review package, statistical audit excerpts), onboarding docs
(`RADEON_CLOUD_QUICKSTART`, `CLASSROOM_WORKFLOWS`, `REPRODUCTION_GUIDE`,
`CONTRIBUTING_TUTORIAL`, `TROUBLESHOOTING`).

## 6. Gate decision for P0

Findings that block the curriculum rewrite are tracked in
`SCIENTIFIC_CLAIMS_AUDIT.md` (metric drift) and `IMAGE_EVIDENCE_AUDIT.md`
(availability claims). Both are remediated in commit
`audit(course): freeze claims and reconcile quality evidence` **before** the
P1/P3 curriculum work begins, satisfying the mission gate:
*no unresolved P0 scientific contradiction before the curriculum rewrite*.
