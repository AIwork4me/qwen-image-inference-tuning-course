# P6 — Source Provenance Audit (Dirty-Worktree Gap of the Original P4 Campaign)

**Session:** final reproducibility & publication polish, 2026-11-06.
**Subject:** `runs/p4-fairab-225326/pipeline_profile/pipeline_profile.json`
(blob `7d100108…`, immutable) and the exact source that produced it.

## 1. The recorded gap

The run's own `environment.git` record (never rewritten):

- HEAD `4ec5905a`, branch `feat/world-class-bilingual-inference-course`,
  `dirty: true`, dirty files: untracked `docs/publication-closure/`,
  untracked `lab/m51_pipeline_fair_ab.py`, untracked run `p4-fairab-224922` (host-side trail),
  modified `lab/m51_pipeline_profile.py`.

Git HEAD therefore does **not** identify the executed bytes. This audit
establishes what is and is not recoverable, and hardens future runs.

## 2. Which files affected the measurements

| Class | Files | Identity at run time (22:53:27–23:18:49 UTC) |
|---|---|---|
| Experiment code, DIRTY | `lab/m51_pipeline_fair_ab.py` (untracked), `lab/m51_pipeline_profile.py` (modified) | not pinned by any run-time artifact — the subject of §3 |
| Experiment code, CLEAN (tracked) | `lab/m51_pipeline.py`, `lab/m51_e2e.py`, `lab/comfyui_client.py`, `lab/server.py`, `lab/ov_worker.py`, `lab/lab_env.py`, `lab/custom_nodes/linuxlab_cond_safetensors.py` | exactly commit `4ec5905a`'s blobs (absent from `dirty_files` — `git status` is complete evidence: a tracked file not listed there was byte-identical to HEAD) |
| Frozen config/model identity | prompts/seed/latent via the modules above; OV artifacts `/opt/m51-artifacts/bf16` (manifest digest `62eee985…`, size 13,892,143,718 B, byte-verified again this session) | pinned |

## 3. Forensics on the two dirty modules

Surviving artifacts examined (all digests below computed this session):

1. **Worktree copies** — NOT original bytes: branch switches during the
   closure (PR#3 → PR#2 → merge) deleted and re-materialized both files; their
   mtimes (23:33:38) post-date the run. Their content equals HEAD `6503e8d`'s
   blobs (`git diff` empty) — i.e. the *committed* bytes, not provably the
   runtime bytes.
2. **Bytecode cache** `lab/__pycache__/*.cpython-312.pyc` (written 23:23,
   pre-dating the first commit `cc05284` at ~23:30): embedded source mtime
   **23:23:18** and source sizes **17216 / 23060 bytes** — sizes exactly equal
   to the committed files' sizes. Note: 23:23:18 is ~4.5 min *after* run end
   (23:18:49): an **unexplained write event** touched both modules after the
   run and before the first commit; the run-time pyc it invalidated was
   overwritten, so no direct run-time byte artifact survives.
3. **Bytecode identity test**: the pre-commit pyc's code objects compare
   **structurally identical** (recursively: `co_code`, `co_consts`,
   `co_names`, `co_varnames`, `co_flags`, `co_firstlineno` — every function)
   to the committed sources' compilation. (The only marshalled-stream
   differences are CPython intern-flag artifacts for synthetic names like
   `<genexpr>`, proven non-semantic by object-level comparison.)

**Chain of evidence.** [source @ 23:23:18] == [committed source] to within a
same-size, same-line-number, same-token change (i.e., only a comment/whitespace
edit could differ); [source @ run time] → [source @ 23:23:18] is **not
independently provable** because of the unexplained 23:23:18 write event.

**Classification (mission taxonomy): `SOURCE_RECONSTRUCTED_WITH_EVIDENCE`.**

- Recoverable and verified: committed bytes ≈ pre-commit bytes (pyc
  structural identity + size identity); clean-tracked dependencies = `4ec5905a`
  exactly; artifact identity pinned.
- Not recoverable: a direct byte-level artifact of the two dirty modules as
  they existed at 22:53–23:18. The session's own engineering log claims no
  post-run edits, but claim ≠ artifact — hence RECONSTRUCTED, not EXACT.
- Could a difference influence results? Only if the 23:23:18 write changed
  executable behavior *after* the measurements — excluded at the
  bytecode-structure level for any edit that preserves size and line numbers
  (comments/whitespace only); any semantic difference would appear in the pyc
  comparison, and none does. Residual risk: a semantic edit between run end
  and 23:23:18 followed by a byte-restoring revert — no evidence, unfalsifiable
  from artifacts, and precisely what the clean-HEAD regression (§5) makes
  irrelevant.

## 4. Could the difference change the science?

No mechanism identified: (a) the executed logic is structurally the committed
logic (§3.3); (b) the committed logic was itself CPU-tested (AST scope,
preregistration strings) and is the code the reproduction guide teaches;
(c) the clean-HEAD physical regression below re-measures the four cells from
a fully verified source state. The original campaign keeps its
`SOURCE_RECONSTRUCTED_WITH_EVIDENCE` label permanently; nothing in this audit
upgrades it.

## 5. Future-run hardening (implemented this pass)

New module [`lab/source_provenance.py`](../../lab/source_provenance.py),
wired into `lab.m51_pipeline_fair_ab.main()` **before anything executes**:

- git head/branch/dirty/dirty-files + `tracked_diff_sha256` (digest of the
  full tracked diff — pins every tracked modification);
- per-module SHA256 + size + status for the nine modules that materially
  affect the experiment (`EXPERIMENT_MODULES`), each classified
  `COMMITTED_CLEAN / TRACKED_MODIFIED / UNTRACKED / MISSING`;
- **exact-byte snapshots** of any dirty/untracked module into
  `<run>/pipeline_profile/source_snapshot/` (≤ 512 KiB each — bytes preserved,
  not just digests);
- `provenance_class` computed fail-closed: `EXACT_SOURCE_VERIFIED` only for a
  fully clean worktree with all modules committed; dirty/untracked bytes
  preserved → `EXACT_SOURCE_VERIFIED_WITH_SNAPSHOT`; otherwise
  `SOURCE_DIRTY_DIGESTS_ONLY`. A run with untracked/modified source is never
  labeled EXACT unless the corresponding bytes are preserved;
- `argv`, dependency probes (python/openvino/torch for both venvs), cgroup
  limits, artifact-root + bf16 manifest (pre-existing), recorded UTC start.
- Never reads `os.environ` (no secret capture; test-enforced).

**Tests** (`tests/test_source_provenance.py`, CPU-only, 8 cases): clean
worktree → EXACT_SOURCE_VERIFIED; dirty tracked file detected with
worktree-digest (not HEAD-digest) recorded; untracked module snapshot
byte-equality; untracked without snapshot → DIGESTS_ONLY; MISSING reported,
never invented; deterministic manifest (order-stable, timestamp excluded from
the equality check); planted secret token absent from the record; the real
repo's `EXPERIMENT_MODULES` all resolve.

## 6. Clean-HEAD physical regression

Per this audit's finding, a bounded clean-HEAD rerun was executed from a
fully committed, verified source state — see
[`P6_CLEAN_HEAD_PHYSICAL_VALIDATION.md`](P6_CLEAN_HEAD_PHYSICAL_VALIDATION.md).
That run carries `EXACT_SOURCE_VERIFIED` provenance by construction. It does
not retroactively change the original campaign's classification; both
datasets remain separately labeled.
