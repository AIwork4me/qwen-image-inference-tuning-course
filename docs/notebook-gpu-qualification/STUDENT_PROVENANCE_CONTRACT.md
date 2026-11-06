# Student Experiment Provenance Contract

Status: ACTIVE (student-independence closure, PR #4)
Implementation: `lab/student_provenance.py` + the `Session` class in `lab/notebook_runtime.py`
Tests: `tests/test_student_provenance.py` (41 cases, CPU-only)

## 1. Why this contract exists

PR #4 ships real physical W7900 run records *inside the repository* (`runs/q1-075553-*`,
`runs/m51-20261105-cleanroom-01`, `runs/p4-fairab-*`, …). A fresh clone therefore
contains physical evidence before the student executes anything. Earlier notebook
code selected records with `sorted(glob(...))[-1]` ("newest file wins") and stamped
the result `MEASURED_TODAY` — an instructor's committed record could be displayed as
the student's own current measurement. That class of provenance failure is now
structurally impossible.

## 2. The eight origin categories (exact enums)

| Category | Meaning | Who may claim it |
|---|---|---|
| `CURRENT_SESSION_PHYSICAL` | This notebook session initiated **and awaited** the workload; the run dir carries a `PROVENANCE.json` binding to this session's `session_id`. | Only the session that executed it |
| `SAME_STUDENT_PRIOR_PHYSICAL` | An earlier session on this workspace, selected by exact run id; that session's record must still exist under `runs/notebook-sessions/` (or a campaign `sessions/`). | Same workspace, explicit id |
| `SHARED_CAMPAIGN_PHYSICAL` | A verified physical run from a declared campaign (`$QIR_NOTEBOOK_CAMPAIGN` or `runs/notebook-gpu-qualification/`) reused deliberately. | Anyone, explicitly |
| `COMMITTED_PHYSICAL_REFERENCE` | A physical result tracked in Git. Real hardware evidence; **never** newly measured by the current student. | Anyone, explicitly |
| `FROZEN_REFERENCE` | Official archived evidence under `evidence/`. | Anyone, explicitly |
| `SIMULATED` | Analytical estimate or scheduler simulation. | n/a |
| `NOT_RUN` | No record exists. | n/a |
| `BLOCKED` | Admission/verification refused; no measurement. | n/a |

## 3. Hard rules (fail closed)

1. **No latest-file-wins.** Notebooks 01–07 select records by explicit run id or
   explicit committed path. The remaining `latest_campaign_run` reuse path goes
   through `Session.consume()`, which resolves and prints the category.
2. **Run identity beats file mtime.** Classification reads bindings and Git —
   never timestamps.
3. **A Git-tracked run is never `CURRENT_SESSION_PHYSICAL`** (enforced in
   `classify_run`, tested).
4. **A current-session claim needs an execution-to-output binding.** `Session`
   writes `runs/<run_id>/PROVENANCE.json` (session id, kind, source head/dirty,
   timestamp) only *after* the physical CLI exits successfully.
5. **Missing provenance fails closed.** An unbound, untracked, non-campaign run
   cannot be consumed at all (`ProvenanceError`).
6. **Paths are constrained.** Run ids are single path components under `runs/`;
   traversal and symlink escapes are rejected.
7. **One physical run is one measurement.** English and Chinese notebooks may
   consume the same run by id, labelled `SAME_STUDENT_PRIOR_PHYSICAL` /
   `SHARED_CAMPAIGN_PHYSICAL`; the category system makes double-counting visible.
8. **No cross-source equivalence.** A/C pair records carry workload identity
   (requests, seed, prompt digest); mismatches fail validation.
9. **No env-only proof.** `$M51_OV_ARTIFACTS` resolves *where* artifacts live;
   it never certifies *that* an experiment ran. Only bindings do.
10. **No credentials or personal data** in provenance records (session ids are
    random hex; source identity is the Git head + dirty flag only).

## 4. Session identity

Every notebook session gets `session_id = stu-YYYYmmdd-HHMMSS-<6 hex>` created in
`Session.__init__` — before any admission or GPU work. `session.finish()` writes the
session record (now including `session_id`, `source.head`, `source.dirty`, and
per-execution `run_origin`) under `runs/notebook-sessions/` (or the campaign
`sessions/` root). These records are what make `SAME_STUDENT_PRIOR_PHYSICAL`
verifiable later.

Run tags are de-conflicted (`HHMMSS-nbq` → `HHMMSS-<hex>-nbq` on collision), so a
re-run never collides with append-only records.

## 5. What the student sees

Every consumed number is printed next to a pinned label:

```
[CONSUMED PHYSICAL RECORD] run_id=q1-075553-H3_OV_BF16 (…) explicit reference
[PROVENANCE] COMMITTED_PHYSICAL_REFERENCE | run_id=q1-075553-H3_OV_BF16 | committed physical reference tracked in Git (instructor campaign; NOT measured by you)
```

Executed runs print `[PROVENANCE] CURRENT_SESSION_PHYSICAL | run_id=… | measured on
THIS GPU in THIS session (physical)`. Reuse is never labelled as measured-today
(`_origin_stamp` maps categories to stamp text inside every notebook).

## 6. Machine-checkable result records (P5)

`Session`-closing cells in notebooks 05/06 build a student result record validated by
`prov.validate_result_record()` (fail closed on error). Fields: experiment id, session
id, run ids, mode, provenance category, execution status (`MEASURED|REUSED|SIMULATED|
NOT_RUN|BLOCKED|FAIL`), GPU identity, source head/dirty, workload identity,
populations, measurements, throughput (numerator/denominator/unit), speedup,
memory (CPU RSS, GPU VRAM), numerical status, human-quality status, classification
(`FORMAL_PROTOCOL_MEASURED|BOUNDED_PHYSICAL_COMPARISON|SMOKE_PIPELINE_CHECK|
REFERENCE_ONLY|SIMULATION_ONLY`), A/C pair identity, result status, source paths,
reproduction command.

Rejected (tested): reference data as current physical; committed instructor runs as
student-owned; smoke/bounded populations as FORMAL; missing source head on new
claims; A/C workload mismatch; duplicate run ids with conflicting metadata; invalid
enums; stale-session claims; `MEASURED` on non-current categories; numerical FAIL
with human PASS; M6 release PASS while human review PENDING; REFERENCE_ONLY records
carrying speedups.

## 7. Backward compatibility

Historical records (no bindings) remain fully readable as explicitly labelled
references: tracked ones classify `COMMITTED_PHYSICAL_REFERENCE`; frozen `evidence/`
classifies `FROZEN_REFERENCE`. Nothing about past records was rewritten.
