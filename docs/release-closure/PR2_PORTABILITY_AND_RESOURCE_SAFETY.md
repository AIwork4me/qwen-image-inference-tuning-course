# PR #2 — Phase P3: Resource Safety and Portable Reproduction

## P1-A. Conservative RAM admission

**Before** (`lab/m51_run.py:48-52`): `free_ram = cgroup memory.max` — the 55 GiB *ceiling* was
treated as *available*, so with 38 GiB already consumed the gate still admitted a 45 GiB FP32
export.

**After** (`lab/m51_contract.py::conservative_ram_gate`, wired into `m51_run`):

- `available = min(host MemAvailable, cgroup_limit − cgroup_anon)` — cgroup `memory.current`
  is deliberately NOT treated as unavailable (it is mostly reclaimable page cache); the
  non-reclaimable `anon` from `memory.stat` is what constrains a new anonymous working set
  (explicit policy, documented in-code),
- `usable = available − safety_headroom (2 GiB) − concurrent_reserve` (e.g. a resident
  ComfyUI server; parameterized per experiment),
- unlimited cgroup (`memory.max == "max"`) falls back to host MemAvailable only,
- no memory information at all → BLOCKED (never guess).

**Tests**: finite-cgroup headroom math (`55−38−2 = 15 GiB` usable, 16 GiB need → BLOCKED),
unlimited-cgroup host fallback, min-of-both-bases, concurrent reserve, no-info BLOCKED,
reader shape (`test_ram_gate_*`, 6 cases).

**Physical check (audit host)**: cgroup current 59.0 GiB (page cache from hashing), anon
3.15 GiB → gate correctly reports 50.1 GiB usable and admits all arms; the old code would
have reported 55 GiB regardless of anon pressure.

## P1-B. Machine-specific paths removed

Single documented resolution policy — `$M51_OV_ARTIFACTS`, else `<repo>/runs/m51-ov-artifacts` —
now used consistently by:

| Consumer | Before | After |
|---|---|---|
| `lab/m51_run.py` (CLI H3) | env var honored already | `artifact_root()` helper (env read at call time; testable) |
| `lab/m51_pipeline.py` arms C/D | **hardcoded `/opt/m51-artifacts/{bf16,int8}`** | `artifact_root()` (shared helper) |
| `scripts/m51/provision_ov.sh` | env var honored; `TMPDIR` defaulted to `/workspace/tmp` | same env policy; portable scratch `TMPDIR=$REPO/.tmp` (mkdir'ed), still honors a preset TMPDIR |
| `lab/m51_env.py` | could silently overwrite the frozen `environment.json` on re-run | refuses without `--force` or a new `--run` dir (frozen-evidence protection) |
| `lab/m51_baseline.py` | `python lab/m51_baseline.py` crashed (`lab/profile.py` shadows stdlib `profile` → breaks `torch`/`cProfile` import chain) | lab/ filtered from `sys.path` exactly like `m51_pipeline.py`/`ov_encoder.py` do — discovered live during this audit's P4 rerun |

Verified on this host with a non-default root: `M51_OV_ARTIFACTS=/opt/m51-artifacts … --check-only`
verifies the /opt tree; unset, the CLI resolves `runs/m51-ov-artifacts` (both roots' bf16 bin
digests match the manifest pin `62eee985…`). The original cloud location remains valid via the
env var; nothing about the frozen campaign records was rewritten.

## P1-C. Provisioning, recovery, worker shutdown

`scripts/m51/provision_ov.sh` audit results:

- `--check-only` now performs **byte verification only** (no download, no conversion, no venv
  changes) and exits non-zero on any BLOCKED reason — verified live (38 s for 45.5 GB).
- Already-verified artifacts are reused; partial/corrupt artifacts are diagnosed by reason
  (`missing_artifacts`, `digest_mismatch`, `size_mismatch`, `identity_mismatch`,
  `corrupt_manifest`, `unsafe_path`).
- Disk budgeting uses `df` on the **artifact volume** (`$ART`), not `/` or `/tmp` — correct
  target filesystem.
- Small `/tmp` (4 GB tmpfs here): pip scratch defaults to `$REPO/.tmp`; per-file hashing is
  streaming (no tmp usage).
- No global ROCm/PyTorch package is touched: OpenVINO lives in `.venv-ov` with a read-only
  `.pth` bridge; the working ROCm environment was not modified (verified: `/opt/venv`
  PyTorch 2.9.1+gitff65f5b intact).
- The OpenVINO environment stays optional; M0–M4 startup path (`scripts/cloud/start.sh`,
  `.venv-lab`) is untouched by these changes (full CPU suite incl. M0–M4 tests passes).
- Interruption: exports write per-directory; `ov_compress`/`ov_quantize` write the manifest
  last, so a partial dir is detected as `missing_manifest`/digest-mismatch on the next
  `--check-only`; no user files are deleted anywhere.
- Worker lifecycle (`lab/ov_worker.py`, `lab/m51_pipeline.py::WorkerClient`): quit-line →
  15 s grace → kill; requests time out (1200 s) without orphaning readline threads (process
  kill reaps them); conditioning files are written atomically (`.tmp<pid>` + `os.replace`),
  so a killed worker cannot leave a partial file a later run mistakes for fresh output.
  Run-scoped output dirs prevent cross-run staleness.

## Result

Resource admission is conservative and honest, paths are portable via one documented env var,
and provisioning/recovery behavior is verified on the physical host.
