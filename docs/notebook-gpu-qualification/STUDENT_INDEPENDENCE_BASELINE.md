# Student Independence Baseline (P0 freeze)

Recorded BEFORE any source modification in the student-independence closure mission.

## Git / PR state
- PR: https://github.com/AIwork4me/qwen-image-inference-tuning-course/pull/4
- Branch: `feat/bilingual-jupyter-optimization-labs` (base `main`)
- HEAD at freeze: `9e550e8b9b80a13d01adadd8256c29a5c86a5bc7` ("docs(validation): reference only tracked evidence paths (CI link gate)")
- PR state: OPEN, not draft, MERGEABLE; CI run 42222858351 = SUCCESS (175 CPU tests; 18-notebook REFERENCE execution)
- Dev worktree: /workspace/rehearsal/qwen-image-inference-tuning-course (ff-synced to 9e550e8; tracked tree clean; only untracked venvs/weights/prior run outputs remain)
- Other worktree /workspace/m51/qwen-image-inference-tuning-course is on the older branch feat/world-class-bilingual-inference-course and is NOT used for this closure.

## Frozen blob identities at 9e550e8 (git rev-parse HEAD:<path>)
- `6219dd60f66bd0a5d3e9f89654328000aef0aebe` lab/notebook_runtime.py
- `3ca42245b1173a9bdf5ad9b6477c77cf09541e99` lab/notebook_support.py
- `25e0a46cf40d425548bbc38a0417f759910ddb05` lab/m51_contract.py
- `abff3bb2b6bb230ffd25daa413fb7b6753558898` lab/m51_pipeline_fair_ab.py
- `3ace1fd6c78cd813d17677de6bf1aadb412511bd` lab/m51_pipeline.py
- `088fa26a1328fe2d32fb048f765f8ab889d61fe6` lab/m51_run.py
- `d316a4a3e0fa511dbfe4207b684661c85cc4a625` lab/runner.py
- `86f2441e9ded3562226859eeeaa4f59e183da970` notebooks/05_cpu_offload_negative.zh-CN.ipynb
- `4339c66b062187c7db1ddc8a4621cfff3ff594d6` notebooks/05_cpu_offload_negative.ipynb
- `b44c1d43d94dd244d97b4495af98fd4edb3bd3f9` notebooks/06_openvino_and_fair_ab.zh-CN.ipynb
- `68ef35660fac5aab0d1558a7b00e2f9d7955c391` notebooks/06_openvino_and_fair_ab.ipynb
- `fb1d7e011f871d6af64c68d9cd77ccf2a603de0e` scripts/notebook-qualification/run_all.py
- `0ef3b722a8bf21000999c80bd81c03cd0f8b5d8f` scripts/notebook-qualification/validate_evidence.py
- `3e8cea7e30a9fe211ba92b271116839e208297b9` scripts/notebook-qualification/build_notebooks.py
- `488fe5aff3f9f763ae70cd1f627ead2967e2ef73` tests/test_notebook_gpu_protocol.py
- `352845346790f5fccbdda040e0c893c5d4358be0` tests/test_notebooks.py
- `edecbb2c8abb7004c5f09e6c58816ce895721e5f` tests/test_m51_contract.py
- `d3e4e29969aa43e325e7fdfa445b7382f4c57cb5` docs/notebook-gpu-qualification/FINAL_RELEASE_GATE_MATRIX.md

## Tracked runs/ inventory
- 439 tracked files under runs/ (81 top-level run directories).
- Frozen instructor campaign records relied on by notebooks: `m51-20261105-cleanroom-01`, `p4-fairab-225326`, `p4-fairab-cleanhead-003750`, `q1-075553-*` (B0/L1/F1/F2/BEST/H1-cpu-te/H2-pipeline/H3_OV_BF16/m51-te-cpu/m51-te-cuda/m51pipe/profile), `q1-fairab-campaign`, `q1-ovfp32-*`, `q1-ovint8-*`.
- Qualification campaign: `runs/notebook-gpu-qualification/w7900-20261106-073429`.
- evidence/ frozen tree untouched (validate_evidence enforces clean git status).

## GPU campaign identities
- Physical campaigns on record: w7900-20261106-073429 (18-notebook GPU_CLASSROOM qualification), q1-075553-* (formal A/B), p4-fairab-* (clean-HEAD 2x2 factorial), m51-20261105-cleanroom-01 (M5.0/M5.1 cleanroom references).

## Hardware / environment at freeze (this workstation)
- GPU: gfx1100 (Radeon Cloud W7900 class), single device, VRAM total 51522830336 B (~48 GiB), idle at freeze.
- cgroup memory.max = 59055800320 B (~55 GiB); cpu.max = 1600000/100000 (16 CPU quota); 128 host cores; ROCm present.
- OpenVINO artifacts root /opt/m51-artifacts with bf16/fp32/int8 subtrees (shared, provisioned; not repo-tracked).
- openvino python only inside repo .venv-ov (2026.4.1); torch 2.9.1+rocm inside .venv-lab.

## Known student-independence gaps to close (from P0 audit)
1. `Session` has no unique session id; notebook run ids use minute-resolution HHMMSS tags (collision risk).
2. Notebook 05 cell 7 and Notebook 06 cells 5/7 select records via `sorted(glob(...))[-1]` (latest-file-wins) across student, shared-campaign, and committed run directories.
3. Notebook 06 artifact verification `ov_verify()` returns a subprocess return code that is never enforced; a BLOCKED verification cannot stop the notebook.
4. Notebook 06 never initiates its own A/C arms; `RUN_OWN_GPU_ARMS`/`want_own_arms()` are declared but unused.
5. No machine-validated provenance categories (CURRENT_SESSION_PHYSICAL / ... / BLOCKED) exist; only string stamps MEASURED_TODAY/HISTORICAL/ESTIMATE.

All source changes in this closure happen AFTER this freeze; `evidence/` and prior qualification records remain unmodified.
