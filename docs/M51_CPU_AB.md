# M5.1 Same-Machine CPU A/B

**Artifact:** `runs/m51-20261105-cleanroom-01/performance/encoder_ab.json`
**Primary ratio:** `Median(C0 warm) / Median(C1 warm)` on the **same CPU**, same frozen tokens, same checkpoint lineage, 16-thread budget.

## 1. Controls

- Host: EPYC 9334, 16-CPU cgroup quota, 55 GiB memory cap, no CPU governor exposed (`no-cpufreq`).
- Threads: PyTorch `torch.set_num_threads(16)`, interop 1; OpenVINO `INFERENCE_NUM_THREADS=16`, `NUM_STREAMS=1`.
- Affinity/NUMA: no explicit pinning; both processes scheduled by the container. Recorded as a limitation.
- Prompts: frozen `p1-zh-scene`, `p2-en-scene`, `p3-poster-text`, empty negative, identical token IDs.
- Warm definition: repeated encode of the same frozen sequence in the same process.
- Cold definition: model load + compile/init + first conditioning tensor.
- Sample size: **C0 n=1 warm per prompt** (the second identical encode; first p1 encode is the process warm-up), **C1 n=2 warm per prompt**. C0 run-to-run spread between the two identical encodes was **<0.3%**, but the formal uncertainty is larger than C1's; this is an explicit reduced-sample baseline.

## 2. Results

| Arm | Load/compile | Cold→first | p1 | p2 | p3 | neg | RSS peak |
|---|---:|---:|---:|---:|---:|---:|---:|
| C0 PyTorch CPU | 3.85 s | 179.4 s | 176.02 s | 136.81 s | 207.45 s | 61.51 s | 34.5 GB |
| C1a OV FP32 | 89.2 s (84.6 compile) | 95.0 s | 0.779 s | 0.648 s | 0.875 s | 0.398 s | 50.5 GB |
| **C1b OV BF16*** | **3.82 s** (1.2 compile) | **5.5 s** | **0.532 s** | **0.285 s** | **0.555 s** | **0.187 s** | **30.0 GB** |
| C2 OV INT8 | 5.04 s (1.6 compile) | 7.2 s | 0.361 s | 0.249 s | 0.406 s | 0.144 s | 16.8 GB |

\* FP16-stored deployment IR, BF16 execution hint.

| Speedup vs C0 | p1 | p2 | p3 | negative |
|---|---:|---:|---:|---:|
| C1a OV FP32 | 226.0× | 211.0× | 237.2× | 154.6× |
| **C1b OV BF16** | **330.9×** | **480.4×** | **373.5×** | **328.4×** |
| C2 OV INT8 | 487.5× | 549.4× | 510.5× | 426.9× |

These ratios are against the **course's ComfyUI PyTorch CPU path** (FP32 manual-cast compute with BF16-stored weights), not a generic tuned PyTorch BF16 GEMM.

## 3. Thread sweep (OpenVINO BF16, same model)

Two warm encodes per prompt; min reported.

| Threads | p1 (66 tok) | p3 (78 tok) |
|---:|---:|---:|
| 4 | 1.587 s | 1.692 s |
| 8 | 0.575 s | 0.660 s |
| **16** | **0.514 s** | **0.562 s** |
| 32 | 0.644 s | 0.768 s |

16 threads is the best-tuned point under the 16-CPU quota; 32 oversubscribes and regresses. This is a **matched-resource** and **best-tuned** comparison: the OpenVINO arm is not given more CPU than C0.

## 4. Memory interpretation

- C1a FP32 keeps 26 GB of FP32 weights resident and peaks at **50.5 GB**, leaving little room to co-exist with a GPU process under this 55 GiB cap.
- C1b's FP16-stored IR cuts runtime RSS to **30.0 GB** with essentially the same measured error (1.09–1.25% positive).
- C2 is the memory winner at **16.8 GB** but has the largest conditioning error, especially on the negative path.
- The C0 process shows 34.5 GB RSS for 17 GB BF16 weights because of the manual-cast/allocator working set.

## 5. Long-sequence behavior

All clean IRs ran to 526 kept tokens. Warm timings:

| Arm | en_long 180 | zh_long 230 | en_xlong 408 | zh_xlong 526 |
|---|---:|---:|---:|---:|
| C1b BF16 | 1.271 s | 2.554 s | 2.168 s | 2.686 s |
| C2 INT8 | 1.080 s | 1.850 s | 1.892 s | 2.545 s |
| C1a FP32 | 1.958 s | 5.188 s | 4.039 s | 5.209 s |

These are one encode per case in a fresh process per arm; the first case includes some kernel warm-up. Exact JSON: `ov/longseq/longseq_*.json`.

## 6. What this does and does not prove

- Proves: on this CPU, the same checkpoint and prompts, OpenVINO CPU is 155–549× faster across the measured C1/C2 arms (C1b 328–480×), with BF16 deployment conditioning error ≤1.25% on positive prompts.
- Does not prove: a universal AMD CPU speedup, a tuned PyTorch BF16 comparison, or that encoder speed translates into image-generation speed. See `M51_E2E_AND_PIPELINE.md`.
