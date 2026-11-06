# M5.1 Conditioning Alignment

**Artifact:** `runs/m51-20261105-cleanroom-01/alignment/conditioning_alignment.json`
**Gate policy:** fixed before E2E quality runs. FP32 rel L2 ≤ `1e-4` / cos ≥ `0.99999`; BF16 ≤ `2e-2` / cos ≥ `0.999`; INT8 ≤ `8e-2` / cos ≥ `0.995`. Long-sequence stress (>180 tokens) is descriptive, not a product gate.

## 1. Contract verified

- Tokenizer: ComfyUI `QwenImage21Tokenizer` (T2I system prefix, `thinking=True`, no think block).
- Token IDs: exactly equal across G, C0, C1a, C1b, C2 for all four prompts (66/51/78/22 full tokens).
- Trimming: keep from the **second** `<|im_start|>` (151644), matching `QwenImage21TEModel.encode_token_weights`.
- Output: layer-36 residual stream **before final RMSNorm**, `[1, seq_kept, 4096]`, FP32, finite.
- Not used: logits, post-RMSNorm features, generic chat template, arbitrary layers.

## 2. Positive and negative prompt alignment

| Comparison | p1 (52 kept) | p2 (37) | p3 (64) | negative (8) | Gate |
|---|---:|---:|---:|---:|---|
| G vs C0 | relL2 7e−6 / cos 1.0 | 7e−6 / 1.0 | 7e−6 / 1.0 | 3.1e−5 / 1.0 | reference |
| **C1a vs C0** | **3.5e−6 / 1.0** | **3.4e−6 / 1.0** | **3.0e−6 / 1.0** | **5.4e−6 / 1.0** | **PASS** |
| **C1b* vs C0** | **1.25% / 0.999922** | **1.15% / 0.999934** | **1.09% / 0.999941** | **1.91% / 0.999817** | **PASS** |
| C1b-uncompressed vs C0 | 1.32% / 0.999913 | 1.24% / 0.999923 | 1.16% / 0.999933 | **6.63% / 0.997797** | **FAIL negative** |
| **C2 vs C1a** | 6.38% / 0.997963 | 6.36% / 0.997975 | 5.62% / 0.998428 | **28.25% / 0.960822** | **FAIL negative** |

\* C1b = FP16-stored deployment IR + BF16 execution. The uncompressed variant is the FP32-stored IR with BF16 execution; it is retained as a precision reference, not the production artifact.

Raw metrics (max abs, mean abs, per-token cosine, determinism) are in the JSON. Example C1b p1: max abs 14.82, mean abs 0.0848, minimum per-token cosine 0.999479. C1a p1: max abs 0.00458, mean abs 2.30e−5.

## 3. Determinism

Within each arm, repeated runs of the same frozen token sequence produced **identical SHA256 hashes** of the conditioning tensor. GPU PyTorch is bitwise repeatable on this stack. Cross-arm differences are therefore numerical, not run-to-run noise.

## 4. Long-sequence stress (known Zen4 INT8 hazard)

Clean FP32, BF16 and INT8 IRs all encoded **180, 194, 230, 244, 408, 422 and 526 kept tokens** without the historical upstream `language_model/aten::add/Add_3` shape-inference failure. **No runtime graph patch was applied.** The failing upstream artifact was not reused; it is a different serialized IR and lineage.

| Case | Kept tokens | BF16 rel L2 vs FP32 | INT8 rel L2 vs FP32 |
|---|---:|---:|---:|
| en_long | 180 | 2.18% | 7.53% |
| zh_long | 230 | 3.19% | 11.29% |
| en_xlong | 408 | 7.23% | 14.61% |
| zh_xlong | 526 | 10.79% | 18.40% |

Precision error grows with sequence length; this is a **stress characterization**, not a claim that 526-token conditioning is production-equivalent.

## 5. Failure classification

- C2 negative path: **quantization error** (not tokenization, extraction, export or runtime).
- C1b-uncompressed negative path: **BF16 execution error amplified on an 8-token empty prompt**; the FP16-stored deployment artifact does not show it.
- No tokenization mismatch, wrong-layer extraction, checkpoint mismatch or export error was found.
- The original failed export attempt (uninitialized RoPE `inv_freq` after `meta`/`to_empty`) was an **export bug**, caught by the gate, fixed by constructing the model on CPU, and is documented in `M51_LIMITATIONS.md`.
