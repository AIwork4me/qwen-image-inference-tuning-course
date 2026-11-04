"""Tuning Lab — a teaching package for inference-acceleration engineering.

A runnable laboratory for the course "Making Generative AI Faster", built
around a real case study: Qwen-Image-2.1 image generation on a single AMD
Radeon PRO W7900 (gfx1100, 48 GB) under Linux ROCm.

Methodology encoded in the CLI (``python -m lab ...``):

    doctor    -> environment preflight (fail-closed)
    baseline  -> frozen B0 baseline (measure before optimizing)
    profile   -> pipeline + rocprofv3 kernel profiling
    run       -> one experiment manifest (L1 F1 F2 LF* H1 H2 BEST ...)
    report    -> aggregate records into SUMMARY + charts

Derived from AIwork4me/qwen-image-rocm (MIT) — the Windows/7900XTX parent
project of this Linux case study. All shipped evidence was measured on the
physical W7900 on 2026-11-03; see evidence/ and docs/CLAIM_EVIDENCE_MATRIX.md.
"""
