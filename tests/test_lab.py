"""CPU-only guards for the tuning lab (no GPU needed — mirrors CI).

Covers: manifest validation, record schema conformance, kernel classifier,
workflow builders, report aggregation, weight pins, frozen prompts, CLI.
GPU validation is a separate, physical-GPU activity (docs/REPRODUCTION_GUIDE.md).
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lab import lab_env, manifests, weights
from lab.comfyui_client import build_workflow
from lab.core import load_schema, plan_runs, steady_step_rate, validate_record
from lab.report import summarize_experiment
from lab.runner import build_record
from lab.workflows import build_workflow_manual_sigmas

FROZEN_SEEDS = [42, 1337, 20261030]


def test_frozen_protocol_prompts():
    d = json.loads((lab_env.LAB_DIR / "prompts.json").read_text(encoding="utf-8"))
    assert d["seeds"] == FROZEN_SEEDS
    ids = [p["id"] for p in d["prompts"]]
    assert ids == ["p1-zh-scene", "p2-en-scene", "p3-poster-text"]


def test_plan_runs_cache_defeating_seeds():
    specs = plan_runs(["p1"], [(1024, 1024)], FROZEN_SEEDS, warmup=1, measured=3)
    kinds = [s["kind"] for s in specs]
    assert kinds == ["cold", "warmup", "measure", "measure", "measure"]
    assert specs[0]["seed"] == FROZEN_SEEDS[0] + 1_000_000
    assert specs[1]["seed"] == FROZEN_SEEDS[0] + 2_000_000
    assert [s["seed"] for s in specs[2:]] == FROZEN_SEEDS


def test_steady_step_rate_excludes_first_step():
    assert steady_step_rate([9.9, 1.0, 1.1, 1.2]) == pytest.approx(1.1)
    assert steady_step_rate([9.9]) is None


def test_all_lab_manifests_load_and_validate():
    ids = manifests.list_manifests()
    assert set(ids) == set(manifests.KNOWN_IDS)
    for mid in ids:
        m = manifests.load_manifest(mid)
        assert m["id"] == mid
        assert m["framework"] == "comfyui"
        assert m["cfg"] == 1.0 and m["sampler"] == "euler"
        assert m["clip"] == "qwen3vl_8b_bf16.safetensors"
        assert m["vae"] == "qwen_image_2.1_vae_bf16.safetensors"


def test_manifest_rejects_unknown_id(tmp_path):
    saved = manifests.EXPERIMENTS_DIR
    try:
        manifests.EXPERIMENTS_DIR = tmp_path
        (tmp_path / "XX9.json").write_text(json.dumps({
            "id": "XX9", "category": "other", "description": "x",
            "framework": "comfyui", "precision": "bf16", "resolution": 1024,
            "steps": 1, "cfg": 1.0, "sampler": "euler", "scheduler": "simple",
            "unet": "u", "clip": "c", "vae": "v", "weight_dtype": "default",
            "tiled_vae": False, "attention_backend": "x", "measured": 1}))
        with pytest.raises(ValueError, match="methodology matrix"):
            manifests.load_manifest("XX9")
    finally:
        manifests.EXPERIMENTS_DIR = saved


def test_b0_is_the_unoptimized_baseline():
    m = manifests.load_manifest("B0")
    assert m["sdp_mode"] == "none"
    assert m["steps"] == 40
    assert m["unet"] == "qwen_image_2.1_bf16.safetensors"
    assert m["server_flags"] == []


def test_b0_math_forcing_rejected():
    m = dict(manifests.load_manifest("B0"))
    m["sdp_mode"] = "math"
    with pytest.raises(ValueError, match="normal Linux attention"):
        manifests.validate_manifest(m)


def test_l1o_carries_official_sigmas():
    m = manifests.load_manifest("L1O")
    assert m["scheduler"] == "manual_sigmas"
    sigmas = [float(x) for x in m["sigmas"].split(",")]
    assert len(sigmas) == 9  # 9 sigma values = 8 steps
    assert sigmas[0] == 1.0 and sigmas[-1] == 0.0


def test_workflow_builders_shapes():
    wf = build_workflow(unet="u.safetensors", clip="c.safetensors",
                        vae="v.safetensors", prompt="p", negative="",
                        width=1024, height=1024, steps=8, seed=42,
                        sampler="euler", scheduler="simple")
    classes = sorted({n["class_type"] for n in wf.values()})
    assert classes == ["CLIPLoader", "CLIPTextEncode", "EmptyLatentImage",
                       "KSampler", "SaveImage", "UNETLoader", "VAEDecode",
                       "VAELoader"]
    wf2 = build_workflow_manual_sigmas(
        unet="u.safetensors", clip="c.safetensors", vae="v.safetensors",
        prompt="p", negative="", width=1024, height=1024, seed=42,
        sigmas="1.0, 0.5, 0")
    classes2 = sorted({n["class_type"] for n in wf2.values()})
    assert "ManualSigmas" in classes2 and "SamplerCustomAdvanced" in classes2
    assert "KSampler" not in classes2


def test_build_record_schema_valid():
    m = manifests.load_manifest("B0")
    summary = {
        "text_encode_s": 0.5, "vae_decode_s": 0.9, "ksampler_s": 53.9,
        "step_times_s": [1.5] + [1.35] * 39, "node_timings_s": {}, "cached_nodes": [],
        "vram_peak_mb": 33000.0, "vram_residency_mb": 32000.0,
        "monitor_error": None, "vram_device_peak_mb": 34000.0,
        "gpu_util_pct_p50": 99.0, "gpu_util_pct_max": 100.0,
    }
    rec = build_record(m, {"kind": "measure", "prompt_id": "p1-zh-scene",
                           "width": 1024, "height": 1024, "seed": 42, "run_idx": 0},
                       total_s=55.5, execution_time_s=55.0, summary=summary,
                       env_ref="snapshot-test.json", run_id="test-B0",
                       server_boot_s=None, attention_evidence={})
    validate_record(rec)  # raises on violation
    assert rec["steady_step_median_s"] == 1.35  # step[0] excluded
    assert rec["runs"][0]["kind"] == "warm"


def test_summarize_experiment():
    m = manifests.load_manifest("B0")
    summary = {
        "text_encode_s": 0.5, "vae_decode_s": 0.9, "ksampler_s": 53.9,
        "step_times_s": [1.5] + [1.35] * 39, "node_timings_s": {}, "cached_nodes": [],
        "vram_peak_mb": 33000.0, "vram_residency_mb": 32000.0, "monitor_error": None,
    }
    recs = []
    for i, kind in enumerate(["cold", "measure", "measure", "measure", "fresh_prompt"]):
        recs.append(build_record(
            m, {"kind": kind, "prompt_id": "p1-zh-scene", "width": 1024,
                "height": 1024, "seed": 42, "run_idx": i},
            total_s=75.0 if kind == "cold" else (57.0 if kind == "fresh_prompt" else 55.5),
            execution_time_s=55.0, summary=summary,
            env_ref="s.json", run_id="t", server_boot_s=None,
            attention_evidence={}))
    s = summarize_experiment(recs)
    assert s["experiment"] == "B0"
    assert s["n_measure"] == 3
    assert s["warm_total_median_s"] == 55.5
    assert s["fresh_total_median_s"] == 57.0
    assert s["cold_total_s"] == 75.0


def test_kernel_classifier():
    from lab.profile import _classify_kernel as c
    assert c("void hipblasLt::gemm_tune_xdlops") == "gemm"
    assert c("Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT96x96x32") == "gemm"
    assert c("void fmha_fwd_kernel_mfma") == "attention"
    assert c("at::native::vectorized_elementwise_kernel") == "elementwise/activation"
    assert c("at::native::layer_norm_kernel") == "norm"
    assert c("some_unknown_kernel") == "other"


def test_weight_pins_complete():
    # every file the experiment manifests reference must have a pinned digest
    needed = set()
    for mid in manifests.list_manifests():
        if mid in ("H1", "H2"):
            continue
        m = manifests.load_manifest(mid)
        needed.add(f"diffusion_models/{m['unet']}")
        needed.add(f"text_encoders/{m['clip']}")
        needed.add(f"vae/{m['vae']}")
    assert needed <= set(weights.WEIGHTS), sorted(needed - set(weights.WEIGHTS))
    for rel, (size, digest) in weights.WEIGHTS.items():
        assert size > 0 and len(digest) == 64


def test_evidence_records_schema_and_integrity():
    """Every shipped record validates against the schema (claim integrity)."""
    root = lab_env.RESULTS_ROOT
    if not root.is_dir():
        pytest.skip("evidence not present")
    n = 0
    for f in root.rglob("*.json"):
        if f.name.startswith(("server", "H1_", "H2_", "STATUS", "client",
                              "noise", "doctor", "SUMMARY")):
            continue
        try:
            r = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(r, dict) and "experiment" in r and "runs" in r:
            validate_record(r)
            n += 1
    assert n >= 100, f"expected the shipped evidence set, found {n} records"


def test_shipped_summary_matches_records():
    """SUMMARY.json (the chart source) must be rebuildable from records."""
    root = lab_env.RESULTS_ROOT
    summary_path = root / "SUMMARY.json"
    if not summary_path.exists():
        pytest.skip("evidence not present")
    from lab.report import _load_records, summarize_experiment

    by_exp: dict[str, list[dict]] = {}
    for r in _load_records():
        by_exp.setdefault(r["experiment"], []).append(r)
    shipped = json.loads(summary_path.read_text(encoding="utf-8"))
    shipped_warm = {e["experiment"]: e["warm_total_median_s"]
                    for e in shipped["per_experiment"]}
    for exp, recs in by_exp.items():
        fresh = summarize_experiment(recs)
        if exp in shipped_warm and shipped_warm[exp] is not None:
            # warm medians agree within 0.05s (shipping rounded r1+r2 sets)
            assert abs(fresh["warm_total_median_s"] - shipped_warm[exp]) < 0.05, exp


def test_cli_importable():
    from lab.__main__ import main
    with pytest.raises(SystemExit):
        main(["--help"])


def test_lab_paths_are_course_layout():
    assert lab_env.RESULTS_ROOT.name == "results"
    assert lab_env.RESULTS_ROOT.parent.name == "evidence"
    assert lab_env.QUALITY_ROOT.name == "quality"
    assert lab_env.LAB_DIR.name == "lab"
    assert (lab_env.LAB_DIR / "experiments").is_dir()
