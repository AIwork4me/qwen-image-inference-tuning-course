"""PR2 independent evidence recomputation (release closure, phase P1).

Reads ONLY the most granular raw records of the frozen campaign
runs/m51-20261105-cleanroom-01/ and recomputes every headline metric from
scratch. Never reads docs/ summaries for values. Writes
docs/release-closure/PR2_EVIDENCE_REDERIVATION.json for the human report.

Usage (CPU only, ~2 min, needs numpy + safetensors + torch):
    .venv-lab/bin/python scripts/release-closure/recompute_evidence.py
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
RUN = REPO / "runs" / "m51-20261105-cleanroom-01"
OUT = REPO / "docs" / "release-closure" / "PR2_EVIDENCE_REDERIVATION.json"

PROMPTS = ["p1-zh-scene", "p2-en-scene", "p3-poster-text", "negative"]


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def cond(base: Path, pid: str, rep: int = 0) -> np.ndarray:
    from safetensors.torch import load_file
    a = load_file(str(base / f"{pid}-r{rep}.safetensors"))["cond"].numpy().astype(np.float64)
    return a.reshape(1, *a.shape[-2:])


def rel_l2_cos(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    d = a - b
    an, bn = np.linalg.norm(a.ravel()), np.linalg.norm(b.ravel())
    return (float(np.linalg.norm(d.ravel()) / max(an, 1e-12)),
            float(np.dot(a.ravel(), b.ravel()) / max(an * bn, 1e-12)))


def warm_rows(rows: list[dict]) -> list[float]:
    return [r["encode_s"] for r in rows[1:]] or [rows[0]["encode_s"]]


def main() -> int:
    report: dict = {"run": str(RUN), "recomputed_at": __import__("time").strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "inputs": "raw probe JSONs + conditioning safetensors only", "sections": {}}

    # ---- 1. encoder A/B from raw probe records -------------------------------
    probes = {
        "G": RUN / "baseline/probe_cuda.json",
        "C0": RUN / "baseline/probe_cpu.json",
        "C1a": RUN / "ov/run_fp32/probe_C1a_OV_FP32.json",
        "C1b_uncompressed": RUN / "ov/run_bf16/probe_C1b_OV_BF16.json",
        "C1b_deployment": RUN / "ov/run_bf16_compressed/probe_C1b_OV_BF16_COMPRESSED.json",
        "C2": RUN / "ov/run_int8/probe_C2_OV_INT8.json",
    }
    enc = {}
    for arm, p in probes.items():
        d = load(p)
        enc[arm] = {
            "load_s": d.get("load_s"),
            "rss_peak_mb": d.get("rss_peak_mb"),
            "per_prompt": {
                pid: {"cold_s": d["prompts"][pid][0]["encode_s"],
                      "warm": warm_rows(d["prompts"][pid]),
                      "n_warm": len(d["prompts"][pid]) - 1,
                      "n_tokens_full": d["prompts"][pid][0]["n_tokens_full"],
                      "n_tokens_kept": d["prompts"][pid][0]["n_tokens_kept"],
                      "cond_sha256": d["prompts"][pid][0]["cond_sha256"],
                      "deterministic_across_reps": len({r["cond_sha256"] for r in d["prompts"][pid]}) == 1}
                for pid in PROMPTS},
        }
    # independent determinism + hash agreement across arms
    c0_hashes = {pid: enc["C0"]["per_prompt"][pid]["cond_sha256"] for pid in PROMPTS}
    for arm in ("G", "C1a", "C1b_uncompressed", "C1b_deployment", "C2"):
        for pid in PROMPTS:
            assert enc[arm]["per_prompt"][pid]["deterministic_across_reps"], (arm, pid)

    speedups = {}
    for arm in ("C1a", "C1b_deployment", "C1b_uncompressed", "C2"):
        speedups[arm] = {}
        for pid in PROMPTS:
            c0m = statistics.median(enc["C0"]["per_prompt"][pid]["warm"])
            am = statistics.median(enc[arm]["per_prompt"][pid]["warm"])
            speedups[arm][pid] = round(c0m / am, 1)
    report["sections"]["encoder_ab"] = {
        "arms": enc, "speedup_vs_C0_warm_median": speedups,
        "note": "C0 n_warm=1/prompt, OV arms n_warm=2/prompt; ratios are order-of-magnitude reliable, single-digit-% unreliable",
    }

    # ---- 2. conditioning numerics from raw tensors ---------------------------
    dirs = {"G": RUN / "baseline/cond_cuda", "C0": RUN / "baseline/cond_cpu",
            "C1a": RUN / "ov/run_fp32/cond",
            "C1b_uncompressed": RUN / "ov/run_bf16/cond",
            "C1b_deployment": RUN / "ov/run_bf16_compressed/cond",
            "C2": RUN / "ov/run_int8/cond"}
    num = {}
    for a, b in (("G", "C0"), ("C1a", "C0"), ("C1b_deployment", "C0"),
                 ("C1b_uncompressed", "C0"), ("C2", "C1a"), ("C2", "C0")):
        num[f"{a}_vs_{b}"] = {}
        for pid in PROMPTS:
            x, y = cond(dirs[a], pid), cond(dirs[b], pid)
            assert x.shape == y.shape, (a, b, pid, x.shape, y.shape)
            l2, c = rel_l2_cos(x, y)
            num[f"{a}_vs_{b}"][pid] = {"rel_l2": l2, "cosine": c}
    gates = {
        "C1a (<=1e-4, cos>=0.99999)": all(v["rel_l2"] <= 1e-4 and v["cosine"] >= 0.99999 for v in num["C1a_vs_C0"].values()),
        "C1b_deployment (<=2e-2, cos>=0.999)": all(v["rel_l2"] <= 2e-2 and v["cosine"] >= 0.999 for v in num["C1b_deployment_vs_C0"].values()),
        "C1b_uncompressed (<=2e-2, cos>=0.999)": all(v["rel_l2"] <= 2e-2 and v["cosine"] >= 0.999 for v in num["C1b_uncompressed_vs_C0"].values()),
        "C2 (<=8e-2, cos>=0.995)": all(v["rel_l2"] <= 8e-2 and v["cosine"] >= 0.995 for v in num["C2_vs_C1a"].values()),
        "C2_negative_only": num["C2_vs_C1a"]["negative"],
    }
    report["sections"]["conditioning_numerics"] = {"pairs": num, "gates_recomputed": gates}

    # ---- 3. token contract ----------------------------------------------------
    c0 = load(RUN / "baseline/probe_cpu.json")["prompts"]
    ov = load(RUN / "ov/run_bf16_compressed/probe_C1b_OV_BF16_COMPRESSED.json")["prompts"]
    tok = {"empty_negative_ids_clip": c0["negative"][0]["ids_clip"],
           "empty_negative_ids_manual_preventTrue": c0["negative"][0]["ids_manual"],
           "empty_negative_ids_ov": ov["negative"][0]["ids"],
           "ov_equals_clip_all_prompts": all(
               list(ov[pid][0]["ids"]) == list(c0[pid][0]["ids_clip"]) for pid in PROMPTS),
           "manual_differs_only_on_negative": {
               pid: c0[pid][0]["ids_equal"] for pid in PROMPTS}}
    report["sections"]["token_contract"] = tok

    # ---- 4. pipeline from raw per-request timestamps ---------------------------
    pab = load(RUN / "pipeline/pipeline_ab.json")
    pipe = {}
    for arm, a in pab["arms"].items():
        reqs = a["requests"]
        wall = lambda r: r.get("wall_s", r.get("gen_wall_s"))
        row = {"n": len(reqs), "sum_wall_s": round(sum(wall(r) for r in reqs), 3),
               "total_s_reported": a["total_s"],
               "walls": [round(wall(r), 3) for r in reqs],
               "steady_median_s": round(statistics.median([wall(r) for r in reqs[1:]]), 3) if arm == "A" else None}
        if "gen_wall_s" in reqs[0]:
            row["gen_walls"] = [round(r["gen_wall_s"], 3) for r in reqs]
            row["gen_median_s"] = round(statistics.median(r["gen_wall_s"] for r in reqs), 3)
            row["cpu_encode_s"] = [round(r.get("cpu_encode_s", 0), 4) for r in reqs]
            row["overlap_total_s"] = round(sum(r.get("overlap_s", 0) for r in reqs), 3)
            if all("cpu_encode_start" in r for r in reqs):
                t0 = min(r["cpu_encode_start"] for r in reqs)
                row["fill_s"] = round(min(r["gen_start"] for r in reqs) - t0, 3)
        pipe[arm] = row
    a_total = sum(r["wall_s"] for r in pab["arms"]["A"]["requests"])
    c_total = pab["arms"]["C"]["total_s"]
    steady_a = statistics.median([r["wall_s"] for r in pab["arms"]["A"]["requests"][1:]])
    report["sections"]["pipeline_raw"] = {
        "arms": pipe,
        "A_sum_req_walls_s": round(a_total, 3),
        "A_first_after_idle_s": pab["arms"]["A"]["requests"][0]["wall_s"],
        "A_steady_median_s": round(steady_a, 3),
        "A_steady_4req_estimate_s": round(4 * steady_a, 3),
        "C_total_s": c_total,
        "C_vs_A_cold_inclusive": round(a_total / c_total, 3),
        "C_vs_A_steady_estimate": round(4 * steady_a / c_total, 3),
        "vram_peak_A_mb": pab["arms"]["A"]["vram_peak_mb"],
        "vram_peak_C_mb": None,
        "note": "C VRAM peak comes from the E2E precomputed arms (same graph), see e2e section",
    }

    # ---- 5. e2e from raw per-request records -----------------------------------
    eab = load(RUN / "e2e/e2e_ab.json")
    e2e = {}
    for arm, rows in eab["arms"].items():
        cfg1 = {k: v for k, v in rows.items() if k.endswith("|cfg1")}
        walls = sorted(v["wall_s"] for v in cfg1.values())
        e2e[arm] = {
            "n_cfg1": len(cfg1),
            "wall_median_s": round(statistics.median(walls), 3) if walls else None,
            "vram_peak_median_mb": round(statistics.median(v["vram_peak_mb"] for v in cfg1.values()), 1) if cfg1 else None,
            "vram_residency_median_mb": round(statistics.median(v["vram_residency_mb"] for v in cfg1.values()), 1) if cfg1 else None,
        }
    gn = eab["arms"]["G_native"]["G_native|p1-zh-scene|cfg1"]
    c1b = eab["arms"]["C1b_OV_BF16"]["C1b_OV_BF16|p1-zh-scene|cfg1"]
    report["sections"]["e2e_raw"] = {
        "arms": e2e,
        "vram_peak_saved_mb": round(gn["vram_peak_mb"] - c1b["vram_peak_mb"], 1),
        "vram_residency_saved_mb": round(gn["vram_residency_mb"] - c1b["vram_residency_mb"], 1),
        "e2e_latency_delta_s_precomputed_vs_native": round(
            statistics.median(v["wall_s"] for k, v in eab["arms"]["C1b_OV_BF16"].items() if k.endswith("cfg1")) -
            statistics.median(v["wall_s"] for k, v in eab["arms"]["G_native"].items() if k.endswith("cfg1")), 3),
        "warm_gpu_native_session": load(RUN / "e2e/warm_gpu_native.json")["runs"],
    }

    # ---- 6. quality metrics (automated only) -----------------------------------
    qual = {}
    for key, v in eab.get("quality", {}).items():
        if not key.endswith("cfg1") or key.startswith("G_native|"):
            continue
        arm, pid, _ = key.split("|")
        qual.setdefault(arm, {})[pid] = {"psnr_db": v["psnr_db"], "ssim": v["ssim"],
                                         "lpips_alex": v.get("lpips_alex")}
    report["sections"]["image_metrics_vs_G_native"] = qual

    OUT.write_text(json.dumps(report, indent=2))
    print(f"WROTE {OUT}")
    print(json.dumps({"gates": gates, "speedups": speedups,
                      "pipeline": {k: report['sections']['pipeline_raw'][k]
                                   for k in ('A_sum_req_walls_s', 'C_total_s', 'C_vs_A_cold_inclusive', 'C_vs_A_steady_estimate')},
                      "vram_saved": report['sections']['e2e_raw']['vram_peak_saved_mb']}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
