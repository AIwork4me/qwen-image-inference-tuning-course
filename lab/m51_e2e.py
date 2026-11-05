"""M5.1 end-to-end W7900 experiment: inject validated conditioning into the
unchanged Qwen-Image-2.1 turbo DiT/VAE and measure real images.

Arms:
  G_native         ComfyUI GPU CLIPTextEncode + GPU DiT/VAE
  C0_native        ComfyUI CLIPLoader(device=cpu) + GPU DiT/VAE (one prompt)
  G_precomputed    GPU-derived cond injected via safetensors (bridge check)
  C1a/C1b/C2       OpenVINO FP32/BF16/INT8 cond injected via safetensors
  C0_precomputed   PyTorch CPU cond injected via safetensors
"""
from __future__ import annotations

import argparse, hashlib, json, math, shutil, statistics, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != REPO / "lab"]
sys.path.insert(0, str(REPO))

import numpy as np
from PIL import Image
from safetensors.torch import load_file, save_file
import torch

PROMPTS = {p["id"]: p for p in json.loads((REPO / "lab/prompts.json").read_text())["prompts"]}
UNET = "qwen_image_2.1_turbo_bf16.safetensors"
CLIP = "qwen3vl_8b_bf16.safetensors"
VAE = "qwen_image_2.1_vae_bf16.safetensors"


def install_node():
    src = REPO / "lab/custom_nodes/linuxlab_cond_safetensors.py"
    dst = REPO / "ComfyUI/custom_nodes/tuning_lab_safetensors/__init__.py"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    return dst


def normalize_conds(run: Path) -> dict:
    sources = {
        "G": run / "baseline/cond_cuda",
        "C0": run / "baseline/cond_cpu",
        "C1a": run / "ov/run_fp32/cond",
        "C1b": run / "ov/run_bf16_compressed/cond",
        "C2": run / "ov/run_int8/cond",
    }
    out = {}
    for arm, d in sources.items():
        tgt = (run / "e2e" / "cond" / arm).resolve()
        tgt.mkdir(parents=True, exist_ok=True)
        out[arm] = {}
        for pid in list(PROMPTS) + ["negative"]:
            p = d / f"{pid}-r0.safetensors"
            if not p.exists():
                continue
            t = load_file(str(p))["cond"].to(torch.float32)
            if t.ndim == 2:
                t = t.unsqueeze(0)
            save_file({"cond": t.contiguous()}, str(tgt / f"{pid}.safetensors"))
            out[arm][pid] = str(tgt / f"{pid}.safetensors")
    return out


def wf_native(prompt, negative, seed, cfg=1.0, clip_cpu=False):
    from lab import comfyui_client
    wf = comfyui_client.build_workflow(
        unet=UNET, clip=CLIP, vae=VAE, prompt=prompt, negative=negative,
        width=1024, height=1024, steps=8, seed=seed, sampler="euler",
        scheduler="simple", cfg=cfg)
    if clip_cpu:
        for node in wf.values():
            if node["class_type"] == "CLIPLoader":
                node["inputs"]["device"] = "cpu"
    return wf


def wf_precomputed(pos_path, neg_path, seed, cfg=1.0):
    from lab import comfyui_client
    wf = wf_native("", "", seed, cfg)
    clip_ids = [nid for nid, n in wf.items() if n["class_type"] == "CLIPLoader"]
    pos_ids = [nid for nid, n in wf.items() if n["class_type"] == "CLIPTextEncode"]
    for nid in clip_ids:
        wf.pop(nid)
    assert len(pos_ids) == 2
    for nid, path in zip(pos_ids, (pos_path, neg_path)):
        wf[nid] = {"class_type": "LinuxLabCondSafetensors", "inputs": {"cond_path": path}}
    return wf


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.ndimage import gaussian_filter
    a = a.astype(np.float64); b = b.astype(np.float64)
    if a.ndim == 3:
        return float(np.mean([ssim(a[..., i], b[..., i]) for i in range(a.shape[2])]))
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    mu1 = gaussian_filter(a, 1.5); mu2 = gaussian_filter(b, 1.5)
    s1 = gaussian_filter(a * a, 1.5) - mu1 * mu1
    s2 = gaussian_filter(b * b, 1.5) - mu2 * mu2
    s12 = gaussian_filter(a * b, 1.5) - mu1 * mu2
    return float(np.mean(((2 * mu1 * mu2 + C1) * (2 * s12 + C2)) /
                         ((mu1 ** 2 + mu2 ** 2 + C1) * (s1 + s2 + C2))))


def pixel_metrics(img_a: Path, img_b: Path) -> dict:
    a = np.asarray(Image.open(img_a).convert("RGB"), dtype=np.float64) / 255.0
    b = np.asarray(Image.open(img_b).convert("RGB"), dtype=np.float64) / 255.0
    mse = float(np.mean((a - b) ** 2))
    return {"psnr_db": float("inf") if mse == 0 else 20 * math.log10(1.0 / math.sqrt(mse)),
            "ssim": ssim(a, b), "mae": float(np.mean(np.abs(a - b))),
            "max_abs": float(np.max(np.abs(a - b)))}


def run_one(server, tag, wf, raw_dir, timeout=1800):
    from lab import comfyui_client
    mon = comfyui_client.RunMonitor(server.base_url)
    mon.start()
    t0 = time.perf_counter()
    pid = comfyui_client.post_prompt(server.base_url, wf, client_id=mon.client_id)
    hist = comfyui_client.wait_for_result(server.base_url, pid, timeout_s=timeout)
    wall = time.perf_counter() - t0
    mon.stop()
    s = mon.summarize(wf)
    data = comfyui_client.fetch_image(server.base_url, hist)
    p = raw_dir / (tag.replace("|", "__") + ".png"); p.write_bytes(data)
    s.update({"tag": tag, "wall_s": round(wall, 3), "image": str(p),
              "image_sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)})
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--port", type=int, default=8188)
    ap.add_argument("--skip-native-c0", action="store_true")
    args = ap.parse_args()
    run = Path(args.run); e2e = run / "e2e"; e2e.mkdir(parents=True, exist_ok=True)
    install_node()
    conds = normalize_conds(run)
    raw = e2e / "raw"; raw.mkdir(exist_ok=True)
    from lab.server import ComfyServer
    results = {"run": str(run), "arms": {}, "sessions": {}, "quality": {}, "status": {}}

    def session(name, arm_specs):
        log = e2e / f"server_{name}.log"
        srv = ComfyServer(port=args.port, log_path=log)
        srv.start()
        try:
            rss0 = int(open(f"/proc/{srv.proc.pid}/status").read().split("VmRSS:")[1].split()[0]) / 1024
            for tag, wf in arm_specs:
                print(f"[{name}] {tag}", flush=True)
                rec = run_one(srv, tag, wf, raw)
                results["arms"].setdefault(tag.split("|")[0], {})[tag] = rec
                (e2e / "e2e_ab.partial.json").write_text(json.dumps(results, indent=2))
                print(f"  wall={rec['wall_s']}s te={rec['text_encode_s']} ks={rec['ksampler_s']} "
                      f"vae={rec['vae_decode_s']} vram={rec['vram_peak_mb']}MB", flush=True)
            rss1 = int(open(f"/proc/{srv.proc.pid}/status").read().split("VmRSS:")[1].split()[0]) / 1024
            results["sessions"][name] = {"log": str(log), "boot_s": srv.boot_s,
                                          "server_rss_before_mb": round(rss0, 1),
                                          "server_rss_after_mb": round(rss1, 1),
                                          "attention": srv.scrape_attention_evidence()}
        finally:
            srv.stop()

    seed = 42
    # Session 1: native GPU TE + one native CPU TE, plus cfg=2 negative probe
    native = []
    for pid, pr in PROMPTS.items():
        native.append((f"G_native|{pid}|cfg1", wf_native(pr["prompt"], pr["negative_prompt"], seed, 1.0)))
    native.append(("G_native|p1-zh-scene|cfg2", wf_native(PROMPTS["p1-zh-scene"]["prompt"], "", seed, 2.0)))
    if not args.skip_native_c0:
        native.append(("C0_native|p1-zh-scene|cfg1", wf_native(PROMPTS["p1-zh-scene"]["prompt"], "", seed, 1.0, clip_cpu=True)))
        native.append(("C0_native|p1-zh-scene|cfg2", wf_native(PROMPTS["p1-zh-scene"]["prompt"], "", seed, 2.0, clip_cpu=True)))
    session("native", native)

    # Session 2: precomputed conditioning, no CLIPLoader ever instantiated
    pre = []
    for arm, label in [("G", "G_precomputed"), ("C1a", "C1a_OV_FP32"),
                       ("C1b", "C1b_OV_BF16"), ("C2", "C2_OV_INT8"),
                       ("C0", "C0_precomputed")]:
        for pid, pr in PROMPTS.items():
            pre.append((f"{label}|{pid}|cfg1",
                        wf_precomputed(conds[arm][pid], conds[arm]["negative"], seed, 1.0)))
    # negative-conditioning cfg=2 probe on p1
    for arm, label in [("G", "G_precomputed"), ("C0", "C0_precomputed"), ("C1b", "C1b_OV_BF16"), ("C2", "C2_OV_INT8")]:
        pre.append((f"{label}|p1-zh-scene|cfg2",
                    wf_precomputed(conds[arm]["p1-zh-scene"], conds[arm]["negative"], seed, 2.0)))
    session("precomputed", pre)

    # quality metrics vs native G
    try:
        import lpips
        lp = lpips.LPIPS(net="alex")
        lp_ok = True
    except Exception as e:
        lp_ok = False; results["lpips_error"] = repr(e)
    import torch
    for tag, rec in results["arms"].get("G_native", {}).items():
        pass
    quality = {}
    for arm, runs in results["arms"].items():
        for tag, rec in runs.items():
            parts = tag.split("|");
            if len(parts) != 3: continue
            _, pid, cfg = parts
            ref = results["arms"].get("G_native", {}).get(f"G_native|{pid}|{cfg}")
            if not ref:
                continue
            m = pixel_metrics(Path(ref["image"]), Path(rec["image"]))
            m["cond"] = {"note": "tensor metrics are in alignment/conditioning_alignment.json"}
            if lp_ok:
                a = lp(load_img_tensor(ref["image"]), load_img_tensor(rec["image"]), normalize=True).item()
                m["lpips_alex"] = float(a)
            quality[f"{arm}|{pid}|{cfg}"] = m
    results["quality"] = quality
    results["human_verdict"] = "PENDING_HUMAN_REVIEW"
    results["lpips_available"] = lp_ok
    (e2e / "e2e_ab.json").write_text(json.dumps(results, indent=2))
    print(f"WROTE {e2e/'e2e_ab.json'}")
    for k, v in quality.items():
        print(k, {kk: round(vv, 4) for kk, vv in v.items() if isinstance(vv, (int, float))})


def load_img_tensor(path):
    from torchvision import transforms
    import torch
    im = Image.open(path).convert("RGB")
    t = transforms.ToTensor()(im).unsqueeze(0)
    return t * 2 - 1


if __name__ == "__main__":
    main()
