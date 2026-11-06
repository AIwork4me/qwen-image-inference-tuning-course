"""Build owner review sheets (PR2 polish) — small committed derivatives + full-res index.

- Side-by-side JPEG sheets: frozen-campaign G_native vs C1b_OV_BF16 (release-relevant)
  and audit warm-session A vs C (fresh steady-state equivalence check).
- 100%-zoom typography crops from the p3 Chinese poster full-res originals.
- INDEX with SHA256 of every full-res source and where it lives (committed vs host-only).

Full-resolution originals are NOT re-committed: the frozen campaign PNGs are already
git-tracked; the audit warm-session PNGs stay on the measurement host (gitignored) and are
pinned by SHA256 here.

Usage: .venv-lab/bin/python scripts/release-closure/build_review_sheets.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from PIL import Image

REPO = Path(__file__).resolve().parents[2]
RUN = REPO / "runs/m51-20261105-cleanroom-01"
WARM = REPO / "runs/pr2-p4-pipewarm-120822/pipeline_warm"
OUT = REPO / "docs/release-closure/review-sheets"
SHEET_W = 768  # per-panel width in the side-by-side sheet
JPEG_Q = 88


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def side_by_side(a: Path, b: Path, la: str, lb: str, out: Path, caption: str) -> None:
    ia, ib = Image.open(a).convert("RGB"), Image.open(b).convert("RGB")
    w = SHEET_W
    ha, hb = round(ia.height * w / ia.width), round(ib.height * w / ib.width)
    ia, ib = ia.resize((w, ha), Image.LANCZOS), ib.resize((w, hb), Image.LANCZOS)
    H = max(ha, hb) + 56
    sheet = Image.new("RGB", (2 * w, H), (24, 24, 24))
    sheet.paste(ia, (0, 56))
    sheet.paste(ib, (w, 56))
    from PIL import ImageDraw
    d = ImageDraw.Draw(sheet)
    d.text((12, 8), caption, fill=(240, 240, 240))
    d.text((12, 30), f"LEFT {la}", fill=(120, 220, 120))
    d.text((w + 12, 30), f"RIGHT {lb}", fill=(120, 180, 250))
    sheet.save(out, "JPEG", quality=JPEG_Q)


def crop(p: Path, box, out: Path, label: str) -> None:
    Image.open(p).convert("RGB").crop(box).save(out, "JPEG", quality=JPEG_Q)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    index: dict = {"generated": __import__("time").strftime("%Y-%m-%dT%H:%M:%S%z"),
                   "full_resolution_sources": [], "sheets": []}

    # --- frozen campaign: G_native vs C1b deployment (release-relevant) ---
    for pid in ("p1-zh-scene", "p2-en-scene", "p3-poster-text"):
        g = RUN / f"e2e/raw/G_native__{pid}__cfg1.png"
        c = RUN / f"e2e/raw/C1b_OV_BF16__{pid}__cfg1.png"
        if not (g.is_file() and c.is_file()):
            continue
        sheet = OUT / f"campaign_{pid}_G_vs_C1b.jpg"
        side_by_side(g, c, "G_native (GPU TE reference)", "C1b OV BF16 (deployment candidate)",
                     sheet, f"{pid} | seed 42 | turbo BF16 8-step 1024 cfg=1 | frozen campaign 2026-11-05")
        index["sheets"].append({"file": sheet.name, "comparison": "G_native vs C1b", "prompt": pid})
        index["full_resolution_sources"] += [
            {"path": str(g.relative_to(REPO)), "committed_in_git": True, "sha256": sha256(g)},
            {"path": str(c.relative_to(REPO)), "committed_in_git": True, "sha256": sha256(c)}]

    # --- 100% typography crops from the p3 full-res originals ---
    g3 = RUN / "e2e/raw/G_native__p3-poster-text__cfg1.png"
    c3 = RUN / "e2e/raw/C1b_OV_BF16__p3-poster-text__cfg1.png"
    if g3.is_file() and c3.is_file():
        im = Image.open(g3)
        W, H = im.size
        box_l = (int(W * 0.25), int(H * 0.12), int(W * 0.75), int(H * 0.55))  # title block region
        crop(g3, box_l, OUT / "crop_poster_title_G_native.jpg", "G_native")
        crop(c3, box_l, OUT / "crop_poster_title_C1b.jpg", "C1b")
        index["sheets"] += [
            {"file": "crop_poster_title_G_native.jpg", "comparison": "typography 100% crop", "prompt": "p3-poster-text"},
            {"file": "crop_poster_title_C1b.jpg", "comparison": "typography 100% crop", "prompt": "p3-poster-text"}]

    # --- audit warm steady session: A vs C (fresh equivalence check) ---
    for rid in ("req1", "req2", "req3", "req4"):
        a = WARM / "raw" / f"A1|{rid}|cfg1.png"
        c = WARM / "raw" / f"C1|{rid}|cfg1.png"
        if not (a.is_file() and c.is_file()):
            # run_one may sanitize tags; fall back to any matching pattern
            a = next((WARM / "raw").glob(f"A*{rid}*.png"), a)
            c = next((WARM / "raw").glob(f"C*{rid}*.png"), c)
        if not (a.is_file() and c.is_file()):
            continue
        sheet = OUT / f"warmsteady_{rid}_A_vs_C.jpg"
        side_by_side(a, c, "A GPU-native TE (warm steady)", "C OV BF16 precomputed (warm steady)",
                     sheet, f"{rid} | seed 42 | audit warm balanced session 2026-11-05")
        index["sheets"].append({"file": sheet.name, "comparison": "warm A vs C", "prompt": rid})
        index["full_resolution_sources"] += [
            {"path": str(a.relative_to(REPO)), "committed_in_git": False,
             "sha256": sha256(a), "note": "audit host only; regenerate via lab.m51_pipeline_warm"},
            {"path": str(c.relative_to(REPO)), "committed_in_git": False,
             "sha256": sha256(c), "note": "audit host only; regenerate via lab.m51_pipeline_warm"}]

    (OUT / "INDEX.json").write_text(json.dumps(index, indent=2))
    total = sum(p.stat().st_size for p in OUT.glob("*.jpg"))
    print(f"WROTE {OUT} ({len(list(OUT.glob('*.jpg')))} sheets, {total/2**20:.1f} MiB)")
    for s in index["sheets"]:
        print(" -", s["file"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
