#!/usr/bin/env python3
"""Bilingual, CPU-only structural gate for committed .ipynb files.

No Jupyter runtime is required. Python cell execution is covered by pytest.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CATALOG = ROOT / "notebooks" / "catalog.json"


def source(cell: dict) -> str:
    s = cell.get("source", [])
    return "".join(s) if isinstance(s, list) else s


def run() -> list[str]:
    errors = []
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    assert len(catalog["modules"]) == 9, "Expected M0–M7 including M5.1"
    seen = set()
    for item in catalog["modules"]:
        pair = []
        for language, filename in (("en", item["english"]), ("zh-CN", item["chinese"])):
            path = ROOT / "notebooks" / filename
            if filename in seen:
                errors.append(f"Duplicate notebook in catalog: {filename}")
            seen.add(filename)
            if not path.is_file():
                errors.append(f"Missing notebook: {filename}")
                continue
            try:
                nb = json.loads(path.read_text(encoding="utf-8"))
            except (ValueError, OSError) as e:
                errors.append(f"{filename}: invalid JSON: {e}")
                continue
            if nb.get("nbformat") != 4:
                errors.append(f"{filename}: nbformat must be 4")
            if nb.get("metadata", {}).get("course", {}).get("language") != language:
                errors.append(f"{filename}: incorrect language metadata")
            if nb.get("metadata", {}).get("course", {}).get("gpu_default") is not False:
                errors.append(f"{filename}: GPU must default to OFF")
            if not nb.get("cells") or nb["cells"][0]["cell_type"] != "markdown":
                errors.append(f"{filename}: missing introductory markdown")
            code = []
            for i, cell in enumerate(nb.get("cells", [])):
                if cell["cell_type"] == "code":
                    if cell.get("execution_count") is not None or cell.get("outputs"):
                        errors.append(f"{filename}:{i}: committed notebook outputs must be empty")
                    text = source(cell)
                    try:
                        compile(text, f"{filename}:cell{i}", "exec")
                    except SyntaxError as e:
                        errors.append(f"{filename}:{i}: invalid Python: {e}")
                    code.append(text)
            pair.append(code)
        if len(pair) == 2 and pair[0] != pair[1]:
            errors.append(f"{item['id']}: executable EN/ZH cells differ")
        enlesson = ROOT / item["lesson"]
        zhlesson = ROOT / item["lesson"].replace(".md", ".zh-CN.md")
        if not enlesson.is_file() or not zhlesson.is_file():
            errors.append(f"{item['id']}: corresponding lesson path missing")
    return errors


if __name__ == "__main__":
    errors = run()
    if errors:
        print("\n".join(errors))
        raise SystemExit(1)
    print("notebooks check: OK (9 modules / 18 notebooks, executable EN/ZH parity, GPU defaults OFF)")
