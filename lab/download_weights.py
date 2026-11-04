"""Parallel ranged downloader for hf-mirror (bypasses per-connection throttle).

Why this exists (2026-10-31, measured):
- `hf download` gets 40 MB/s via parallel chunks but its .incomplete part
  files NEVER resume after a forceful kill — every watchdog restart wasted
  the whole partial set (~38GB of duplicates for ~8GB of progress).
- Single `curl -C -` resumes byte-exact but the mirror throttles one
  connection to ~150 KB/s at peak hours.

This splits each file into N byte ranges fetched by N concurrent workers,
each resuming byte-exact on its own chunk file; kill/restart at any time.
Run with any python that has requests (e.g. the phase2 venv):

  phase2-comfyui/.venv/Scripts/python.exe -I tools/download_parallel.py \
      <url> <out_path> [workers]
"""

import os
import struct
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests


def _verify_safetensors_header(path: Path) -> None:
    """Cheap integrity gate: parse the safetensors header length prefix.

    Catches torn/interleaved assemblies at download time instead of at
    ComfyUI model-load time (2026-11-01 lesson: a size-correct but corrupt
    file survived assembly and killed the E1 turbo bench).
    """
    with open(path, "rb") as f:
        prefix = f.read(8)
    if len(prefix) != 8:
        raise RuntimeError(f"{path.name}: cannot read header prefix")
    (header_len,) = struct.unpack("<Q", prefix)
    if not (0 < header_len < 100_000_000) or 8 + header_len > path.stat().st_size:
        raise RuntimeError(f"{path.name}: corrupt safetensors header "
                           f"(len={header_len}, file={path.stat().st_size})")


def _acquire_lock(parts_dir: Path) -> Path:
    """Guard against concurrent instances on the same target (torn writes)."""
    lock = parts_dir.parent / (parts_dir.name + ".lock")
    if lock.exists():
        age = time.time() - lock.stat().st_mtime
        if age < 3600:
            raise RuntimeError(f"another downloader holds {lock} (age {age:.0f}s)")
    lock.write_text(str(os.getpid()))
    return lock


def head_size(url: str) -> int:
    for _ in range(10):
        try:
            r = requests.head(url, allow_redirects=True, timeout=30)
            r.raise_for_status()
            return int(r.headers["Content-Length"])
        except Exception:  # noqa: BLE001
            time.sleep(3)
    raise RuntimeError(f"HEAD failed: {url}")


def download_range(url: str, start: int, end: int, path: Path, stop: threading.Event) -> None:
    want = end - start + 1
    for _ in range(2000):
        if stop.is_set():
            return
        have = path.stat().st_size if path.exists() else 0
        if have >= want:
            return
        headers = {"Range": f"bytes={start + have}-{end}"}
        try:
            with requests.get(url, headers=headers, stream=True, timeout=(15, 90)) as r:
                r.raise_for_status()
                with open(path, "ab") as f:
                    for chunk in r.iter_content(1 << 20):
                        if stop.is_set():
                            return
                        f.write(chunk)
        except Exception:  # noqa: BLE001 - retry with adjusted Range
            time.sleep(3)
    raise RuntimeError(f"chunk {path} did not finish")


def main() -> None:
    url = sys.argv[1]
    out = Path(sys.argv[2])
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 16
    total = head_size(url)
    if out.exists() and out.stat().st_size == total:
        print(f"SKIP {out.name} (complete)")
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    chunk = (total + workers - 1) // workers
    ranges = [(i * chunk, min((i + 1) * chunk, total) - 1)
              for i in range(workers) if i * chunk < total]
    parts_dir = out.parent / (out.name + ".parts")
    parts_dir.mkdir(parents=True, exist_ok=True)
    lock = _acquire_lock(parts_dir)
    stop = threading.Event()
    t0 = time.time()

    def progress():
        while not stop.is_set():
            have = sum(p.stat().st_size for p in parts_dir.glob("*") if p.is_file())
            rate = have / max(time.time() - t0, 1) / 1e6
            eta = (total - have) / (rate * 1e6) if rate > 0.01 else float("inf")
            print(f"  {have/1e9:.2f}/{total/1e9:.2f} GB  {rate:.1f} MB/s  "
                  f"eta {eta/60:.0f}m", flush=True)
            time.sleep(15)

    mon = threading.Thread(target=progress, daemon=True)
    mon.start()
    with ThreadPoolExecutor(len(ranges)) as ex:
        futs = [ex.submit(download_range, url, s, e, parts_dir / f"{i:03d}", stop)
                for i, (s, e) in enumerate(ranges)]
        for f in futs:
            f.result()
    stop.set()

    with open(out, "wb") as w:
        for i in range(len(ranges)):
            with open(parts_dir / f"{i:03d}", "rb") as r:
                while True:
                    b = r.read(1 << 22)
                    if not b:
                        break
                    w.write(b)
    assert out.stat().st_size == total, f"{out}: {out.stat().st_size} != {total}"
    _verify_safetensors_header(out)
    for p in parts_dir.iterdir():
        p.unlink()
    parts_dir.rmdir()
    lock.unlink(missing_ok=True)
    print(f"DONE {out.name} {total} bytes in {(time.time()-t0)/60:.1f} min "
          f"(header verified)", flush=True)


if __name__ == "__main__":
    main()
