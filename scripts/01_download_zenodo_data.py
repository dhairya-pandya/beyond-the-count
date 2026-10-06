#!/usr/bin/env python
"""Download profiles + metadata from Zenodo 10.5281/zenodo.17067683 (CC-BY 4.0) into data/raw/, verifying MD5.

Resumable. Total ~865 MB (cellprofiler 413 MB, dino 400 MB, cpcnn 48 MB, index 2.5 MB, metadata 0.7 MB).
Use --only cpcnn metadata to fetch a subset (cpcnn is enough for a quick start).
"""
import argparse
import hashlib
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cpsa import RAW  # noqa: E402

RECORD = "https://zenodo.org/api/records/17067683"
STEM = {"cellprofiler": "cellprofiler_raw", "cpcnn": "cpcnn_raw", "dino": "dino_raw", "index": "index", "metadata": "metadata"}


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url: str, dest: Path) -> None:
    pos = dest.stat().st_size if dest.exists() else 0
    headers = {"Range": f"bytes={pos}-"} if pos else {}
    with requests.get(url, stream=True, headers=headers, timeout=60) as r:
        if r.status_code == 416:  # already complete
            return
        r.raise_for_status()
        with open(dest, "ab" if pos and r.status_code == 206 else "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", nargs="*", choices=list(STEM), default=list(STEM))
    a = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    files = {f["key"]: f for f in requests.get(RECORD, timeout=60).json()["files"]}
    for name in a.only:
        key = f"{STEM[name]}.parquet"
        meta = files[key]
        dest = RAW / key
        if not (dest.exists() and dest.stat().st_size == meta["size"] and md5(dest) == meta["checksum"].split(":")[1]):
            print(f"downloading {key} ({meta['size'] / 1e6:.1f} MB)")
            fetch(meta["links"]["self"], dest)
        ok = md5(dest) == meta["checksum"].split(":")[1]
        print(f"{key}: {'OK' if ok else 'CHECKSUM MISMATCH'}")
        if not ok:
            sys.exit(1)


if __name__ == "__main__":
    main()
