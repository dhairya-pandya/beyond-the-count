#!/usr/bin/env python
"""OPTIONAL demo extra: fetch a few example Cell Painting images (a handful of files, NOT a bulk download) from the public
Cell Painting Gallery (s3://cellpainting-gallery/cpg0037-oasis, CC0, no-sign-request) and save small false-colour composites
under demo/assets/images/. Uses data/raw/index.parquet to look up the exact object keys; never lists the bucket.
Featured compounds: hit in every assay / Cell-Painting-only hit (morphology without cell loss) / inactive."""
import io
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import tifffile
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cpsa import RAW, ROOT  # noqa: E402
from cpsa.data.load_labels import SI  # noqa: E402

OUT = ROOT / "demo" / "assets" / "images"
HTTP = "https://cellpainting-gallery.s3.amazonaws.com/"
SITE = 5.0
# false-colour: (R, G, B) weights per channel
COLOR = {"DNA": (0, 0, 1), "ER": (0, 1, 0), "AGP": (1, 0, 0), "Mito": (1, 0, 1), "RNA": (0, 1, 1)}


def fetch_tiff(key: str, attempts: int = 4) -> np.ndarray:
    for i in range(attempts):  # ~8.6 MB per channel; the public bucket can be slow
        try:
            r = requests.get(HTTP + key.removeprefix("s3://cellpainting-gallery/"), timeout=300)
            r.raise_for_status()
            return tifffile.imread(io.BytesIO(r.content)).astype("float32")
        except requests.RequestException:
            if i == attempts - 1:
                raise


def composite(files: dict[str, str], crop: int = 1024) -> Image.Image:
    rgb = np.zeros((crop, crop, 3), dtype="float32")
    for ch, key in files.items():
        if ch not in COLOR:
            continue
        a = fetch_tiff(key)
        h, w = a.shape
        a = a[(h - crop) // 2:(h + crop) // 2, (w - crop) // 2:(w + crop) // 2]
        lo, hi = np.percentile(a, [1, 99.7])
        a = np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1)
        for i, wgt in enumerate(COLOR[ch]):
            rgb[..., i] += wgt * a
    return Image.fromarray((np.clip(rgb, 0, 1) * 255).astype("uint8")).resize((512, 512), Image.LANCZOS)


def well_files(index: pd.DataFrame, plate: str, well: str) -> dict[str, str]:
    r = index[(index["Metadata_Plate"] == plate) & (index["Metadata_Well"] == well) & (index["Metadata_Site"] == SITE)]
    return dict(zip(r["Channel"], r["Filename"]))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    index = pd.read_parquet(RAW / "index.parquet")
    meta = pd.read_parquet(RAW / "metadata.parquet")
    hits = pd.read_csv(SI / "hit_summary.csv").dropna(subset=["OASIS_ID"]).drop_duplicates("OASIS_ID")
    groups = {
        "hit_in_all_assays": hits[hits["Hit_in_all_assays"] == "Yes"],
        "cell_painting_only": hits[(hits["Cell_Painting_hit"] == "Yes") & (hits["Cell_count_hit"] == "No") & (hits["MT_hit"] == "No") & (hits["LDH_hit"] == "No")],
        "inactive": hits[(hits["Cell_Painting_hit"] == "No") & (hits["Cell_count_hit"] == "No") & (hits["MT_hit"] == "No") & (hits["LDH_hit"] == "No")],
    }
    rng = np.random.default_rng(0)
    jobs, rows = [], []
    for tag, g in groups.items():
        for oid in rng.choice(g["OASIS_ID"].to_numpy(), size=2, replace=False):
            w = meta[meta["OASIS_ID"] == oid] if "OASIS_ID" in meta else meta[meta["Metadata_OASIS_ID"] == oid]
            top = w[w["Metadata_Concentration"] == w["Metadata_Concentration"].max()].iloc[0]
            dmso = meta[(meta["Metadata_Plate"] == top["Metadata_Plate"]) & (meta["Metadata_Compound"] == "DMSO")].iloc[0]
            for kind, row in (("treated", top), ("dmso", dmso)):
                files = well_files(index, row["Metadata_Plate"], row["Metadata_Well"])
                if len(files) < 5:
                    continue
                name = f"{oid}_{kind}.png"
                jobs.append((name, files))
                rows.append(dict(OASIS_ID=oid, group=tag, kind=kind, file=name, plate=row["Metadata_Plate"], well=row["Metadata_Well"],
                                 concentration_uM=float(row["Metadata_Concentration"])))

    def run(job):
        name, files = job
        if (OUT / name).exists():
            return True
        try:
            composite(files).save(OUT / name)
            print("saved", name, flush=True)
            return True
        except requests.RequestException as e:
            print("FAILED", name, type(e).__name__, flush=True)
            return False

    with ThreadPoolExecutor(max_workers=12) as ex:  # one connection per well; bandwidth per connection is the bottleneck
        ok = dict(zip([j[0] for j in jobs], ex.map(run, jobs)))
    done = pd.DataFrame(rows)
    done = done[done["file"].map(ok)]
    complete = done.groupby("OASIS_ID")["kind"].nunique()  # keep compounds that have both treated and DMSO images
    done[done["OASIS_ID"].isin(complete[complete == 2].index)].to_csv(OUT / "manifest.csv", index=False)


if __name__ == "__main__":
    main()
