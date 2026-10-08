#!/usr/bin/env python
"""OPTIONAL demo extra: example Cell Painting images of three cell systems for the demo's cell-line gallery.

Fetches a handful of single-well, single-site images (NOT a bulk download) from the public Cell Painting Gallery (CC0, no-sign-request):
HepG2 and U2OS from cpg0036-EU-OS-bioactives (FMP site, first replicate batch), each as a DMSO control and with two reference compounds
(nocodazole, a microtubule disruptor, and bortezomib, a proteasome inhibitor that is cytotoxic). Primary hepatocytes (cpg0037-oasis) are fetched for the same two
compounds at the highest tested concentration, with the DMSO control already in demo/assets/images. Channels: DNA blue, ER green, AGP red, mitochondria magenta (this site images four channels, no separate RNA).
Output: demo/assets/cell_lines/*.png and manifest.csv.
"""
import io
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import tifffile
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "demo" / "assets" / "cell_lines"
EU = ROOT / "data" / "external" / "eu_os" / "annotations"
HTTP = "https://cellpainting-gallery.s3.amazonaws.com/"
LOAD = "cpg0036-EU-OS-bioactives/FMP/workspace/load_data_csv/"
COLOR = {"DNA": (0, 0, 1), "ER": (0, 1, 0), "AGP": (1, 0, 0), "Mito": (1, 0, 1)}
SYSTEMS = {
    "HepG2": dict(batch="2021_09_03_Batch1_HepG2", annotation="2022-07-08_Annotation_Bioactives_HepG2.csv"),
    "U2OS": dict(batch="2023_02_15_Batch1_U2OS", annotation="2023-05-23_Annotation_Bioactives_U2OS_Corrected.csv"),
}
COMPOUNDS = {"DMSO control": "DMSO", "Nocodazole": "nocodazole", "Bortezomib": "bortezomib"}
OASIS_IDS = {"Nocodazole": "OASIS1560", "Bortezomib": "OASIS940"}  # the same reference compounds in the primary-hepatocyte data


def fetch_tiff(url: str, attempts: int = 4) -> np.ndarray:
    key = url.removeprefix("s3://cellpainting-gallery/")
    for i in range(attempts):
        try:
            r = requests.get(HTTP + key, timeout=300)
            r.raise_for_status()
            return tifffile.imread(io.BytesIO(r.content)).astype("float32")
        except requests.RequestException:
            if i == attempts - 1:
                raise


def composite(urls: dict[str, str], crop: int = 1024) -> Image.Image:
    with ThreadPoolExecutor(4) as ex:
        arrays = dict(zip(urls, ex.map(fetch_tiff, urls.values())))
    rgb = np.zeros((crop, crop, 3), dtype="float32")
    for ch, a in arrays.items():
        h, w = a.shape
        a = a[(h - crop) // 2:(h + crop) // 2, (w - crop) // 2:(w + crop) // 2]
        lo, hi = np.percentile(a, [1, 99.7])
        a = np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1)
        for i, wgt in enumerate(COLOR[ch]):
            rgb[..., i] += wgt * a
    return Image.fromarray((np.clip(rgb, 0, 1) * 255).astype("uint8")).resize((512, 512), Image.LANCZOS)


def find_well(annotation: pd.DataFrame, compound: str, names: pd.DataFrame, eos_pdid: pd.DataFrame) -> pd.Series:
    """First replicate-1 well of the compound: named control wells directly, else through name -> pdid -> EOS id."""
    a = annotation[annotation["Metadata_Batch"] == "R1"]
    hit = a[a["Metadata_EOS"].str.lower() == compound.lower()]
    if len(hit) == 0:
        pdid = names.loc[names["name"].str.lower() == compound.lower(), "pdid"]
        eos = eos_pdid.loc[eos_pdid["Metadata_pdid"].isin(pdid), "Metadata_EOS"]
        hit = a[a["Metadata_EOS"].isin(eos) & (a["Metadata_Concentration"] > 0)]
    if len(hit) == 0:
        raise LookupError(compound)
    return hit.sort_values(["Metadata_Plate", "Metadata_Well"]).iloc[0]


def oasis_rows() -> list[dict]:
    """Primary-hepatocyte composites for the reference compounds, reusing the helpers of scripts/14_fetch_example_images.py."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("fetch14", ROOT / "scripts" / "14_fetch_example_images.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    from cpsa import RAW
    index, meta = pd.read_parquet(RAW / "index.parquet"), pd.read_parquet(RAW / "metadata.parquet")
    id_col = "OASIS_ID" if "OASIS_ID" in meta else "Metadata_OASIS_ID"
    rows = []
    for label, oid in OASIS_IDS.items():
        w = meta[meta[id_col] == oid]
        top = w[w["Metadata_Concentration"] == w["Metadata_Concentration"].max()].iloc[0]
        files = m.well_files(index, top["Metadata_Plate"], top["Metadata_Well"])
        if len(files) < 5:
            continue
        img = m.composite(files)
        fname = f"Primary hepatocytes_{label.lower()}.png"
        img.save(OUT / fname, optimize=True)
        rows.append(dict(cell_line="Primary hepatocytes", compound=label, plate=top["Metadata_Plate"], well=top["Metadata_Well"],
                         concentration_uM=float(top["Metadata_Concentration"]), file=fname))
        print("saved", fname)
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    names = pd.read_csv(EU / "pd_export_04_2022_2464_compounds_standardized.csv", low_memory=False)[["pdid", "name"]]
    eos_pdid = pd.read_csv(EU / "2024-08-02_EOS_pdid.csv")
    rows = []
    for system, cfg in SYSTEMS.items():
        ann = pd.read_csv(EU / cfg["annotation"])
        load_cache = {}
        for label, compound in COMPOUNDS.items():
            try:
                w = find_well(ann, compound, names, eos_pdid)
            except LookupError:
                print("skip", system, compound)
                continue
            plate = f"{w['Metadata_Plate']}_R1"
            if plate not in load_cache:
                url = f"{HTTP}{LOAD}{cfg['batch']}/{plate}/load_data.csv"
                load_cache[plate] = pd.read_csv(io.BytesIO(requests.get(url, timeout=120).content))
            ld = load_cache[plate]
            r = ld[(ld["Metadata_Well"] == w["Metadata_Well"]) & (ld["Metadata_Site"] == 1)].iloc[0]
            img = composite({"DNA": r["URL_OrigDNA"], "ER": r["URL_OrigER"], "AGP": r["URL_OrigAGP"], "Mito": r["URL_OrigMito"]})
            fname = f"{system}_{compound.lower()}.png"
            img.save(OUT / fname, optimize=True)
            rows.append(dict(cell_line=system, compound=label, plate=plate, well=w["Metadata_Well"], concentration_uM=float(w["Metadata_Concentration"]), file=fname))
            print("saved", fname, plate, w["Metadata_Well"])
    rows += oasis_rows()
    pd.DataFrame(rows).to_csv(OUT / "manifest.csv", index=False)


if __name__ == "__main__":
    main()
