"""EU-OPENSCREEN Bioactive Compound Set (HepG2, four imaging sites) as a second morphology source for the audit.

Profiles: Zenodo 10.5281/zenodo.19347244 (CC BY 4.0), annotations from the v1.0.0 record (10.5281/zenodo.13309566). Each plate file holds
one row per well (median over cells) with a per-well object count; every compound was imaged once per replicate at 10 uM, so only a
single-number cell-count baseline exists here (no dose-response curve).

Compounds are matched to the OASIS compounds of the main audit by the skeleton block of their InChIKey (first 14 characters), so the
existing ToxCast / Tox21 / MT / LDH labels transfer without a new label download.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from cpsa import DATA

EU = DATA / "external" / "eu_os"
AGG = EU / "Aggregated_Profiles" / "aggregated_data"
ANN = EU / "annotations"
SITES = {  # site folder -> annotation file
    "FMP_HepG2": "2022-07-08_Annotation_Bioactives_HepG2.csv",
    "IMTM_HepG2": "2023-08-14_Annotation2_IMTM_HepG2.csv",
    "MEDINA_HepG2": "2023-11-28_Annotation_MEDINA_HepG2.csv",
    "USC_HepG2": "2023-11-28_Annotation_USC_HepG2.csv",
}
META = ["Metadata_Batch", "Metadata_Plate", "Metadata_Well"]
COUNT = "Metadata_Object_Count"
DROP_PATTERNS = ("Center_", "BoundingBox", "ObjectNumber")  # image-position bookkeeping, not morphology


def feature_columns(columns) -> list[str]:
    return [c for c in columns if not c.startswith("Metadata_") and c != "site" and not any(p in c for p in DROP_PATTERNS)]


def load_site(site: str, agg_dir: Path = AGG, ann_dir: Path = ANN, concentration: float = 10.0) -> pd.DataFrame:
    """All plates of one site with compound annotation; keeps DMSO controls and wells of the 10 uM compound plates."""
    ann = pd.read_csv(ann_dir / SITES[site])
    frames = [pd.read_csv(f) for f in sorted((agg_dir / site).glob("*_CP_Profiles_Aggregated.csv"))]
    d = pd.concat(frames, ignore_index=True)
    d = d.merge(ann[["Metadata_Batch", "Metadata_Plate", "Metadata_Well", "Metadata_EOS", "Metadata_Concentration"]], on=META, how="left")
    keep = (d["Metadata_EOS"] == "DMSO") | ((d["Metadata_EOS"].astype(str).str.startswith("EOS")) & (d["Metadata_Concentration"] == concentration))
    d = d[keep].copy()
    d["site"] = site
    return d


def robust_normalize(d: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    """Per plate, centre and scale every feature by the median and MAD of that plate's DMSO wells; the cell count is divided by the DMSO median."""
    out = d[META + ["Metadata_EOS", "site"]].copy()
    norm = np.full((len(d), len(feats)), np.nan, dtype="float32")
    count = np.full(len(d), np.nan)
    keys = d[["site", "Metadata_Batch", "Metadata_Plate"]].astype(str).agg("|".join, axis=1).to_numpy()
    X = d[feats].to_numpy(dtype="float32")
    for k in np.unique(keys):
        m = keys == k
        ctrl = m & (d["Metadata_EOS"] == "DMSO").to_numpy()
        if ctrl.sum() < 3:
            continue
        med = np.nanmedian(X[ctrl], axis=0)
        mad = np.nanmedian(np.abs(X[ctrl] - med), axis=0) * 1.4826
        mad[mad < 1e-9] = np.nan
        norm[m] = (X[m] - med) / mad
        count[m] = d.loc[m, COUNT].to_numpy() / np.nanmedian(d.loc[ctrl, COUNT].to_numpy())
    out[feats] = norm
    out["cell_count_ratio"] = count
    return out


def compound_map(overlap_csv: Path = EU / "overlap_oasis_eu.csv") -> pd.DataFrame:
    """EOS id -> OASIS_ID through pdid and the InChIKey skeleton (one OASIS id per EOS id; ambiguous duplicates keep the first)."""
    eos = pd.read_csv(ANN / "2024-08-02_EOS_pdid.csv")
    ov = pd.read_csv(overlap_csv)[["OASIS_ID", "pdid"]].drop_duplicates()
    m = eos.merge(ov, left_on="Metadata_pdid", right_on="pdid", how="inner")[["Metadata_EOS", "OASIS_ID"]]
    return m.drop_duplicates("Metadata_EOS")


def build_overlap_tables(sites=tuple(SITES), min_wells: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(profile, baseline) with one row per OASIS compound: median normalised profile over replicates and sites; baseline = median cell-count ratio."""
    parts = [load_site(s) for s in sites]
    cols = [c for c in parts[0].columns if c in set.intersection(*[set(p.columns) for p in parts])]
    feats = feature_columns(cols)
    d = pd.concat([p[[c for c in p.columns if c in set(cols)]] for p in parts], ignore_index=True)
    numeric = d[feats].apply(pd.to_numeric, errors="coerce")
    finite = np.isfinite(numeric.to_numpy(dtype="float64")).all(axis=0)  # features with any missing or infinite well are dropped
    feats = [f for f, k in zip(feats, finite) if k]
    d[feats] = numeric[feats]
    n = robust_normalize(d, feats)
    n = n[n["Metadata_EOS"] != "DMSO"].merge(compound_map(), on="Metadata_EOS", how="inner")
    n = n.dropna(subset=["cell_count_ratio"])
    g = n.groupby("OASIS_ID")
    wells = g.size()
    keep = wells[wells >= min_wells].index
    profile = g[feats].median().loc[keep]
    profile = profile.loc[:, profile.notna().all()]
    baseline = pd.DataFrame({"cell_count_ratio": g["cell_count_ratio"].median().loc[keep]})
    return profile, baseline
