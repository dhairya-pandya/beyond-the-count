"""Technical (layout / batch) features per compound, for the technical-confound probe.

At compound level plate and well are undefined, but each compound occupies a fixed set of ~15 plates from two production
batches, so ordinal plate number, mean well row/column and batch number summarise *where* it was measured.
"""
from __future__ import annotations

import pandas as pd

from cpsa import RAW

TECH_COLUMNS = ["plate_min", "plate_mean", "row_mean", "col_mean", "batch_mean", "n_wells", "min_wells_per_conc", "max_conc_uM"]


def technical_features() -> pd.DataFrame:
    m = pd.read_parquet(RAW / "metadata.parquet")
    t = m[(m["Metadata_Compound"] != "DMSO") & m["Metadata_OASIS_ID"].notna()].copy()
    t["plate_num"] = t["Metadata_Plate"].str.extract(r"(\d+)$", expand=False).astype(float)
    t["batch_num"] = t["Metadata_source"].str.extract(r"(\d+)$", expand=False).astype(float)
    g = t.groupby("Metadata_OASIS_ID")
    out = pd.DataFrame({
        "plate_min": g["plate_num"].min(), "plate_mean": g["plate_num"].mean(),
        "row_mean": g["Metadata_row"].mean(), "col_mean": g["Metadata_col"].mean(), "batch_mean": g["batch_num"].mean(),
    })
    # replicate / dosing irregularities: 829 compound-concentration cells have a single well, a few have 4 or 6; 14 compounds use a 100x lower series
    per = t.groupby(["Metadata_OASIS_ID", "Metadata_Concentration"]).size().groupby(level=0)
    out["n_wells"] = g.size()
    out["min_wells_per_conc"] = per.min()
    out["max_conc_uM"] = g["Metadata_Concentration"].max()
    out.index.name = "OASIS_ID"
    return out
