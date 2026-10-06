#!/usr/bin/env python
"""Build the small, shippable asset tables the Streamlit demo reads (no raw data needed at demo time).

demo/assets/compounds.parquet      OASIS_ID, name, CASRN, DTXSID, paper hit calls (cell count / MT / LDH / Cell Painting)
demo/assets/dose_response.parquet  per compound x concentration: MT, LDH (normalised, mean of replicates), cell count / plate-DMSO median
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cpsa import RAW, ROOT  # noqa: E402
from cpsa.data.load_labels import ANNOT, SI  # noqa: E402

OUT = ROOT / "demo" / "assets"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    m = pd.read_parquet(RAW / "metadata.parquet")
    dmso = m[m["Metadata_Compound"] == "DMSO"].groupby("Metadata_Plate")["Metadata_Count_Cells"].median()
    t = m[(m["Metadata_Compound"] != "DMSO") & m["Metadata_OASIS_ID"].notna()].copy()
    t["cell_count_ratio"] = t["Metadata_Count_Cells"] / t["Metadata_Plate"].map(dmso)
    dr = (t.groupby(["Metadata_OASIS_ID", "Metadata_Concentration"], as_index=False)
            .agg(mt=("Metadata_mtt_normalized", "mean"), ldh=("Metadata_ldh_normalized", "mean"), cell_count_ratio=("cell_count_ratio", "mean"))
            .rename(columns={"Metadata_OASIS_ID": "OASIS_ID", "Metadata_Concentration": "concentration_uM"}))
    dr.to_parquet(OUT / "dose_response.parquet")

    ann = pd.read_csv(ANNOT / "v5_oasis_03Sept2024_simple.csv")[["OASIS_ID", "PREFERRED_NAME", "CASRN", "DTXSID"]]
    hits = pd.read_csv(SI / "hit_summary.csv").dropna(subset=["OASIS_ID"]).drop_duplicates("OASIS_ID")
    comp = ann.merge(hits.drop(columns=["Compound_name"]), on="OASIS_ID", how="inner").rename(columns={"PREFERRED_NAME": "name"})
    comp = comp[comp["OASIS_ID"].isin(dr["OASIS_ID"].unique())]
    comp.to_parquet(OUT / "compounds.parquet")
    print(f"{len(comp)} compounds, {len(dr)} dose-response rows -> {OUT}")


if __name__ == "__main__":
    main()
