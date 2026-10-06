"""Compound-level annotations used for class breakdowns: vendor class labels (pathway / target / research area / clinical phase) from the
well metadata and the paper's own hit pattern across readouts (cell count, MT, LDH, Cell Painting)."""
from __future__ import annotations

import pandas as pd

from cpsa import RAW
from cpsa.data.load_labels import SI

MULTI = {"pathway": "Metadata_compound_pathway", "target": "Metadata_compound_target", "research_area": "Metadata_compound_research_area"}
SINGLE = {"clinical_phase": "Metadata_compound_clinical_information"}


def hit_pattern() -> pd.Series:
    """Mechanism-free class of every compound from the paper's hit calls: which readouts react?"""
    h = pd.read_csv(SI / "hit_summary.csv").dropna(subset=["OASIS_ID"]).drop_duplicates("OASIS_ID").set_index("OASIS_ID")
    yes = {c: h[c] == "Yes" for c in ("Cell_count_hit", "MT_hit", "LDH_hit", "Cell_Painting_hit")}
    out = pd.Series("inactive", index=h.index)
    out[yes["Cell_Painting_hit"] & ~yes["Cell_count_hit"] & ~yes["MT_hit"] & ~yes["LDH_hit"]] = "cell_painting_only"
    out[(yes["MT_hit"] | yes["LDH_hit"]) & ~yes["Cell_count_hit"]] = "mt_or_ldh_without_cell_loss"
    out[yes["Cell_count_hit"]] = "cell_loss"
    return out.rename("hit_pattern")


def vendor_classes() -> dict[str, pd.DataFrame]:
    """Long tables (OASIS_ID, group) per grouping. Multi-valued vendor labels ('A; B') are split; 'Others' is dropped."""
    m = pd.read_parquet(RAW / "metadata.parquet", columns=["Metadata_OASIS_ID", *MULTI.values(), *SINGLE.values()])
    m = m[m["Metadata_OASIS_ID"].notna()].drop_duplicates("Metadata_OASIS_ID").set_index("Metadata_OASIS_ID")
    out = {}
    for name, col in MULTI.items():
        s = m[col].dropna().str.split(";").explode().str.strip()
        s = s[~s.isin(["Others", ""])]
        out[name] = s.rename("group").rename_axis("OASIS_ID").reset_index()
    for name, col in SINGLE.items():
        out[name] = m[col].dropna().rename("group").rename_axis("OASIS_ID").reset_index()
    return out
