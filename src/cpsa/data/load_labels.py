"""Outcome labels: ToxCast/Tox21 hit-calls (paper repo annotations) and the paper's native LDH/MT/cell-count hits."""
from __future__ import annotations

import pandas as pd

from cpsa import EXTERNAL

REPO = EXTERNAL / "ewald_repo"
ANNOT = REPO / "1_snakemake" / "inputs" / "annotations"
SI = REPO / "2_downstream_analysis" / "compiled_results" / "SI_tables"

TOXCAST_CATEGORIES = ("cellbased", "cellfree", "cytotox")
NATIVE = {"MT_hit": "MT", "LDH_hit": "LDH", "Cell_count_hit": "cell_count"}


def load_toxcast_binary(category: str) -> pd.DataFrame:
    """Compound x endpoint matrix of 0/1/NaN hit-calls, index OASIS_ID."""
    if category not in TOXCAST_CATEGORIES:
        raise ValueError(f"category must be one of {TOXCAST_CATEGORIES}")
    return pd.read_parquet(ANNOT / f"toxcast_{category}_binary.parquet").set_index("OASIS_ID").astype(float)


def load_native_hits() -> pd.DataFrame:
    """The paper's own readouts as binary labels (Yes -> 1): MT, LDH, cell_count. Index OASIS_ID."""
    h = pd.read_csv(SI / "hit_summary.csv").dropna(subset=["OASIS_ID"]).drop_duplicates("OASIS_ID").set_index("OASIS_ID")
    return pd.DataFrame({new: (h[old] == "Yes").astype(float) for old, new in NATIVE.items()})


def load_all_labels() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (labels, endpoints).

    labels: OASIS_ID x endpoint_id, values 0/1/NaN.
    endpoints: one row per endpoint with category, description and assay/target-family annotation.
    """
    frames, meta = [], []
    for cat in TOXCAST_CATEGORIES:
        lab = load_toxcast_binary(cat)
        frames.append(lab)
        if cat == "cytotox":
            for col in lab.columns:
                src_type, _, src = col.partition("__")
                meta.append(dict(endpoint_id=col, category=cat, endpoint_description=f"ToxCast cytotoxicity burst: {src_type} = {src}",
                                 assay_target_family="cytotoxicity", assay_design_type="cytotoxicity burst", cell_short_name=None, tissue=None))
            continue
        info = pd.read_parquet(ANNOT / f"toxcast_{cat}_info.parquet").drop_duplicates("assay_component_endpoint_name").set_index("assay_component_endpoint_name")
        for col in lab.columns:
            r = info.loc[col] if col in info.index else None
            meta.append(dict(
                endpoint_id=col, category=cat,
                endpoint_description=None if r is None else r["assay_component_desc"],
                assay_target_family=None if r is None else r["intended_target_family"],
                assay_design_type=None if r is None else r["assay_design_type"],
                cell_short_name=None if r is None else r["cell_short_name"],
                tissue=None if r is None else r["tissue"],
            ))
    nat = load_native_hits()
    frames.append(nat)
    for col in nat.columns:
        meta.append(dict(endpoint_id=col, category="native", endpoint_description=f"Ewald et al. {col} hit (concentration-response active)",
                         assay_target_family="native readout", assay_design_type="native", cell_short_name="PHH", tissue="liver"))
    labels = pd.concat(frames, axis=1)
    endpoints = pd.DataFrame(meta)
    if endpoints["endpoint_id"].duplicated().any():
        raise ValueError("endpoint ids collide across categories: " + str(endpoints.loc[endpoints["endpoint_id"].duplicated(), "endpoint_id"].tolist()))
    return labels, endpoints.set_index("endpoint_id")
