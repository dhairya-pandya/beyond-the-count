"""Join preprocessed compound profiles with baseline features and labels -> modeling-ready tables."""
from __future__ import annotations

import numpy as np
import pandas as pd

from cpsa import PROCESSED
from cpsa.data.load_labels import SI, load_all_labels
from cpsa.data.load_profiles import load_raw
from cpsa.data.preprocess import (AGG_METHODS, aggregate_compounds, cell_count_curve_features,
                                  correlation_filter, mad_normalize)

BASELINE_SCALAR = ["Cell_Count"]  # paper-faithful baseline


def baseline_strong_columns(df: pd.DataFrame) -> list[str]:
    """Cell-count-only dose-response baseline: scalar count + curve summaries (+ cell-count POD)."""
    return BASELINE_SCALAR + [c for c in df.columns if c.startswith("cc_")]


def profile_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in set(baseline_strong_columns(df))]


def load_pods(feature_set: str) -> tuple[pd.Series, pd.Series]:
    """(Cell Painting bioactivity POD, cell-count POD) in uM, indexed by OASIS_ID (from the paper's SI tables)."""
    cp = pd.read_csv(SI / f"cellpainting_{feature_set}_pods.csv")
    cp = cp[cp["Bioactivity_POD"].astype(str).str.lower() == "true"].dropna(subset=["OASIS_ID"])
    cc = pd.read_csv(SI / "cellcount_pods.csv").dropna(subset=["OASIS_ID"])
    return cp.groupby("OASIS_ID")["POD_um"].min(), cc.groupby("OASIS_ID")["POD_um"].min()


def normalized_profiles(feature_set: str, corr_threshold: float = 0.9) -> tuple[pd.DataFrame, list[str]]:
    """Cached: raw -> MAD-normalised -> correlation-filtered well-level profiles."""
    path = PROCESSED / f"{feature_set}_norm.parquet"
    if path.exists():
        df = pd.read_parquet(path)
        return df, [c for c in df.columns if not c.startswith("Metadata_")]
    raw = load_raw(feature_set)
    norm, feats = mad_normalize(raw)
    feats = correlation_filter(norm, feats, corr_threshold)
    keep = [c for c in norm.columns if c.startswith("Metadata_")] + feats
    norm = norm[keep]
    PROCESSED.mkdir(parents=True, exist_ok=True)
    norm.to_parquet(path)
    return norm, feats


def build_compound_table(feature_set: str, agg_method: str, *, use_cache: bool = True) -> pd.DataFrame:
    """One row per compound (index OASIS_ID): Cell_Count, cc_* curve features, then profile features."""
    if agg_method not in AGG_METHODS:
        raise ValueError(f"agg_method must be one of {AGG_METHODS}")
    path = PROCESSED / f"{feature_set}_{agg_method}.parquet"
    if use_cache and path.exists():
        return pd.read_parquet(path)
    norm, feats = normalized_profiles(feature_set)
    pod_cp, pod_cc = load_pods(feature_set)
    agg = aggregate_compounds(norm, feats, agg_method, pod_cp, pod_cc)
    cc = cell_count_curve_features(norm)
    cc["cc_pod_log10"] = np.log10(pod_cc.reindex(cc.index))
    out = pd.concat([agg[BASELINE_SCALAR], cc.reindex(agg.index), agg[feats]], axis=1)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path)
    return out


def build_endpoint_dataset(table: pd.DataFrame, labels: pd.DataFrame, endpoint: str) -> pd.DataFrame | None:
    """Rows = compounds with a label for ``endpoint``; adds ``label`` column. Index = OASIS_ID (= CV group)."""
    y = labels[endpoint].dropna()
    ids = y.index.intersection(table.index)
    if len(ids) == 0:
        return None
    d = table.loc[ids].copy()
    d["label"] = y.loc[ids].astype(int)
    return d


def save_labels() -> None:
    labels, endpoints = load_all_labels()
    PROCESSED.mkdir(parents=True, exist_ok=True)
    labels.to_parquet(PROCESSED / "labels.parquet")
    endpoints.to_parquet(PROCESSED / "endpoints.parquet")
