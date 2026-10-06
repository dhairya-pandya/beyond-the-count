"""Re-implementation of the profile preprocessing in Ewald et al. (2024_09_09_Axiom_OASIS/1_snakemake).

raw well profiles -> per-plate DMSO-MAD normalisation -> correlation filter -> per-compound aggregation.
Aggregation methods (``all`` / ``allpod`` / ``allpodcc``) mirror ``classifier/aggregate_profiles.py``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import median_abs_deviation

from cpsa.data.load_profiles import CONTROL, feature_columns

AGG_METHODS = ("all", "allpod", "allpodcc")
N_CONC = 8
CONC_GRID_UM = (0.1, 1.0, 10.0, 100.0)


def mad_normalize(raw: pd.DataFrame, feats: list[str] | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Normalise to per-plate DMSO median/MAD; keep only 'variant' features.

    Features are dropped if non-finite anywhere, or if in any plate the DMSO MAD is 0
    or |MAD/median| <= 1e-3 (same rule as ``preprocessing.stats.select_variant_features``).
    """
    feats = list(feats) if feats is not None else feature_columns(raw)
    vals = raw[feats].to_numpy(dtype="float64")
    feats = [f for f, ok in zip(feats, np.isfinite(vals).all(axis=0)) if ok]
    meta_cols = [c for c in raw.columns if c.startswith("Metadata_")]
    ctrl = raw[raw["Metadata_Compound"] == CONTROL]

    med = ctrl.groupby("Metadata_Plate", observed=True)[feats].median()
    mad = ctrl.groupby("Metadata_Plate", observed=True)[feats].apply(
        lambda g: pd.Series(median_abs_deviation(g.to_numpy(), axis=0, nan_policy="omit"), index=feats)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        cv = (mad / med).abs().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    variant = ((mad != 0) & (cv > 1e-3)).all(axis=0)
    feats = [f for f in feats if variant[f]]

    plates = raw["Metadata_Plate"].to_numpy()
    out = raw[meta_cols].copy()
    x = raw[feats].to_numpy(dtype="float64")
    x = (x - med.loc[plates, feats].to_numpy()) / mad.loc[plates, feats].to_numpy()
    out = pd.concat([out, pd.DataFrame(x.astype("float32"), index=raw.index, columns=feats)], axis=1)
    return out, feats


def correlation_filter(df: pd.DataFrame, feats: list[str], threshold: float = 0.9) -> list[str]:
    """Greedy filter: scan features in order, drop any whose |Pearson r| with an already-kept feature > threshold."""
    corr = np.abs(np.corrcoef(df[feats].to_numpy(dtype="float64"), rowvar=False))
    keep: list[int] = []
    for i in range(len(feats)):
        if not keep or (corr[i, keep] <= threshold).all():
            keep.append(i)
    return [feats[i] for i in keep]


def _select_wells(t: pd.DataFrame, method: str, pod_cp: pd.Series | None, pod_cc: pd.Series | None) -> pd.DataFrame:
    if method == "all":
        return t
    ids = t["Metadata_OASIS_ID"]
    conc = t["Metadata_Concentration"]
    pcp = ids.map(pod_cp) if pod_cp is not None and len(pod_cp) else pd.Series(np.nan, index=t.index)
    above = (conc > pcp).fillna(False)
    if method == "allpod":
        sel = t[above]
        return pd.concat([sel, t[~ids.isin(sel["Metadata_OASIS_ID"])]])
    if method == "allpodcc":
        pcc = ids.map(pod_cc) if pod_cc is not None and len(pod_cc) else pd.Series(np.nan, index=t.index)
        window = above & (conc < pcc).fillna(False)  # null cc-POD -> empty window, as in the paper
        sel = t[window]
        rest = t[~ids.isin(sel["Metadata_OASIS_ID"])]
        # fallback 1: first concentration above the POD
        min_conc = t[above].groupby("Metadata_OASIS_ID")["Metadata_Concentration"].min()
        fb1 = rest[rest["Metadata_Concentration"] == rest["Metadata_OASIS_ID"].map(min_conc)]
        rest2 = rest[~rest["Metadata_OASIS_ID"].isin(fb1["Metadata_OASIS_ID"])]  # fallback 2: everything
        return pd.concat([sel, fb1, rest2])
    raise ValueError(f"unknown aggregation method {method!r}; expected one of {AGG_METHODS}")


def aggregate_compounds(
    norm: pd.DataFrame,
    feats: list[str],
    method: str,
    pod_cp: pd.Series | None,
    pod_cc: pd.Series | None,
) -> pd.DataFrame:
    """One row per compound (index ``OASIS_ID``): mean profile + ``Cell_Count`` (mean well cell count).

    ``pod_cp`` / ``pod_cc``: Series indexed by OASIS_ID giving the Cell Painting / cell-count
    point of departure in the same units as ``Metadata_Concentration`` (uM).
    """
    treated = norm[(norm["Metadata_Compound"] != CONTROL) & norm["Metadata_OASIS_ID"].notna()]
    sel = _select_wells(treated, method, pod_cp, pod_cc)
    g = sel.groupby("Metadata_OASIS_ID")
    agg = g[feats].mean()
    agg.insert(0, "Cell_Count", g["Metadata_Count_Cells"].mean())
    agg.index.name = "OASIS_ID"
    return agg


def cell_count_curve_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Cell-count-only dose-response summary per compound (the 'strong' shortcut baseline).

    cc_r1..cc_r8: mean cell count / plate-DMSO median at each of the compound's concentration ranks;
    cc_min, cc_auc: min and mean of those ratios. Uses ONLY Metadata_Count_Cells.
    """
    d = raw[["Metadata_Plate", "Metadata_Compound", "Metadata_OASIS_ID", "Metadata_Concentration", "Metadata_Count_Cells"]].copy()
    dmso_med = d[d["Metadata_Compound"] == CONTROL].groupby("Metadata_Plate")["Metadata_Count_Cells"].median()
    d = d[(d["Metadata_Compound"] != CONTROL) & d["Metadata_OASIS_ID"].notna()].copy()
    d["ratio"] = d["Metadata_Count_Cells"] / d["Metadata_Plate"].map(dmso_med)
    d["rank"] = d.groupby("Metadata_OASIS_ID")["Metadata_Concentration"].rank(method="dense").astype(int)
    wide = d.pivot_table(index="Metadata_OASIS_ID", columns="rank", values="ratio", aggfunc="mean")
    wide = wide.reindex(columns=range(1, N_CONC + 1))
    wide.columns = [f"cc_r{i}" for i in wide.columns]
    wide["cc_min"] = wide.min(axis=1)
    wide["cc_auc"] = wide[[f"cc_r{i}" for i in range(1, N_CONC + 1)]].mean(axis=1)
    # concentration-aware features: concentration series differ between source batches (a few compounds use a 100x lower series), so
    # rank k is not the same dose for everyone. Interpolate on log10(concentration); NaN outside the tested range (not measured != 0).
    per_conc = d.groupby(["Metadata_OASIS_ID", "Metadata_Concentration"])["ratio"].mean()
    grid = np.log10(CONC_GRID_UM)
    at = {}
    for oid, g in per_conc.groupby(level=0):
        x = np.log10(g.index.get_level_values(1).to_numpy(dtype=float))
        order = np.argsort(x)
        v = np.interp(grid, x[order], g.to_numpy()[order])
        v[(grid < x.min()) | (grid > x.max())] = np.nan
        at[oid] = v
    at = pd.DataFrame(at, index=[f"cc_at_{c:g}uM" for c in CONC_GRID_UM]).T
    wide = wide.join(at)
    wide.index.name = "OASIS_ID"
    return wide
