"""Which *compounds* does morphology help with? Per-compound morphology benefit and its breakdown by compound class.

Benefit uses the decomposition AUROC = mean over actives of r_i = mean over inactives of r_i, where r_i is the fraction of compounds of the
opposite class that compound i is ranked correctly against. benefit_i = r_i(full) - r_i(cell-count baseline), averaged over the powered
endpoints in which compound i has a label. Positive = morphology orders this compound better than cell count does.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, rankdata

from cpsa.stats.multiple_testing import bh_fdr


def ranking_accuracy(y: np.ndarray, score: np.ndarray) -> np.ndarray:
    """r_i: for an active, P(score > score of a random inactive) (+0.5 ties); for an inactive, P(score < score of a random active)."""
    y = np.asarray(y).astype(int)
    s = np.asarray(score, dtype=float)
    r = np.full(len(y), np.nan)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return r
    neg_sorted, pos_sorted = np.sort(neg), np.sort(pos)
    a = s[y == 1]
    lo, hi = np.searchsorted(neg_sorted, a, "left"), np.searchsorted(neg_sorted, a, "right")
    r[y == 1] = (lo + 0.5 * (hi - lo)) / len(neg)
    b = s[y == 0]
    lo, hi = np.searchsorted(pos_sorted, b, "left"), np.searchsorted(pos_sorted, b, "right")
    r[y == 0] = ((len(pos) - hi) + 0.5 * (hi - lo)) / len(pos)
    return r


def compound_benefit(oof: pd.DataFrame, endpoints: list[str], base_col: str = "p_strong_cc", full_col: str = "p_full") -> pd.DataFrame:
    """Per compound: mean benefit over endpoints (``benefit``), over the endpoints where it is active (``benefit_active``), and counts."""
    parts = []
    for e, g in oof[oof["endpoint_id"].isin(endpoints)].groupby("endpoint_id"):
        d = ranking_accuracy(g["y"].to_numpy(), g[full_col].to_numpy()) - ranking_accuracy(g["y"].to_numpy(), g[base_col].to_numpy())
        parts.append(pd.DataFrame({"OASIS_ID": g["OASIS_ID"].to_numpy(), "y": g["y"].to_numpy(), "d": d}))
    allp = pd.concat(parts, ignore_index=True).dropna()
    out = allp.groupby("OASIS_ID").agg(benefit=("d", "mean"), n_endpoints=("d", "size"))
    out["benefit_active"] = allp[allp["y"] == 1].groupby("OASIS_ID")["d"].mean()
    out["n_active_endpoints"] = allp[allp["y"] == 1].groupby("OASIS_ID")["d"].size()
    return out


def class_table(benefit: pd.DataFrame, groups: pd.Series | pd.DataFrame, min_n: int = 8, n_boot: int = 1000, seed: int = 0, col: str = "benefit") -> pd.DataFrame:
    """Mean benefit per class with bootstrap CI, Mann-Whitney (class vs all other compounds) and BH q within this grouping.

    ``groups``: Series (one label per compound) or a long DataFrame with columns OASIS_ID, group (a compound may belong to several classes)."""
    if isinstance(groups, pd.Series):
        groups = groups.rename("group").rename_axis("OASIS_ID").reset_index()
    g = groups.dropna(subset=["group"]).drop_duplicates()
    g = g[g["OASIS_ID"].isin(benefit.index)]
    rng = np.random.default_rng(seed)
    rows = []
    for level, ids in g.groupby("group")["OASIS_ID"]:
        ids = pd.Index(ids)
        if len(ids) < min_n:
            continue
        x = benefit.loc[ids, col].dropna()
        rest = benefit.drop(index=ids, errors="ignore")[col].dropna()
        if len(x) < min_n or len(rest) < min_n:
            continue
        boots = [rng.choice(x.to_numpy(), len(x)).mean() for _ in range(n_boot)]
        rows.append(dict(group=level, n_compounds=len(x), mean_benefit=float(x.mean()), ci_lo=float(np.percentile(boots, 2.5)), ci_hi=float(np.percentile(boots, 97.5)),
                         mean_benefit_rest=float(rest.mean()), p=float(mannwhitneyu(x, rest, alternative="two-sided").pvalue)))
    t = pd.DataFrame(rows)
    if len(t):
        t["q"] = bh_fdr(t["p"].to_numpy())
        t = t.sort_values("p").reset_index(drop=True)
    return t
