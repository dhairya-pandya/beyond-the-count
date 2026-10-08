"""Is morphology's advantage concentrated in particular assay / target families? One-sided Fisher exact tests + BH."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

from cpsa.stats.multiple_testing import bh_fdr

TESTED = ("morphology_advantage", "no_advantage")


def assay_cluster(endpoint_ids: pd.Series) -> pd.Series:
    """Assay cluster of an endpoint: the first two underscore-separated tokens of its id (e.g. ``BSK_SAg``, ``TOX21_ERa``).

    Endpoints of one assay share compounds, cell system and readout, so they are not independent observations."""
    return endpoint_ids.astype(str).str.split("_").str[:2].str.join("_")


def cluster_permutation_p(d: pd.DataFrame, col: str, level, cluster_col: str, n_perm: int = 5000, seed: int = 0) -> float:
    """One-sided cluster-level permutation p for 'advantage share is higher inside ``level`` than outside'.

    Every assay cluster is assigned to the level that most of its endpoints belong to. Under the null the level is unrelated to
    which assays carry advantages, so the same number of clusters is drawn at random as the 'in' set; the p-value is the share of
    draws whose in-minus-out advantage share is at least the observed one. A group that is a single assay therefore has a
    smallest possible p of about 1 / (number of clusters), however many endpoints that assay contributes."""
    uniq, inv = np.unique(d[cluster_col].to_numpy(), return_inverse=True)
    k = len(uniq)
    n_c = np.bincount(inv, minlength=k).astype(float)
    a_c = np.bincount(inv, weights=d["adv"].to_numpy().astype(float), minlength=k)
    n_in = np.bincount(inv, weights=(d[col] == level).to_numpy().astype(float), minlength=k)
    member = n_in > n_c / 2  # majority vote (ties stay outside)
    k_in = int(member.sum())
    if k_in == 0 or k_in == k:
        return float("nan")

    def stat(sel):  # sel: (..., k) boolean membership
        a_i, n_i = sel @ a_c, sel @ n_c
        return a_i / n_i - (a_c.sum() - a_i) / (n_c.sum() - n_i)

    obs = stat(member.astype(float))
    draws = np.random.default_rng(seed).random((n_perm, k)).argsort(axis=1)[:, :k_in]
    sel = np.zeros((n_perm, k))
    np.put_along_axis(sel, draws, 1.0, axis=1)
    return float((1 + (stat(sel) >= obs - 1e-12).sum()) / (n_perm + 1))


def enrichment_table(audit: pd.DataFrame, by: list[str], verdict_col: str = "verdict", min_group_size: int = 5,
                     cluster_col: str | None = None, n_perm: int = 5000) -> pd.DataFrame:
    """For every level of every column in ``by``: is it over-represented among 'morphology_advantage' endpoints
    relative to all other tested endpoints? Indeterminate endpoints and controls are not part of either class.
    q-values are BH-adjusted within each grouping variable.

    ``cluster_col``: if given (e.g. from :func:`assay_cluster`), every group also gets a cluster-level permutation p-value
    ``p_cluster`` (and BH ``q_cluster``): whole assays, not single endpoints, are the exchangeable units, so correlated endpoints
    from one assay count once. See :func:`cluster_permutation_p`."""
    d = audit[audit[verdict_col].isin(TESTED)].copy()
    d["adv"] = d[verdict_col] == "morphology_advantage"
    out = []
    for col in by:
        rows = []
        for level, g in d.dropna(subset=[col]).groupby(col):
            n_in = len(g)
            if n_in < min_group_size:
                continue
            a = int(g["adv"].sum())
            rest = d[(d[col] != level)]
            b = int(rest["adv"].sum())
            table = [[a, n_in - a], [b, len(rest) - b]]
            odds, p = fisher_exact(table, alternative="greater")
            row = dict(by=col, group=level, n_in_group=n_in, n_adv_in_group=a, n_adv_out=b, n_out=len(rest),
                       frac_adv_in=a / n_in, frac_adv_out=b / max(len(rest), 1), odds_ratio=odds, p=p)
            if cluster_col:
                row["p_cluster"] = cluster_permutation_p(d.dropna(subset=[col]), col, level, cluster_col, n_perm)
            rows.append(row)
        if rows:
            t = pd.DataFrame(rows)
            t["q"] = bh_fdr(t["p"].to_numpy())
            if cluster_col:
                t["q_cluster"] = bh_fdr(t["p_cluster"].to_numpy())
            out.append(t)
    if not out:
        return pd.DataFrame(columns=["by", "group", "n_in_group", "n_adv_in_group", "n_adv_out", "n_out", "frac_adv_in", "frac_adv_out", "odds_ratio", "p", "q"]
                            + (["p_cluster", "q_cluster"] if cluster_col else []))
    return pd.concat(out, ignore_index=True).sort_values(["by", "p"]).reset_index(drop=True)
