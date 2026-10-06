"""Is morphology's advantage concentrated in particular assay / target families? One-sided Fisher exact tests + BH."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

from cpsa.stats.multiple_testing import bh_fdr

TESTED = ("morphology_advantage", "no_advantage")


def enrichment_table(audit: pd.DataFrame, by: list[str], verdict_col: str = "verdict", min_group_size: int = 5) -> pd.DataFrame:
    """For every level of every column in ``by``: is it over-represented among 'morphology_advantage' endpoints
    relative to all other tested endpoints? Indeterminate endpoints and controls are not part of either class.
    q-values are BH-adjusted within each grouping variable."""
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
            rows.append(dict(by=col, group=level, n_in_group=n_in, n_adv_in_group=a, n_adv_out=b, n_out=len(rest),
                             frac_adv_in=a / n_in, frac_adv_out=b / max(len(rest), 1), odds_ratio=odds, p=p))
        if rows:
            t = pd.DataFrame(rows)
            t["q"] = bh_fdr(t["p"].to_numpy())
            out.append(t)
    if not out:
        return pd.DataFrame(columns=["by", "group", "n_in_group", "n_adv_in_group", "n_adv_out", "n_out", "frac_adv_in", "frac_adv_out", "odds_ratio", "p", "q"])
    return pd.concat(out, ignore_index=True).sort_values(["by", "p"]).reset_index(drop=True)
