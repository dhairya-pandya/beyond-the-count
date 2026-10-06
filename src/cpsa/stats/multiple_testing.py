"""Benjamini-Hochberg FDR across powered endpoints and final verdict assignment."""
from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from cpsa.stats.power_check import power_flag

VERDICTS = ("morphology_advantage", "no_advantage", "indeterminate")


def bh_fdr(pvalues: np.ndarray) -> np.ndarray:
    """BH-adjusted q-values; NaN p-values stay NaN and are excluded from the pool."""
    p = np.asarray(pvalues, dtype=float)
    q = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    if ok.any():
        q[ok] = multipletests(p[ok], method="fdr_bh")[1]
    return q


def assign_verdicts(df: pd.DataFrame, alpha: float = 0.05, min_active: int = 15, min_inactive: int = 15) -> pd.DataFrame:
    """Add ``powered``, ``fdr_q`` and ``verdict`` columns.

    Only powered endpoints enter the BH pool; underpowered ones are ``indeterminate`` and get q = NaN.
    Needs columns n_active, n_inactive, delta_auroc, bootstrap_p.
    """
    out = df.copy()
    out["powered"] = [power_flag(a, i, min_active, min_inactive) for a, i in zip(out["n_active"], out["n_inactive"])]
    p = np.where(out["powered"], out["bootstrap_p"], np.nan)
    out["fdr_q"] = bh_fdr(p)
    adv = out["powered"] & (out["delta_auroc"] > 0) & (out["fdr_q"] < alpha)
    out["verdict"] = np.where(~out["powered"], "indeterminate", np.where(adv, "morphology_advantage", "no_advantage"))
    return out
