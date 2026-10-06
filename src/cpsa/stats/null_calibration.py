"""Empirical-null calibration of the paired-bootstrap p-values.

The paired bootstrap resamples compounds but keeps the out-of-fold predictions fixed, so it ignores training / split
variability. A label-permutation control (true delta = 0 by construction) shows that z = delta / SE_bootstrap then has a
null standard deviation above 1 and the raw p-values are anti-conservative. Following Efron's empirical-null idea we estimate
that null sd from permutation runs and widen the reference distribution: p_cal = P(Z > z / sd_null).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from cpsa.stats.multiple_testing import assign_verdicts

Z95 = 1.959964


def z_scores(delta, ci_lo, ci_hi):
    """delta / bootstrap SE, with SE recovered from the 95% percentile interval width."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.asarray(delta, dtype=float) / ((np.asarray(ci_hi, dtype=float) - np.asarray(ci_lo, dtype=float)) / (2 * Z95))


def calibrated_p(z, null_sd: float, null_mean: float = 0.0):
    """One-sided p for z under the empirical null N(null_mean, null_sd^2)."""
    return norm.sf((np.asarray(z, dtype=float) - null_mean) / null_sd)


def estimate_null(null: pd.DataFrame, min_active: int = 15, min_inactive: int = 15) -> dict:
    """Null sd of z for both baselines from permutation runs (only runs that satisfy the power rule are used)."""
    n = null[(null["n_active"] >= min_active) & (null["n_inactive"] >= min_inactive) & null["delta_AUROC"].notna()]
    out = {"n_runs": int(len(null)), "n_powered_runs": int(len(n))}
    for tag, suf in (("strong", ""), ("scalar", "_scalar")):
        z = z_scores(n[f"delta_AUROC{suf}"], n[f"delta_ci_lo{suf}"], n[f"delta_ci_hi{suf}"])
        out[f"sd_{tag}"] = float(np.std(z, ddof=1))
        out[f"mean_z_{tag}"] = float(np.mean(z))
        out[f"type1_at_0.05_{tag}_uncalibrated"] = float(np.mean(n[f"bootstrap_p{suf}"] < 0.05))
    return out


def calibrate_table(audit: pd.DataFrame, sd_strong: float, sd_scalar: float, alpha: float = 0.05,
                    min_active: int = 15, min_inactive: int = 15) -> pd.DataFrame:
    """Recompute p, BH q and verdicts with null-calibrated p-values; raw columns are kept as ``*_uncalibrated``."""
    out = audit.copy()
    for col in ("verdict", "verdict_scalar", "fdr_q", "fdr_q_scalar"):
        out[f"{col}_uncalibrated"] = out[col]
    is_ctrl = out["verdict_uncalibrated"] == "positive_control"
    scored = out["delta_AUROC"].notna() & ~is_ctrl
    for suf, sd in (("", sd_strong), ("_scalar", sd_scalar)):
        z = z_scores(out[f"delta_AUROC{suf}"], out[f"delta_ci_lo{suf}"], out[f"delta_ci_hi{suf}"])
        out[f"delta_z{suf}"] = z
        out[f"calibrated_p{suf}"] = calibrated_p(z, sd)
        sub = out.loc[scored, ["n_active", "n_inactive"]].assign(delta_auroc=out.loc[scored, f"delta_AUROC{suf}"], bootstrap_p=out.loc[scored, f"calibrated_p{suf}"])
        v = assign_verdicts(sub, alpha, min_active, min_inactive)
        out[f"fdr_q{suf}"] = np.nan
        out.loc[scored, f"fdr_q{suf}"] = v["fdr_q"]
        out[f"verdict{suf}"] = out[f"verdict{suf}_uncalibrated"]
        out.loc[scored, f"verdict{suf}"] = v["verdict"]
    out["null_sd_strong"], out["null_sd_scalar"] = sd_strong, sd_scalar
    return out
