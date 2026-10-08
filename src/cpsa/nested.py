"""Nested comparison: does morphology add information *on top of* cell count?

The main audit compares morphology alone against cell count alone. Here the full model sees morphology **and** the
cell-count curve features (``nested``), and is compared against the cell-count-only model (``strong_cc``) on identical
folds, so a positive delta means the profile carries information the cell count does not. The same label-permutation
machinery calibrates the bootstrap p-values, and an optional regularised logistic regression on the morphology features
serves as a sanity comparator for the tree model.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from cpsa.audit import CONTROL_ENDPOINTS, _class_counts
from cpsa.data.build_dataset import baseline_strong_columns, build_compound_table, build_endpoint_dataset, profile_columns
from cpsa.data.load_labels import load_all_labels
from cpsa.models.train_eval import auroc, evaluate_endpoint, oof_predict
from cpsa.stats.multiple_testing import bh_fdr
from cpsa.stats.null_calibration import calibrated_p, z_scores
from cpsa.stats.power_check import power_flag

NESTED_VERDICTS = ("adds_beyond_count", "no_detectable_addition", "indeterminate")


def nested_columns(df: pd.DataFrame) -> list[str]:
    """Morphology profile plus the cell-count-only features."""
    return profile_columns(df) + baseline_strong_columns(df)


def nested_one(endpoint: str, table: pd.DataFrame, labels: pd.DataFrame, n_splits: int = 5, n_repeats: int = 3, n_boot: int = 1000,
               seed: int = 0, with_logreg: bool = False) -> dict | None:
    """One endpoint: nested model vs the cell-count-only model on identical folds (and optionally a logistic-regression sanity check)."""
    data = build_endpoint_dataset(table, labels, endpoint)
    if data is None:
        return None
    y = data.pop("label")
    n_act, n_inact = int(y.sum()), int((1 - y).sum())
    k = min(n_splits, n_act, n_inact)
    row = dict(endpoint_id=endpoint, n_compounds=len(y), n_active=n_act, n_inactive=n_inact)
    if k < 3:
        return row
    # "full" below is the nested model (morphology + count features); evaluate_endpoint compares "full" against each other model
    res = evaluate_endpoint(data, y, {"full": nested_columns(data), "strong_cc": baseline_strong_columns(data)},
                            n_splits=k, n_repeats=n_repeats, n_boot=n_boot, seed=seed)
    m, d = res["metrics"], res["delta"]["strong_cc"]
    row.update(cv_folds=k, nested_AUROC=m["full"]["auroc"], nested_PRAUC=m["full"]["prauc"], strong_cc_AUROC=m["strong_cc"]["auroc"],
               delta_nested=d["delta"], delta_nested_ci_lo=d["ci_lo"], delta_nested_ci_hi=d["ci_hi"], bootstrap_p_nested=d["p"])
    if with_logreg:
        p = oof_predict(data[profile_columns(data)], y.to_numpy().astype(int), np.asarray(data.index), k, n_repeats, seed, model="logreg").mean(axis=0)
        row["logreg_AUROC"] = auroc(y.to_numpy().astype(int), p)
    return row


def nested_audit(feature_set: str = "cellprofiler", agg_method: str = "allpod", *, endpoints: list[str] | None = None, n_repeats: int = 3,
                 n_boot: int = 1000, n_jobs: int = 8, seed: int = 0, with_logreg: bool = True, min_active: int = 15,
                 min_inactive: int = 15) -> pd.DataFrame:
    """Nested comparison for every powered endpoint (the positive control is excluded)."""
    labels, ep_meta = load_all_labels()
    table = build_compound_table(feature_set, agg_method)
    n1, n0 = _class_counts(labels, table)
    ids = [e for e in ep_meta.index if e not in CONTROL_ENDPOINTS and n1[e] >= min_active and n0[e] >= min_inactive]
    if endpoints:
        ids = [e for e in ids if e in set(endpoints)]
    rows = Parallel(n_jobs=n_jobs, verbose=10)(delayed(nested_one)(e, table, labels, 5, n_repeats, n_boot, seed, with_logreg) for e in ids)
    return pd.DataFrame([r for r in rows if r is not None])


def nested_null(feature_set: str = "cellprofiler", agg_method: str = "allpod", *, n_endpoints: int = 100, n_perms: int = 3,
                n_repeats: int = 3, n_boot: int = 1000, n_jobs: int = 8, seed: int = 0, min_active: int = 15) -> pd.DataFrame:
    """Label-permutation runs of the nested comparison at the audit's own settings (true delta = 0 by construction)."""
    labels, ep_meta = load_all_labels()
    table = build_compound_table(feature_set, agg_method)
    n1, n0 = _class_counts(labels, table)
    cand = [e for e in ep_meta.index if e not in CONTROL_ENDPOINTS and n1[e] >= min_active and n0[e] >= min_active]
    rng = np.random.default_rng(seed)
    chosen = rng.choice(cand, size=min(n_endpoints, len(cand)), replace=False)
    jobs = []
    for e in chosen:
        for k in range(n_perms):
            y = labels[e].dropna()
            name = f"{e}__perm{k}__seed{seed}"
            jobs.append((name, pd.DataFrame({name: pd.Series(rng.permutation(y.to_numpy()), index=y.index)})))
    rows = Parallel(n_jobs=n_jobs, verbose=10)(delayed(nested_one)(name, table, lab, 5, n_repeats, n_boot, seed, False) for name, lab in jobs)
    return pd.DataFrame([r for r in rows if r is not None and "delta_nested" in r])


def estimate_nested_null(null: pd.DataFrame, min_active: int = 15, min_inactive: int = 15) -> dict:
    """Null sd of z = delta / SE from permutation runs that satisfy the power rule."""
    n = null[(null["n_active"] >= min_active) & (null["n_inactive"] >= min_inactive) & null["delta_nested"].notna()]
    z = z_scores(n["delta_nested"], n["delta_nested_ci_lo"], n["delta_nested_ci_hi"])
    return {"n_runs": int(len(n)), "sd": float(np.std(z, ddof=1)), "mean_z": float(np.mean(z)),
            "type1_at_0.05_uncalibrated": float(np.mean(n["bootstrap_p_nested"] < 0.05))}


def calibrate_nested(rows: pd.DataFrame, null_sd: float, alpha: float = 0.05, min_active: int = 15, min_inactive: int = 15) -> pd.DataFrame:
    """Calibrated p, BH q over powered endpoints, and a verdict for the nested comparison."""
    out = rows.copy()
    out["delta_z_nested"] = z_scores(out["delta_nested"], out["delta_nested_ci_lo"], out["delta_nested_ci_hi"])
    out["calibrated_p_nested"] = calibrated_p(out["delta_z_nested"], null_sd)
    powered = np.array([power_flag(a, i, min_active, min_inactive) for a, i in zip(out["n_active"], out["n_inactive"])]) & out["delta_nested"].notna().to_numpy()
    out["fdr_q_nested"] = bh_fdr(np.where(powered, out["calibrated_p_nested"], np.nan))
    adds = powered & (out["delta_nested"] > 0).to_numpy() & (out["fdr_q_nested"] < alpha).to_numpy()
    out["verdict_nested"] = np.where(~powered, "indeterminate", np.where(adds, "adds_beyond_count", "no_detectable_addition"))
    out["null_sd_nested"] = null_sd
    return out
