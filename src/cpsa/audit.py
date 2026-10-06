"""End-to-end shortcut audit for one (feature set, aggregation) configuration."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from cpsa import RESULTS
from cpsa.data.build_dataset import (baseline_strong_columns, build_compound_table, build_endpoint_dataset,
                                      profile_columns, BASELINE_SCALAR)
from cpsa.data.load_labels import load_all_labels
from cpsa.models.train_eval import evaluate_endpoint
from cpsa.stats.calibration import calibration_summary
from cpsa.stats.multiple_testing import assign_verdicts

CONTROL_ENDPOINTS = ("cell_count",)  # label is defined from cell count itself: positive control, outside the FDR pool


def _class_counts(labels: pd.DataFrame, table: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Actives / inactives per endpoint among compounds that actually have a profile (the modeling set)."""
    lab = labels.loc[labels.index.intersection(table.index)]
    return (lab == 1).sum(), (lab == 0).sum()


def _audit_one(endpoint: str, table: pd.DataFrame, labels: pd.DataFrame, n_splits: int, n_repeats: int, n_boot: int, seed: int,
               group_map: pd.Series | None = None):
    data = build_endpoint_dataset(table, labels, endpoint)
    if data is None:
        return None
    y = data.pop("label")
    n_act, n_inact = int(y.sum()), int((1 - y).sum())
    k = min(n_splits, n_act, n_inact)
    if k < 3:
        return dict(endpoint_id=endpoint, n_compounds=len(y), n_active=n_act, n_inactive=n_inact), None
    models = {"full": profile_columns(data), "scalar_cc": BASELINE_SCALAR, "strong_cc": baseline_strong_columns(data)}
    groups = None if group_map is None else group_map.reindex(data.index).fillna(-1).to_numpy()
    if groups is not None and (groups == -1).any():  # compounds without a structure: singleton groups
        groups = np.where(groups == -1, -1 - np.arange(len(groups)), groups)
    res = evaluate_endpoint(data, y, models, n_splits=k, n_repeats=n_repeats, n_boot=n_boot, seed=seed, groups=groups)
    m, d = res["metrics"], res["delta"]
    cal = calibration_summary(res["y"], res["oof"]["full"], seed=seed)
    row = dict(
        endpoint_id=endpoint, n_compounds=len(y), n_active=n_act, n_inactive=n_inact, cv_folds=k,
        scalar_cc_AUROC=m["scalar_cc"]["auroc"], scalar_cc_PRAUC=m["scalar_cc"]["prauc"],
        strong_cc_AUROC=m["strong_cc"]["auroc"], strong_cc_PRAUC=m["strong_cc"]["prauc"],
        full_AUROC=m["full"]["auroc"], full_PRAUC=m["full"]["prauc"], full_AUROC_repeat_sd=m["full"]["auroc_repeat_sd"],
        delta_AUROC=d["strong_cc"]["delta"], delta_ci_lo=d["strong_cc"]["ci_lo"], delta_ci_hi=d["strong_cc"]["ci_hi"], bootstrap_p=d["strong_cc"]["p"],
        delta_AUROC_scalar=d["scalar_cc"]["delta"], delta_ci_lo_scalar=d["scalar_cc"]["ci_lo"], delta_ci_hi_scalar=d["scalar_cc"]["ci_hi"], bootstrap_p_scalar=d["scalar_cc"]["p"],
        calibration_brier=cal["brier"], brier_skill=cal["brier_skill"], ece=cal["ece"], calib_slope=cal["slope"], brier_recalibrated=cal["brier_recalibrated"],
    )
    oof = pd.DataFrame({"endpoint_id": endpoint, "OASIS_ID": res["ids"], "y": res["y"], **{f"p_{k_}": v for k_, v in res["oof"].items()}})
    return row, oof


def finalize(rows: pd.DataFrame, endpoints: pd.DataFrame, alpha: float = 0.05, min_active: int = 15, min_inactive: int = 15) -> pd.DataFrame:
    """Add annotations, BH q-values and verdicts (primary: vs strong cell-count baseline; sensitivity: vs scalar)."""
    df = rows.merge(endpoints, left_on="endpoint_id", right_index=True, how="left")
    is_ctrl = df["endpoint_id"].isin(CONTROL_ENDPOINTS)
    scored = df["delta_AUROC"].notna()
    pool = scored & ~is_ctrl
    for suffix, dcol, pcol in (("", "delta_AUROC", "bootstrap_p"), ("_scalar", "delta_AUROC_scalar", "bootstrap_p_scalar")):
        sub = df.loc[pool, ["n_active", "n_inactive"]].assign(delta_auroc=df.loc[pool, dcol], bootstrap_p=df.loc[pool, pcol])
        v = assign_verdicts(sub, alpha, min_active, min_inactive)
        df[f"fdr_q{suffix}"] = np.nan
        df[f"verdict{suffix}"] = "not_evaluated"
        df.loc[pool, f"fdr_q{suffix}"] = v["fdr_q"]
        df.loc[pool, f"verdict{suffix}"] = v["verdict"]
        df.loc[is_ctrl, f"verdict{suffix}"] = "positive_control"
        df.loc[~scored & ~is_ctrl, f"verdict{suffix}"] = "indeterminate"
    df["powered"] = (df["n_active"] >= min_active) & (df["n_inactive"] >= min_inactive)
    return df


def run_audit(feature_set: str = "cpcnn", agg_method: str = "allpod", *, categories: tuple[str, ...] | None = None,
              endpoints: list[str] | None = None, n_splits: int = 5, n_repeats: int = 3, n_boot: int = 1000,
              n_jobs: int = 8, seed: int = 0, out_dir: Path | None = None, verbose: bool = True, powered_only: bool = False,
              min_active: int = 15, min_inactive: int = 15, group_map: pd.Series | None = None) -> pd.DataFrame:
    """``group_map``: optional compound -> CV group (e.g. chemical cluster); default is one group per compound."""
    labels, ep_meta = load_all_labels()
    table = build_compound_table(feature_set, agg_method)
    ids = list(ep_meta.index)
    if categories:
        ids = [e for e in ids if ep_meta.loc[e, "category"] in categories]
    if endpoints:
        ids = [e for e in ids if e in set(endpoints)]
    if powered_only:  # secondary configurations: skip endpoints that would be indeterminate anyway
        n1, n0 = _class_counts(labels, table)
        ids = [e for e in ids if n1[e] >= min_active and n0[e] >= min_inactive]
    t0 = time.time()
    results = Parallel(n_jobs=n_jobs, verbose=10 if verbose else 0)(
        delayed(_audit_one)(e, table, labels, n_splits, n_repeats, n_boot, seed, group_map) for e in ids)
    rows = pd.DataFrame([r[0] for r in results if r is not None])
    oof_parts = [r[1] for r in results if r is not None and r[1] is not None]
    if not oof_parts:
        raise ValueError("no endpoint had enough compounds in both classes to run cross-validation")
    oof = pd.concat(oof_parts, ignore_index=True)
    final = finalize(rows, ep_meta)
    final.insert(0, "agg_method", agg_method)
    final.insert(0, "feature_set", feature_set)
    out_dir = Path(out_dir or RESULTS / f"{feature_set}_{agg_method}")
    out_dir.mkdir(parents=True, exist_ok=True)
    final.to_csv(out_dir / "audit_table.csv", index=False)
    oof.to_parquet(out_dir / "oof_predictions.parquet")
    if verbose:
        print(f"done {len(final)} endpoints in {time.time() - t0:.0f}s -> {out_dir}")
    return final


def null_control(feature_set: str = "cpcnn", agg_method: str = "allpod", *, n_endpoints: int = 60, n_perms: int = 1,
                 n_repeats: int = 2, n_boot: int = 500, n_jobs: int = 8, seed: int = 0, min_active: int = 15,
                 group_map: pd.Series | None = None) -> pd.DataFrame:
    """Label-permutation control for the whole audit machinery.

    Shuffles the labels of randomly chosen powered endpoints (breaking any compound->label relationship but keeping
    prevalence), re-runs the identical pipeline and returns per-run delta / bootstrap p. Under the null, AUROCs must
    sit near 0.5, p-values must be ~uniform (type-I rate at 0.05 about 5%), and BH must call (almost) nothing.
    """
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
            ys = pd.Series(rng.permutation(y.to_numpy()), index=y.index)
            lab = pd.DataFrame({f"{e}__perm{k}": ys})
            jobs.append((f"{e}__perm{k}", lab))
    results = Parallel(n_jobs=n_jobs)(delayed(_audit_one)(name, table, lab, 5, n_repeats, n_boot, seed, group_map) for name, lab in jobs)
    rows = pd.DataFrame([r[0] for r in results if r is not None and r[1] is not None])
    return finalize(rows.assign(), pd.DataFrame(index=rows["endpoint_id"]))
