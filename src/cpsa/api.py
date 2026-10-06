"""Dataset-agnostic entry point of the shortcut audit.

Give it any phenotypic-profiling table (samples x features), a baseline table that holds *only* the shortcut information
(for example cell-count features) and binary endpoint labels (samples x endpoints, 1 / 0 / NaN); it returns, per endpoint, whether the
full profile beats the baseline after multiple-testing correction, with explicit 'indeterminate' verdicts for under-powered endpoints.

    from cpsa.api import shortcut_audit, permutation_null, estimate_null_sd
    null = permutation_null(profile, baseline, labels, n_runs=100)          # optional but recommended
    res = shortcut_audit(profile, baseline, labels, null_sd=estimate_null_sd(null))
    res.table

Nothing here knows about Cell Painting, ToxCast or hepatocytes; ``cpsa.audit`` is the dataset-specific orchestration built on the same pieces.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from cpsa.models.train_eval import evaluate_endpoint
from cpsa.stats.multiple_testing import assign_verdicts
from cpsa.stats.null_calibration import calibrated_p, z_scores


@dataclass
class AuditResult:
    table: pd.DataFrame
    oof: pd.DataFrame
    null_sd: float | None


def _one(endpoint, data, y, full_cols, base_cols, n_splits, n_repeats, n_boot, seed, groups):
    n_act, n_inact = int(y.sum()), int((1 - y).sum())
    k = min(n_splits, n_act, n_inact)
    row = dict(endpoint_id=endpoint, n_samples=len(y), n_active=n_act, n_inactive=n_inact)
    if k < 3:
        return row, None
    res = evaluate_endpoint(data, y, {"full": full_cols, "baseline": base_cols}, n_splits=k, n_repeats=n_repeats, n_boot=n_boot, seed=seed, groups=groups)
    m, d = res["metrics"], res["delta"]["baseline"]
    row.update(baseline_AUROC=m["baseline"]["auroc"], baseline_PRAUC=m["baseline"]["prauc"], full_AUROC=m["full"]["auroc"], full_PRAUC=m["full"]["prauc"],
               delta_AUROC=d["delta"], delta_ci_lo=d["ci_lo"], delta_ci_hi=d["ci_hi"], bootstrap_p=d["p"])
    oof = pd.DataFrame({"endpoint_id": endpoint, "sample": res["ids"], "y": res["y"], "p_full": res["oof"]["full"], "p_baseline": res["oof"]["baseline"]})
    return row, oof


def _prepare(profile: pd.DataFrame, baseline: pd.DataFrame, labels: pd.DataFrame, groups):
    if baseline.index.has_duplicates or profile.index.has_duplicates:
        raise ValueError("sample index must be unique (one row per sample; pass `groups` for replicate structure)")
    common = profile.index.intersection(baseline.index).intersection(labels.index)
    if len(common) == 0:
        raise ValueError("profile, baseline and labels share no sample ids")
    both = pd.concat([baseline.loc[common].add_prefix("__base__"), profile.loc[common]], axis=1)
    g = None if groups is None else groups.reindex(common)
    return both, list(profile.columns), [f"__base__{c}" for c in baseline.columns], labels.loc[common], g


def _run(both, full_cols, base_cols, labels, endpoints, groups, n_splits, n_repeats, n_boot, seed, n_jobs):
    jobs = []
    for e in endpoints:
        y = labels[e].dropna()
        data = both.loc[y.index]
        g = None if groups is None else groups.loc[y.index].to_numpy()
        jobs.append(delayed(_one)(e, data, y.astype(int), full_cols, base_cols, n_splits, n_repeats, n_boot, seed, g))
    return Parallel(n_jobs=n_jobs)(jobs)


def shortcut_audit(profile: pd.DataFrame, baseline: pd.DataFrame, labels: pd.DataFrame, *, groups: pd.Series | None = None,
                   controls: tuple[str, ...] = (), n_splits: int = 5, n_repeats: int = 3, n_boot: int = 1000, seed: int = 0, n_jobs: int = 1,
                   min_active: int = 15, min_inactive: int = 15, alpha: float = 0.05, null_sd: float | None = None) -> AuditResult:
    """Audit every endpoint column of ``labels``. ``controls`` are endpoints defined from the baseline itself (excluded from the FDR pool).

    ``groups``: optional CV groups (e.g. compound id for replicate wells, chemical cluster); default = each row its own group.
    ``null_sd``: empirical-null sd of z (see ``estimate_null_sd``); if given, p-values are calibrated against it before BH.
    """
    both, full_cols, base_cols, lab, g = _prepare(profile, baseline, labels, groups)
    results = _run(both, full_cols, base_cols, lab, list(lab.columns), g, n_splits, n_repeats, n_boot, seed, n_jobs)
    t = pd.DataFrame([r[0] for r in results])
    oof = pd.concat([r[1] for r in results if r[1] is not None], ignore_index=True) if any(r[1] is not None for r in results) else pd.DataFrame()
    scored = t["delta_AUROC"].notna() if "delta_AUROC" in t else pd.Series(False, index=t.index)
    is_ctrl = t["endpoint_id"].isin(controls)
    pool = scored & ~is_ctrl
    t["delta_z"] = np.nan
    t["calibrated_p"] = np.nan
    t["fdr_q"] = np.nan
    t["verdict"] = np.where(is_ctrl, "positive_control", "indeterminate")
    t["powered"] = (t["n_active"] >= min_active) & (t["n_inactive"] >= min_inactive)
    if pool.any():
        z = z_scores(t.loc[pool, "delta_AUROC"], t.loc[pool, "delta_ci_lo"], t.loc[pool, "delta_ci_hi"])
        t.loc[pool, "delta_z"] = z
        p = calibrated_p(z, null_sd) if null_sd is not None else t.loc[pool, "bootstrap_p"].to_numpy()
        t.loc[pool, "calibrated_p"] = p
        sub = t.loc[pool, ["n_active", "n_inactive", "delta_AUROC"]].rename(columns={"delta_AUROC": "delta_auroc"}).assign(bootstrap_p=p)
        v = assign_verdicts(sub, alpha, min_active, min_inactive)
        t.loc[pool, "fdr_q"] = v["fdr_q"]
        t.loc[pool, "verdict"] = v["verdict"]
    return AuditResult(t, oof, null_sd)


def permutation_null(profile: pd.DataFrame, baseline: pd.DataFrame, labels: pd.DataFrame, *, n_runs: int = 100, groups: pd.Series | None = None,
                     n_splits: int = 5, n_repeats: int = 3, n_boot: int = 1000, seed: int = 0, n_jobs: int = 1,
                     min_active: int = 15, min_inactive: int = 15) -> pd.DataFrame:
    """Label-permutation runs (true delta = 0): returns one row per run with delta, bootstrap CI, raw p and z, for powered endpoints only."""
    both, full_cols, base_cols, lab, g = _prepare(profile, baseline, labels, groups)
    rng = np.random.default_rng(seed)
    ok = [e for e in lab.columns if (lab[e] == 1).sum() >= min_active and (lab[e] == 0).sum() >= min_inactive]
    if not ok:
        raise ValueError("no endpoint satisfies the power rule; cannot build a permutation null")
    picks = rng.choice(ok, size=n_runs, replace=True)
    perm = {}
    for i, e in enumerate(picks):
        y = lab[e].dropna()
        perm[f"{e}__perm{i}"] = pd.Series(rng.permutation(y.to_numpy()), index=y.index)
    plab = pd.DataFrame(perm).reindex(lab.index)
    results = _run(both, full_cols, base_cols, plab, list(plab.columns), g, n_splits, n_repeats, n_boot, seed, n_jobs)
    t = pd.DataFrame([r[0] for r in results if r[1] is not None])
    t["z"] = z_scores(t["delta_AUROC"], t["delta_ci_lo"], t["delta_ci_hi"])
    return t


def estimate_null_sd(null: pd.DataFrame) -> float:
    """sd of z over permutation runs (>= 1; below 1 would mean the bootstrap is conservative, which we do not exploit)."""
    return float(max(np.std(null["z"], ddof=1), 1.0))
