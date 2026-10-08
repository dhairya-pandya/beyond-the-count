"""What the audit concludes at organ-on-chip scale (simulation) and how to plan a chip study.

Liver-on-chip studies test tens of compounds, not about a thousand, and each compound is usually run on several chips. This module
(1) re-runs the audit on simulated data with known ground truth at chip-like sample sizes, (2) shows why grouping cross-validation by
compound matters when compounds are replicated across chips, and (3) turns the real endpoints' precision into a planning curve
(smallest detectable AUROC advantage against the number of compounds). Nothing here is chip data; it describes what the method needs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from cpsa.api import estimate_null_sd, permutation_null, shortcut_audit
from cpsa.models.train_eval import auroc, evaluate_endpoint
from cpsa.simulate import make_dataset
from cpsa.stats.precision import detectable_effect

DEFAULT_ENDPOINTS = {"null": 60, "shortcut": 40, "adds": 60, "morph_only": 20}
DEFAULT_PREVALENCE = (0.2, 0.35, 0.5)


def chip_scale_study(sizes=(20, 30, 40, 60, 100, 200), seeds=(0, 1, 2), *, endpoints: dict[str, int] | None = None,
                     prevalence: tuple[float, ...] = DEFAULT_PREVALENCE, n_null_runs: int = 80, n_repeats: int = 2, n_boot: int = 500,
                     chip_min: int = 5, n_jobs: int = 1) -> pd.DataFrame:
    """Audit simulated datasets of ``sizes`` compounds; one row per endpoint with raw and null-calibrated verdicts.

    The rule ``chip_min`` (actives and non-hits needed to run an endpoint at all) is lower than the 15/15 rule of the main audit, so that
    small designs produce estimates whose reliability can be examined; the strict rule is applied afterwards in :func:`summarise_study`.
    """
    endpoints = endpoints or DEFAULT_ENDPOINTS
    parts = []
    for n in sizes:
        for seed in seeds:
            d = make_dataset(n=n, n_features=120, endpoints=endpoints, prevalence=prevalence, seed=seed)
            kw = dict(n_repeats=n_repeats, n_boot=n_boot, seed=seed, n_jobs=n_jobs, min_active=chip_min, min_inactive=chip_min)
            sd = estimate_null_sd(permutation_null(d.profile, d.baseline, d.labels, n_runs=n_null_runs, **kw))
            raw = shortcut_audit(d.profile, d.baseline, d.labels, **kw).table
            cal = shortcut_audit(d.profile, d.baseline, d.labels, null_sd=sd, **kw).table
            t = raw[["endpoint_id", "n_active", "n_inactive", "powered", "delta_AUROC", "delta_ci_lo", "delta_ci_hi", "verdict"]]
            t = t.rename(columns={"verdict": "verdict_raw", "powered": "powered_chip"})
            t["verdict_calibrated"] = cal["verdict"].to_numpy()
            t["min_detectable_effect"] = detectable_effect(t["delta_ci_lo"], t["delta_ci_hi"], sd)
            t = t.merge(d.truth, on="endpoint_id")
            t["n_compounds"], t["seed"], t["null_sd"] = n, seed, sd
            parts.append(t)
    return pd.concat(parts, ignore_index=True)


def summarise_study(t: pd.DataFrame, strict_min: int = 15) -> pd.DataFrame:
    """Per number of compounds: how many endpoints can be judged, how often non-advantage endpoints are credited, and the power to credit real ones."""
    rows = []
    for n, g in t.groupby("n_compounds"):
        strict = (g.n_active >= strict_min) & (g.n_inactive >= strict_min)
        chip = g[g["powered_chip"]]
        null_like = chip[chip.kind.isin(["null", "shortcut"])]
        real = chip[chip.kind.isin(["adds", "morph_only"])]
        rows.append(dict(
            n_compounds=int(n), n_endpoints=len(g),
            indeterminate_strict=float(1 - strict.mean()), indeterminate_chip=float(1 - g["powered_chip"].mean()),
            credited_raw_on_null=float((null_like.verdict_raw == "morphology_advantage").mean()) if len(null_like) else np.nan,
            credited_calibrated_on_null=float((null_like.verdict_calibrated == "morphology_advantage").mean()) if len(null_like) else np.nan,
            power_raw=float((real.verdict_raw == "morphology_advantage").mean()) if len(real) else np.nan,
            power_calibrated=float((real.verdict_calibrated == "morphology_advantage").mean()) if len(real) else np.nan,
            median_min_detectable_effect=float(chip["min_detectable_effect"].median()) if len(chip) else np.nan,
            null_sd=float(g["null_sd"].mean()),
        ))
    return pd.DataFrame(rows)


def replicate_leakage_demo(n_compounds: int = 40, n_chips: int = 3, n_features: int = 60, n_labels: int = 30, chip_noise: float = 0.7,
                           seed: int = 0, n_repeats: int = 2) -> pd.DataFrame:
    """Compounds replicated over chips with compound-level labels that are pure noise (true AUROC 0.5).

    Cross-validation that treats every chip as independent lets a model recognise a compound it has already seen on another chip, so the
    AUROC rises above 0.5. Grouping the folds by compound removes that, which is why the audit's CV groups are compounds (and, for
    donor-structured designs, donors)."""
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(n_compounds, n_features))
    comp = np.repeat(np.arange(n_compounds), n_chips)
    X = pd.DataFrame(base[comp] + chip_noise * rng.normal(size=(len(comp), n_features)), index=[f"r{i}" for i in range(len(comp))])
    out = []
    for j in range(n_labels):
        y_comp = np.zeros(n_compounds, dtype=int)
        y_comp[rng.choice(n_compounds, n_compounds // 3, replace=False)] = 1
        y = pd.Series(y_comp[comp], index=X.index)
        for name, groups in (("grouped by compound", comp), ("each chip independent", np.arange(len(comp)))):
            r = evaluate_endpoint(X, y, {"full": list(X.columns)}, n_splits=5, n_repeats=n_repeats, n_boot=50, seed=seed + j, groups=groups)
            out.append(dict(label=j, cv=name, AUROC=auroc(r["y"], r["oof"]["full"])))
    return pd.DataFrame(out)


def fit_planning_curve(detectable: pd.DataFrame) -> dict:
    """log(minimum detectable effect) against log(effective sample size) from the real endpoints: MDE ≈ exp(a) · n_eff^b.

    ``n_eff = 4 / (1/n_active + 1/n_inactive)`` is the sample size of a balanced design with the same AUROC precision."""
    d = detectable.dropna(subset=["min_detectable_effect"])
    d = d[(d.min_detectable_effect > 0) & (d.n_active > 0) & (d.n_inactive > 0)]
    n_eff = 4 / (1 / d.n_active + 1 / d.n_inactive)
    x, y = np.log(n_eff.to_numpy()), np.log(d.min_detectable_effect.to_numpy())
    b, a = np.polyfit(x, y, 1)
    resid = y - (a + b * x)
    r2 = 1 - resid.var() / y.var()
    return dict(a=float(a), b=float(b), resid_sd=float(resid.std(ddof=2)), r2=float(r2), n_endpoints=int(len(d)))


def planned_detectable_effect(n_compounds: int, prevalence: float, curve: dict, band: float = 1.0) -> tuple[float, float, float]:
    """(typical, low, high) minimum detectable AUROC advantage for a study of ``n_compounds`` with the given share of actives.

    The band is ±``band`` residual standard deviations of the real endpoints around the fitted curve; values above 0.5 mean the design cannot
    detect any advantage."""
    n1 = n_compounds * prevalence
    n0 = n_compounds - n1
    if n1 < 1 or n0 < 1:
        return (np.nan, np.nan, np.nan)
    n_eff = 4 / (1 / n1 + 1 / n0)
    mid = curve["a"] + curve["b"] * np.log(n_eff)
    return (float(np.exp(mid)), float(np.exp(mid - band * curve["resid_sd"])), float(np.exp(mid + band * curve["resid_sd"])))
