import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from cpsa.stats.null_calibration import calibrate_table, calibrated_p, estimate_null, z_scores


def _null_runs(n=400, sd=1.5, seed=0):
    rng = np.random.default_rng(seed)
    se = rng.uniform(0.05, 0.1, n)
    delta = rng.normal(0, sd * se)
    return pd.DataFrame({
        "delta_AUROC": delta, "delta_ci_lo": delta - 1.96 * se, "delta_ci_hi": delta + 1.96 * se,
        "delta_AUROC_scalar": delta, "delta_ci_lo_scalar": delta - 1.96 * se, "delta_ci_hi_scalar": delta + 1.96 * se,
        "bootstrap_p": norm.sf(delta / se), "bootstrap_p_scalar": norm.sf(delta / se),
        "n_active": 30, "n_inactive": 100,
    })


def test_z_scores_use_ci_width_as_se():
    assert z_scores(0.2, 0.1, 0.3) == pytest.approx(0.2 / (0.2 / 3.92), rel=1e-3)


def test_estimate_null_recovers_inflation_and_ignores_underpowered_runs():
    null = _null_runs(sd=1.5)
    junk = null.head(50).assign(n_active=5, delta_AUROC=5.0, delta_ci_lo=4.99, delta_ci_hi=5.01, delta_AUROC_scalar=5.0,
                                delta_ci_lo_scalar=4.99, delta_ci_hi_scalar=5.01)  # would wreck the sd if included
    est = estimate_null(pd.concat([null, junk], ignore_index=True))
    assert est["sd_strong"] == pytest.approx(1.5, abs=0.2) and est["n_powered_runs"] == 400
    assert est["type1_at_0.05_strong_uncalibrated"] > 0.10  # null sd 1.5 -> expected ~0.136, raw p-values anti-conservative


def test_calibrated_p_widens_with_null_sd():
    assert calibrated_p(3.0, 1.0) == pytest.approx(norm.sf(3.0))
    assert calibrated_p(3.0, 2.0) == pytest.approx(norm.sf(1.5))


def _audit_rows():
    se = 0.05
    rows = []
    for name, z in (("big", 8.0), ("borderline", 3.0), ("neg", -2.0)):
        d = z * se
        rows.append(dict(endpoint_id=name, n_active=40, n_inactive=100, delta_AUROC=d, delta_ci_lo=d - 1.96 * se, delta_ci_hi=d + 1.96 * se,
                         delta_AUROC_scalar=d, delta_ci_lo_scalar=d - 1.96 * se, delta_ci_hi_scalar=d + 1.96 * se,
                         bootstrap_p=norm.sf(z), bootstrap_p_scalar=norm.sf(z), verdict="x", verdict_scalar="x", fdr_q=0.0, fdr_q_scalar=0.0))
    rows.append(dict(endpoint_id="few", n_active=5, n_inactive=100, delta_AUROC=0.3, delta_ci_lo=0.2, delta_ci_hi=0.4, delta_AUROC_scalar=0.3,
                     delta_ci_lo_scalar=0.2, delta_ci_hi_scalar=0.4, bootstrap_p=0.001, bootstrap_p_scalar=0.001, verdict="indeterminate",
                     verdict_scalar="indeterminate", fdr_q=np.nan, fdr_q_scalar=np.nan))
    rows.append(dict(endpoint_id="cell_count", n_active=200, n_inactive=700, delta_AUROC=-0.01, delta_ci_lo=-0.02, delta_ci_hi=0.0,
                     delta_AUROC_scalar=0.0, delta_ci_lo_scalar=-0.02, delta_ci_hi_scalar=0.0, bootstrap_p=1.0, bootstrap_p_scalar=1.0,
                     verdict="positive_control", verdict_scalar="positive_control", fdr_q=np.nan, fdr_q_scalar=np.nan))
    return pd.DataFrame(rows)


def test_calibrate_table_keeps_raw_verdicts_and_demotes_borderline():
    out = calibrate_table(_audit_rows(), sd_strong=2.0, sd_scalar=2.0).set_index("endpoint_id")
    assert out.loc["big", "verdict"] == "morphology_advantage"      # z = 8 -> calibrated z = 4
    assert out.loc["borderline", "verdict"] == "no_advantage"        # z = 3 -> calibrated z = 1.5, p = 0.067
    assert out.loc["neg", "verdict"] == "no_advantage"
    assert out.loc["few", "verdict"] == "indeterminate" and out.loc["cell_count", "verdict"] == "positive_control"
    assert out.loc["borderline", "verdict_uncalibrated"] == "x"      # raw verdict preserved
    assert out.loc["big", "calibrated_p"] == pytest.approx(norm.sf(4.0), rel=1e-2)
    assert out["null_sd_strong"].iloc[0] == 2.0
