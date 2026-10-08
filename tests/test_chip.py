import numpy as np
import pandas as pd
import pytest

from cpsa.chip import (chip_scale_study, fit_planning_curve, planned_detectable_effect, replicate_leakage_demo, summarise_study)


def test_replicate_leakage_inflates_ungrouped_cv_on_noise_labels():
    r = replicate_leakage_demo(n_compounds=30, n_chips=3, n_features=40, n_labels=8, n_repeats=1, seed=1)
    m = r.groupby("cv")["AUROC"].mean()
    assert m["each chip independent"] > m["grouped by compound"] + 0.1
    assert abs(m["grouped by compound"] - 0.5) < 0.12  # noise labels stay near chance when folds respect compounds


def test_planning_curve_recovers_inverse_sqrt_scaling():
    rng = np.random.default_rng(0)
    n1 = rng.integers(10, 200, 200)
    n0 = rng.integers(10, 400, 200)
    n_eff = 4 / (1 / n1 + 1 / n0)
    d = pd.DataFrame({"n_active": n1, "n_inactive": n0, "min_detectable_effect": 2.5 / np.sqrt(n_eff) * np.exp(rng.normal(0, 0.1, 200))})
    c = fit_planning_curve(d)
    assert c["b"] == pytest.approx(-0.5, abs=0.05) and c["r2"] > 0.9
    mid, lo, hi = planned_detectable_effect(40, 0.5, c)
    assert lo < mid < hi and mid > planned_detectable_effect(400, 0.5, c)[0]  # more compounds detect smaller effects


def test_planned_detectable_effect_needs_both_classes():
    c = dict(a=0.0, b=-0.5, resid_sd=0.1)
    assert np.isnan(planned_detectable_effect(30, 0.0, c)[0])


def test_summarise_study_counts_indeterminate_and_credit():
    rows = []
    for kind, v in (("null", "no_advantage"), ("null", "morphology_advantage"), ("adds", "morphology_advantage"), ("adds", "no_advantage")):
        rows.append(dict(n_compounds=40, n_active=20, n_inactive=20, powered_chip=True, kind=kind, verdict_raw=v, verdict_calibrated=v,
                         min_detectable_effect=0.2, null_sd=1.3))
    rows.append(dict(n_compounds=40, n_active=4, n_inactive=36, powered_chip=False, kind="null", verdict_raw="indeterminate",
                     verdict_calibrated="indeterminate", min_detectable_effect=np.nan, null_sd=1.3))
    s = summarise_study(pd.DataFrame(rows)).iloc[0]
    assert s.indeterminate_strict == pytest.approx(0.2) and s.indeterminate_chip == pytest.approx(0.2)
    assert s.credited_calibrated_on_null == 0.5 and s.power_calibrated == 0.5


def test_chip_scale_study_runs_on_a_tiny_grid():
    t = chip_scale_study(sizes=(40,), seeds=(0,), endpoints={"null": 4, "adds": 4}, n_null_runs=6, n_repeats=1, n_boot=60, n_jobs=1)
    assert {"verdict_raw", "verdict_calibrated", "min_detectable_effect", "n_compounds"} <= set(t.columns)
    assert (t.n_compounds == 40).all() and len(t) == 8
