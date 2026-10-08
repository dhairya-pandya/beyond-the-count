import numpy as np
import pandas as pd

from cpsa.models.train_eval import auroc, oof_predict
from cpsa.nested import calibrate_nested, estimate_nested_null, nested_columns, nested_one
from cpsa.simulate import make_dataset


def _table(sim):
    base = sim.baseline.copy()
    base.columns = ["Cell_Count"] + [f"cc_{i}" for i in range(1, base.shape[1])]
    return pd.concat([base, sim.profile], axis=1)


def test_nested_columns_are_profile_plus_count():
    sim = make_dataset(n=120, n_features=30, endpoints={"null": 1}, seed=0)
    t = _table(sim)
    cols = nested_columns(t)
    assert set(cols) == set(t.columns) and "Cell_Count" in cols and "f000" in cols


def test_logreg_oof_is_informative_and_covers_every_row():
    rng = np.random.default_rng(0)
    n = 200
    y = (rng.random(n) < 0.3).astype(int)
    X = pd.DataFrame(rng.normal(size=(n, 20)))
    X[0] += 1.5 * y
    p = oof_predict(X, y, np.arange(n), 5, 1, 0, model="logreg")
    assert p.shape == (1, n) and np.isfinite(p).all() and auroc(y, p[0]) > 0.7


def test_nested_detects_morphology_beyond_count_and_not_a_count_only_endpoint():
    sim = make_dataset(n=400, n_features=60, endpoints={"shortcut": 2, "morph_only": 2}, prevalence=(0.3,), seed=1)
    t = _table(sim)
    kinds = sim.truth.set_index("endpoint_id")["kind"]
    res = {e: nested_one(e, t, sim.labels, n_repeats=1, n_boot=200, seed=0) for e in kinds.index}
    morph = [res[e]["delta_nested"] for e in kinds.index if kinds[e] == "morph_only"]
    short = [res[e]["delta_nested"] for e in kinds.index if kinds[e] == "shortcut"]
    assert min(morph) > 0.05 and max(short) < 0.05


def test_calibrate_nested_widens_p_values_and_marks_underpowered():
    rows = pd.DataFrame({
        "endpoint_id": ["a", "b", "c"], "n_active": [40, 40, 5], "n_inactive": [200, 200, 200],
        "delta_nested": [0.15, 0.01, 0.2], "delta_nested_ci_lo": [0.08, -0.05, 0.0], "delta_nested_ci_hi": [0.22, 0.07, 0.4],
        "bootstrap_p_nested": [0.001, 0.4, 0.02],
    })
    narrow = calibrate_nested(rows, null_sd=1.0)
    wide = calibrate_nested(rows, null_sd=2.0)
    assert narrow.loc[0, "verdict_nested"] == "adds_beyond_count" and narrow.loc[1, "verdict_nested"] == "no_detectable_addition"
    assert narrow.loc[2, "verdict_nested"] == "indeterminate"
    assert wide.loc[0, "calibrated_p_nested"] > narrow.loc[0, "calibrated_p_nested"]


def test_estimate_nested_null_uses_only_powered_runs():
    rng = np.random.default_rng(0)
    d = rng.normal(0, 0.03, 60)
    null = pd.DataFrame({"n_active": 30, "n_inactive": 100, "delta_nested": d, "delta_nested_ci_lo": d - 0.05, "delta_nested_ci_hi": d + 0.05,
                         "bootstrap_p_nested": rng.random(60)})
    null.loc[:9, "n_active"] = 3  # under-powered runs are ignored
    est = estimate_nested_null(null)
    assert est["n_runs"] == 50 and est["sd"] > 0
