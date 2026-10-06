import numpy as np
import pandas as pd
import pytest

from cpsa.api import permutation_null, shortcut_audit


def _toy(n=300, p=15, seed=0):
    """count drives 'shortcut' labels; a hidden morphology factor drives 'adds' labels on top of count; 'null' is noise."""
    rng = np.random.default_rng(seed)
    count = rng.normal(size=n)
    z = rng.normal(size=n)
    morph = pd.DataFrame(rng.normal(size=(n, p)), columns=[f"m{i}" for i in range(p)])
    morph["m0"] = count + 0.3 * rng.normal(size=n)      # morphology also encodes density
    morph["m1"] = z + 0.3 * rng.normal(size=n)          # ... and a factor cell count cannot see
    base = pd.DataFrame({"count": count + 0.1 * rng.normal(size=n)})
    sig = lambda x: (rng.random(n) < 1 / (1 + np.exp(-2.5 * x))).astype(float)
    labels = pd.DataFrame({"shortcut": sig(count), "adds": sig(0.5 * count + 1.5 * z), "null": (rng.random(n) < 0.3).astype(float)})
    ids = [f"C{i}" for i in range(n)]
    for d in (morph, base, labels):
        d.index = ids
    return morph, base, labels


def test_shortcut_audit_credits_morphology_only_where_it_adds_information():
    morph, base, labels = _toy()
    res = shortcut_audit(morph, base, labels, n_repeats=2, n_boot=300, min_active=15, min_inactive=15)
    t = res.table.set_index("endpoint_id")
    assert t.loc["adds", "verdict"] == "morphology_advantage"
    assert t.loc["shortcut", "verdict"] == "no_advantage"
    assert t.loc["shortcut", "baseline_AUROC"] > 0.7
    assert t.loc["null", "full_AUROC"] < 0.62


def test_underpowered_endpoint_is_indeterminate_and_outside_fdr_pool():
    morph, base, labels = _toy()
    labels["rare"] = 0.0
    labels.iloc[:6, labels.columns.get_loc("rare")] = 1.0
    t = shortcut_audit(morph, base, labels, n_repeats=1, n_boot=100).table.set_index("endpoint_id")
    assert t.loc["rare", "verdict"] == "indeterminate" and np.isnan(t.loc["rare", "fdr_q"])


def test_calibration_with_a_larger_null_sd_widens_p_values_and_keeps_raw_columns():
    morph, base, labels = _toy()
    one = shortcut_audit(morph, base, labels[["adds"]], n_repeats=1, n_boot=200, null_sd=1.0).table.iloc[0]
    two = shortcut_audit(morph, base, labels[["adds"]], n_repeats=1, n_boot=200, null_sd=2.0).table.iloc[0]
    assert two["calibrated_p"] > one["calibrated_p"]
    assert two["bootstrap_p"] == pytest.approx(one["bootstrap_p"])  # raw bootstrap p untouched by calibration (it is resolution-limited at 1/(B+1))


def test_permutation_null_returns_one_z_per_run_with_roughly_zero_centre():
    morph, base, labels = _toy()
    z = permutation_null(morph, base, labels[["shortcut", "adds"]], n_runs=6, n_repeats=1, n_boot=100, seed=1)
    assert len(z) == 6 and abs(z["z"].mean()) < 1.5
