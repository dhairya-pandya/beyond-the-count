import numpy as np
import pandas as pd
import pytest

from cpsa.stats.signal_class import classify_signal, null_auroc_thresholds


def _null(n=400, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"full_AUROC": rng.normal(0.5, 0.1, n), "strong_cc_AUROC": rng.normal(0.5, 0.08, n), "n_active": 30, "n_inactive": 100})


def test_null_thresholds_are_the_95th_percentile_of_powered_null_runs():
    null = _null()
    junk = pd.DataFrame({"full_AUROC": [0.99] * 50, "strong_cc_AUROC": [0.99] * 50, "n_active": 5, "n_inactive": 100})  # underpowered: ignored
    thr = null_auroc_thresholds(pd.concat([null, junk], ignore_index=True))
    assert thr["strong"] == pytest.approx(0.5 + 1.645 * 0.08, abs=0.03)
    assert thr["full"] == pytest.approx(0.5 + 1.645 * 0.10, abs=0.03)


def test_classes_separate_count_explained_from_no_signal():
    a = pd.DataFrame({
        "verdict": ["morphology_advantage", "no_advantage", "no_advantage", "no_advantage", "indeterminate", "positive_control"],
        "strong_cc_AUROC": [0.80, 0.85, 0.52, 0.50, 0.50, 1.0],
        "full_AUROC": [0.90, 0.86, 0.75, 0.52, 0.99, 0.99],
    })
    out = classify_signal(a, thr_full=0.66, thr_strong=0.63)
    assert out.tolist() == ["morphology_advantage", "count_explained", "morphology_signal_uncertified", "no_detectable_signal",
                            "indeterminate", "positive_control"]
