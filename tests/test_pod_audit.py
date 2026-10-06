import numpy as np
import pandas as pd
import pytest

from cpsa.stats.pod_audit import paired_fold


def test_paired_fold_is_geometric_mean_ratio_over_compounds_with_both_pods():
    a = pd.Series({"c1": 10.0, "c2": 100.0, "c3": 5.0, "only_a": 1.0})
    b = pd.Series({"c1": 1.0, "c2": 10.0, "c3": 0.5, "only_b": 1.0})
    r = paired_fold(a, b, n_boot=200, seed=0)
    assert r["n_pairs"] == 3 and r["fold"] == pytest.approx(10.0)
    assert r["ci_lo"] <= 10.0 + 1e-9 and r["ci_hi"] >= 10.0 - 1e-9  # degenerate interval (all ratios equal); tolerate float rounding


def test_fold_below_one_means_the_first_assay_is_more_sensitive():
    rng = np.random.default_rng(0)
    b = pd.Series(10 ** rng.normal(0, 0.5, 80), index=[f"c{i}" for i in range(80)])
    a = b * 0.25 * 10 ** rng.normal(0, 0.1, 80)
    r = paired_fold(a, b, n_boot=300, seed=0)
    assert r["fold"] == pytest.approx(0.25, rel=0.1) and r["ci_hi"] < 0.3
