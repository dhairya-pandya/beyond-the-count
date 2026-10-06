import numpy as np
import pandas as pd
import pytest

from cpsa.stats.multiple_testing import assign_verdicts, bh_fdr
from cpsa.stats.power_check import power_flag


def test_bh_fdr_matches_hand_computation():
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216])
    q = bh_fdr(p)
    # classic BH: q_i = min_j>=i p_j * m / j
    assert q[0] == pytest.approx(0.01)
    assert q[1] == pytest.approx(0.04)
    assert q[2] == pytest.approx(0.084)  # raw 0.13, pulled down by the cumulative minimum from rank 5
    assert np.all(np.diff(q) >= -1e-12)  # monotone in sorted p


def test_bh_fdr_ignores_nan_and_preserves_positions():
    q = bh_fdr(np.array([0.5, np.nan, 0.01]))
    assert np.isnan(q[1]) and q[2] < q[0]


def test_power_flag_threshold():
    assert power_flag(n_active=15, n_inactive=15) is True
    assert power_flag(n_active=14, n_inactive=500) is False
    assert power_flag(n_active=500, n_inactive=14) is False


def test_indeterminate_endpoints_excluded_from_fdr_pool():
    # one hugely significant powered endpoint + many indeterminate ones with tiny p
    df = pd.DataFrame({
        "n_active": [30] + [5] * 20, "n_inactive": [100] + [100] * 20,
        "delta_auroc": [0.2] + [0.3] * 20, "bootstrap_p": [0.001] + [0.0005] * 20,
    })
    out = assign_verdicts(df, alpha=0.05, min_active=15, min_inactive=15)
    assert (out["verdict"].iloc[1:] == "indeterminate").all()
    assert out["fdr_q"].iloc[1:].isna().all()
    assert out["fdr_q"].iloc[0] == pytest.approx(0.001)  # pool size 1, not 21
    assert out["verdict"].iloc[0] == "morphology_advantage"


def test_verdict_requires_positive_delta_and_q_below_alpha():
    df = pd.DataFrame({
        "n_active": [30, 30, 30, 30], "n_inactive": [100] * 4,
        "delta_auroc": [0.1, -0.1, 0.1, 0.01], "bootstrap_p": [0.001, 0.001, 0.6, 0.049],
    })
    out = assign_verdicts(df, alpha=0.05, min_active=15, min_inactive=15)
    assert out["verdict"].tolist() == ["morphology_advantage", "no_advantage", "no_advantage", "no_advantage"]
