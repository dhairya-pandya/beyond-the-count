import numpy as np
import pytest

from cpsa.stats.calibration import calibration_summary


def test_perfectly_calibrated_probabilities_have_low_ece_and_positive_skill():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.02, 0.98, 5000)
    y = (rng.random(5000) < p).astype(int)
    c = calibration_summary(y, p)
    assert c["ece"] < 0.03 and c["brier_skill"] > 0.2
    assert c["brier"] == pytest.approx(np.mean((p - y) ** 2))


def test_overconfident_probabilities_are_flagged_and_platt_fixes_them():
    rng = np.random.default_rng(1)
    true_p = rng.uniform(0.2, 0.8, 4000)
    y = (rng.random(4000) < true_p).astype(int)
    logit = np.log(true_p / (1 - true_p)) * 3  # 3x too confident
    p = 1 / (1 + np.exp(-logit))
    c = calibration_summary(y, p)
    assert c["ece"] > 0.1
    assert c["brier_recalibrated"] < c["brier"] - 0.02
    assert c["slope"] < 0.6  # calibration slope << 1 means overconfident


def test_reliability_bins_cover_all_samples():
    rng = np.random.default_rng(2)
    p = rng.random(300); y = (rng.random(300) < p).astype(int)
    bins = calibration_summary(y, p, n_bins=10)["bins"]
    assert sum(b["n"] for b in bins) == 300


def test_platt_oof_reduces_overconfidence_and_keeps_ranking():
    from cpsa.stats.calibration import platt_oof
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(4)
    true_p = rng.uniform(0.2, 0.8, 3000)
    y = (rng.random(3000) < true_p).astype(int)
    p = 1 / (1 + np.exp(-3 * np.log(true_p / (1 - true_p))))
    q = platt_oof(y, p, seed=0)
    assert np.mean((q - y) ** 2) < np.mean((p - y) ** 2) - 0.02
    assert roc_auc_score(y, q) == pytest.approx(roc_auc_score(y, p), abs=0.02)
