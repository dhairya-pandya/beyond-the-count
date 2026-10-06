import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from cpsa.models.train_eval import auroc, evaluate_endpoint, oof_predict, paired_bootstrap_delta


def test_auroc_matches_sklearn_with_ties():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    s = np.round(rng.normal(size=200) + y, 1)  # many ties
    assert auroc(y, s) == pytest.approx(roc_auc_score(y, s))


def test_bootstrap_detects_clear_improvement_and_null():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 300)
    good = y + rng.normal(scale=0.7, size=300)
    bad = rng.normal(size=300)
    res = paired_bootstrap_delta(y, good, bad, n_boot=500, seed=0)
    assert res["delta"] > 0.2 and res["p"] < 0.01 and res["ci_lo"] > 0
    same = paired_bootstrap_delta(y, good, good, n_boot=200, seed=0)
    assert same["delta"] == 0 and same["p"] > 0.5  # identical scores -> no evidence of improvement


def _toy(n=240, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.3).astype(int)
    X = pd.DataFrame(rng.normal(size=(n, 5)), columns=list("abcde"))
    X["a"] += 1.5 * y  # informative
    ids = [f"C{i}" for i in range(n)]
    return X.set_index(pd.Index(ids)), pd.Series(y, index=ids)


def test_oof_predict_covers_every_row_once_per_repeat_and_is_deterministic():
    X, y = _toy()
    p1 = oof_predict(X, y.to_numpy(), groups=X.index.to_numpy(), n_splits=5, n_repeats=2, seed=3)
    p2 = oof_predict(X, y.to_numpy(), groups=X.index.to_numpy(), n_splits=5, n_repeats=2, seed=3)
    assert p1.shape == (2, len(X)) and np.isfinite(p1).all()
    np.testing.assert_allclose(p1, p2)
    assert roc_auc_score(y, p1.mean(0)) > 0.75  # Bayes-optimal here is ~0.855 (1.5 sd shift, one feature)


def test_oof_predict_never_trains_on_a_heldout_group():
    # duplicate each compound 3x (as if replicate wells); a leak would let the model memorise the duplicate
    rng = np.random.default_rng(0)
    n = 120
    y1 = rng.integers(0, 2, n)
    X1 = rng.normal(size=(n, 20))  # pure noise features
    X = pd.DataFrame(np.repeat(X1, 3, axis=0)); y = np.repeat(y1, 3); g = np.repeat(np.arange(n), 3)
    p = oof_predict(X, y, groups=g, n_splits=5, n_repeats=1, seed=0)[0]
    assert abs(roc_auc_score(y, p) - 0.5) < 0.15  # grouped CV -> no memorisation of noise


def test_evaluate_endpoint_reports_informative_beats_uninformative_baseline():
    X, y = _toy(n=300)
    res = evaluate_endpoint(
        X, y, models={"full": ["a", "b", "c", "d", "e"], "baseline": ["e"]},
        n_splits=5, n_repeats=2, n_boot=200, seed=0,
    )
    assert res["metrics"]["full"]["auroc"] > res["metrics"]["baseline"]["auroc"] + 0.2
    assert res["delta"]["baseline"]["p"] < 0.05
    assert set(res["oof"]) == {"full", "baseline"}


def test_unshuffled_cv_matches_paper_protocol_contiguous_stratified_folds():
    X, y = _toy(n=100)
    p = oof_predict(X, y.to_numpy(), groups=X.index.to_numpy(), n_splits=5, n_repeats=1, seed=0, shuffle=False)
    q = oof_predict(X, y.to_numpy(), groups=X.index.to_numpy(), n_splits=5, n_repeats=1, seed=99, shuffle=False)
    np.testing.assert_allclose(p, q)  # unshuffled folds do not depend on the seed's split (model seed only)
    with pytest.raises(ValueError):
        oof_predict(X, y.to_numpy(), groups=np.repeat(np.arange(50), 2), n_splits=5, n_repeats=1, seed=0, shuffle=False)


def test_floored_reference_ignores_below_chance_baseline():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 400)
    full = y + rng.normal(scale=1.5, size=400)       # weak but real signal
    anti = -(y + rng.normal(scale=1.5, size=400))    # baseline that is anti-predictive (AUROC well below 0.5)
    plain = paired_bootstrap_delta(y, full, anti, n_boot=300, seed=0)
    floored = paired_bootstrap_delta(y, full, anti, n_boot=300, seed=0, floor=0.5)
    assert floored["delta"] < plain["delta"]
    assert floored["delta"] == pytest.approx(auroc(y, full) - 0.5)


def test_bootstrap_rejects_single_class_input_instead_of_looping_forever():
    y = np.ones(50, dtype=int)
    with pytest.raises(ValueError):
        paired_bootstrap_delta(y, np.arange(50.0), np.arange(50.0)[::-1], n_boot=10)
