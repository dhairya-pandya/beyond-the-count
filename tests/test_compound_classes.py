import numpy as np
import pandas as pd
import pytest

from cpsa.biology.compound_classes import class_table, compound_benefit, ranking_accuracy
from cpsa.models.train_eval import auroc


def test_ranking_accuracy_decomposes_auroc():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 120)
    s = np.round(rng.normal(size=120) + 0.8 * y, 1)  # ties on purpose
    r = ranking_accuracy(y, s)
    assert r[y == 1].mean() == pytest.approx(auroc(y, s))
    assert r[y == 0].mean() == pytest.approx(auroc(y, s))


def _oof():
    rng = np.random.default_rng(1)
    rows = []
    for e in range(6):
        y = rng.integers(0, 2, 200)
        good = y + rng.normal(scale=0.8, size=200)
        weak = rng.normal(size=200)
        for i in range(200):
            # compounds C000..C049 are 'special': morphology scores them much better than the baseline does
            full = good[i] + (2 * y[i] if i < 50 else 0)
            rows.append(dict(endpoint_id=f"e{e}", OASIS_ID=f"C{i:03d}", y=y[i], p_full=full, p_strong_cc=weak[i]))
    return pd.DataFrame(rows)


def test_compound_benefit_is_higher_for_compounds_morphology_handles_better_and_class_table_flags_them():
    ben = compound_benefit(_oof(), [f"e{e}" for e in range(6)], base_col="p_strong_cc")
    assert ben.loc[[f"C{i:03d}" for i in range(50)], "benefit"].mean() > ben.loc[[f"C{i:03d}" for i in range(50, 200)], "benefit"].mean() + 0.05
    groups = pd.Series(["special"] * 50 + ["other"] * 150, index=[f"C{i:03d}" for i in range(200)])
    t = class_table(ben, groups, min_n=10, n_boot=200).set_index("group")
    assert t.loc["special", "mean_benefit"] > t.loc["other", "mean_benefit"]
    assert t.loc["special", "q"] < 0.05 and t.loc["special", "n_compounds"] == 50
