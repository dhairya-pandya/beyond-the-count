"""Label semantics (teammate's notes on the ToxCast hit-calls): missing is NOT zero; 0 means 'not a (filtered) hit', not 'tested negative'."""
import numpy as np
import pandas as pd
import pytest

from cpsa import EXTERNAL
from cpsa.audit import _class_counts
from cpsa.data.build_dataset import build_endpoint_dataset
from cpsa.data.label_context import filter_applicable, stratum_counts


def _labels():
    return pd.DataFrame({"e1": [1.0, 0.0, np.nan, np.nan, 1.0, 0.0]}, index=[f"C{i}" for i in range(6)])


def _table():
    return pd.DataFrame({"Cell_Count": np.arange(6.0), "f1": np.arange(6.0)}, index=[f"C{i}" for i in range(6)])


def test_untested_pairs_are_dropped_not_zero_filled():
    d = build_endpoint_dataset(_table(), _labels(), "e1")
    assert list(d.index) == ["C0", "C1", "C4", "C5"] and d["label"].tolist() == [1, 0, 1, 0]


def test_class_counts_ignore_missing_labels():
    n1, n0 = _class_counts(_labels(), _table())
    assert n1["e1"] == 2 and n0["e1"] == 2  # NaN counts toward neither class


def test_filter_applicable_follows_the_paper_rule_20pct_fired_and_median_ac50_at_most_100uM():
    info = pd.DataFrame({
        "OASIS_ID": ["A", "B", "C", "D", "E"], "assay_component_endpoint_name": ["x"] * 5,
        "cytotox_ntested": [10, 10, 10, 10, 10], "cytotox_nhit": [2, 1, 5, 5, 0],
        "cytotox_median_ac50": [50.0, 50.0, 150.0, np.nan, np.nan],
    })
    f = filter_applicable(info)
    assert f.loc[("A", "x")] and not f.loc[("B", "x")]          # 20% fired ok / only 10% fired
    assert not f.loc[("C", "x")] and not f.loc[("D", "x")]      # AC50 above 100 uM / no AC50
    assert not f.loc[("E", "x")]


def test_stratum_counts_report_class_sizes_per_stratum():
    y = pd.Series([1, 1, 0, 0, 0, 1], index=list("abcdef"))
    flag = pd.Series([True, True, True, False, False, False], index=y.index)
    c = stratum_counts(y, flag)
    assert c == {"applies": (2, 1), "not_applicable": (1, 2)}


@pytest.mark.skipif(not (EXTERNAL / "ewald_repo").exists(), reason="authors' repo not downloaded")
def test_real_cellbased_matrix_keeps_missing_missing_and_matches_the_documented_counts():
    from cpsa.data.load_labels import load_toxcast_binary
    b = load_toxcast_binary("cellbased")
    assert b.shape == (963, 292)
    assert int(b.notna().sum().sum()) == 107103          # 61.9 % of cells are untested and must stay NaN
    assert set(np.unique(b.to_numpy()[~np.isnan(b.to_numpy())])) <= {0.0, 1.0}
    assert int((b == 1).sum().sum()) == 8400             # positives after the cytotox filter
