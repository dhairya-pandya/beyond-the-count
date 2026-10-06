import numpy as np
import pandas as pd
import pytest

from cpsa.data.preprocess import aggregate_compounds, cell_count_curve_features, mad_normalize


def _wells():
    rows = []
    for plate in ("p1", "p2"):
        for i in range(6):  # DMSO controls: feature f1 ~ 10 (+/- plate shift), spread 1
            rows.append(dict(Metadata_Plate=plate, Metadata_Compound="DMSO", Metadata_OASIS_ID=None,
                             Metadata_Concentration=0.0, Metadata_Count_Cells=1000,
                             f1=10 + (5 if plate == "p2" else 0) + (i - 2.5) * 0.4, f2=1.0))
    # compound A: 4 concentrations x 2 plates
    for plate in ("p1", "p2"):
        for conc, cnt in zip([1, 10, 100, 1000], [1000, 900, 400, 50]):
            rows.append(dict(Metadata_Plate=plate, Metadata_Compound="A", Metadata_OASIS_ID="OASIS_A",
                             Metadata_Concentration=float(conc), Metadata_Count_Cells=cnt,
                             f1=10 + (5 if plate == "p2" else 0) + np.log10(conc), f2=1.0))
    return pd.DataFrame(rows)


def test_mad_normalize_removes_plate_shift_and_constant_features():
    raw = _wells()
    norm, feats = mad_normalize(raw, ["f1", "f2"])
    assert feats == ["f1"]  # f2 has zero MAD in controls -> dropped
    ctrl = norm[norm.Metadata_Compound == "DMSO"]
    assert ctrl.groupby("Metadata_Plate")["f1"].median().abs().max() < 1e-9


def test_aggregate_all_is_mean_over_treated_wells_only():
    norm, feats = mad_normalize(_wells(), ["f1", "f2"])
    agg = aggregate_compounds(norm, feats, method="all", pod_cp=None, pod_cc=None)
    assert list(agg.index) == ["OASIS_A"]
    assert agg.loc["OASIS_A", "Cell_Count"] == pytest.approx(np.mean([1000, 900, 400, 50]))


def test_aggregate_allpod_uses_wells_above_pod_and_falls_back_to_all():
    norm, feats = mad_normalize(_wells(), ["f1"])
    pod_cp = pd.Series({"OASIS_A": 50.0})
    agg = aggregate_compounds(norm, feats, "allpod", pod_cp=pod_cp, pod_cc=None)
    assert agg.loc["OASIS_A", "Cell_Count"] == pytest.approx(np.mean([400, 50]))
    agg2 = aggregate_compounds(norm, feats, "allpod", pod_cp=pd.Series(dtype=float), pod_cc=None)
    assert agg2.loc["OASIS_A", "Cell_Count"] == pytest.approx(np.mean([1000, 900, 400, 50]))


def test_aggregate_allpodcc_window_and_min_conc_fallback():
    norm, feats = mad_normalize(_wells(), ["f1"])
    # window (pod_cp, pod_cc) = (5, 500): wells at conc 10 and 100
    agg = aggregate_compounds(norm, feats, "allpodcc", pod_cp=pd.Series({"OASIS_A": 5.0}), pod_cc=pd.Series({"OASIS_A": 500.0}))
    assert agg.loc["OASIS_A", "Cell_Count"] == pytest.approx(np.mean([900, 400]))
    # empty window (pod_cc below first conc above pod_cp) -> first conc above pod_cp (conc 10)
    agg = aggregate_compounds(norm, feats, "allpodcc", pod_cp=pd.Series({"OASIS_A": 5.0}), pod_cc=pd.Series({"OASIS_A": 6.0}))
    assert agg.loc["OASIS_A", "Cell_Count"] == pytest.approx(900)


def test_cell_count_curve_features_are_dmso_normalised_and_rank_ordered():
    cc = cell_count_curve_features(_wells())
    row = cc.loc["OASIS_A"]
    assert row["cc_r1"] == pytest.approx(1.0) and row["cc_r4"] == pytest.approx(0.05)
    assert row["cc_min"] == pytest.approx(0.05)
    assert row["cc_auc"] == pytest.approx(np.mean([1.0, 0.9, 0.4, 0.05]))


def test_cell_count_curve_features_are_concentration_aware_not_just_rank_aware():
    raw = _wells()  # compound A: conc 1, 10, 100, 1000 uM -> cell count ratio 1.0, 0.9, 0.4, 0.05
    low = raw[raw.Metadata_OASIS_ID == "OASIS_A"].copy()
    low["Metadata_OASIS_ID"], low["Metadata_Compound"] = "OASIS_LOW", "LOW"
    low["Metadata_Concentration"] = low["Metadata_Concentration"] / 100.0  # same curve on a 100x lower series: 0.01 ... 10 uM
    cc = cell_count_curve_features(pd.concat([raw, low], ignore_index=True))
    a, lo = cc.loc["OASIS_A"], cc.loc["OASIS_LOW"]
    assert a["cc_at_1uM"] == pytest.approx(1.0) and a["cc_at_10uM"] == pytest.approx(0.9) and a["cc_at_100uM"] == pytest.approx(0.4)
    assert np.isnan(a["cc_at_0.1uM"])                      # below the lowest tested concentration -> not measured, not zero
    assert lo["cc_r3"] == pytest.approx(a["cc_r3"])        # rank features cannot tell the two series apart ...
    assert lo["cc_at_0.1uM"] == pytest.approx(0.9) and lo["cc_at_10uM"] == pytest.approx(0.05)   # ... concentration features can
    assert np.isnan(lo["cc_at_100uM"])                     # 100 uM was never reached on the low series
