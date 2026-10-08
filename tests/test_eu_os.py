import numpy as np
import pandas as pd
import pytest

from cpsa.data.eu_os import COUNT, META, feature_columns, robust_normalize


def test_feature_columns_drop_metadata_position_and_site():
    cols = ["Metadata_Plate", "site", "Nuc_AreaShape_Area", "Nuc_AreaShape_Center_X", "Cells_AreaShape_BoundingBoxArea",
            "Nuc_Neighbors_FirstClosestObjectNumber_1", "Cyto_Intensity_MeanIntensity_DNA"]
    assert feature_columns(cols) == ["Nuc_AreaShape_Area", "Cyto_Intensity_MeanIntensity_DNA"]


def _plate(plate, shift, n_ctrl=12, n_cmp=6, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_ctrl + n_cmp):
        ctrl = i < n_ctrl
        rows.append({"Metadata_Batch": "R1", "Metadata_Plate": plate, "Metadata_Well": f"A{i:02d}", "Metadata_EOS": "DMSO" if ctrl else f"EOS{i}",
                     "site": "S", COUNT: 1000 * shift * (1 if ctrl else 0.5), "f1": shift + rng.normal(0, 1), "f2": 5 * shift + rng.normal(0, 2)})
    return pd.DataFrame(rows)


def test_robust_normalize_removes_plate_offsets_and_scales_cell_count():
    d = pd.concat([_plate("B1", 1.0, seed=0), _plate("B2", 10.0, seed=1)], ignore_index=True)
    n = robust_normalize(d, ["f1", "f2"])
    ctrl = n[n["Metadata_EOS"] == "DMSO"]
    for p in ("B1", "B2"):  # DMSO wells are centred on zero in every plate
        assert abs(ctrl.loc[ctrl.Metadata_Plate == p, "f1"].median()) < 1e-6
    assert n.loc[n["Metadata_EOS"] != "DMSO", "cell_count_ratio"].round(2).eq(0.5).all()  # half the DMSO cell count on both plates


def test_robust_normalize_skips_plates_without_enough_controls():
    d = _plate("B1", 1.0, n_ctrl=2)
    n = robust_normalize(d, ["f1", "f2"])
    assert n["f1"].isna().all() and n["cell_count_ratio"].isna().all()
