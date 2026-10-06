"""Context needed to read the ToxCast labels correctly (from the OASIS repo's 3B_extract_invitrodb notebook and the authors' Methods).

* A cell-based label is a hit (hitcall > 0.9) that is *set to 0* if the endpoint AC50 exceeds half of the matched consensus cytotoxicity AC50.
* That consensus exists only if >= 20 % of the compound's matched viability tests fired and it is <= 100 uM; elsewhere the filter cannot apply,
  so those labels are NOT cytotoxicity-adjusted and a cell-count shortcut can reach them directly.
* 0 therefore means 'not a (filtered) hit', never 'tested and negative'; untested pairs stay NaN.
The labels themselves always come from the repo's ``*_binary.parquet`` files; the ``*_info`` files are used here ONLY to ask whether the filter could
have applied to a compound-endpoint pair (their ``hitcall`` column is the raw continuous value and is never used as a label).
"""
from __future__ import annotations

import pandas as pd

from cpsa.data.load_labels import ANNOT

MIN_FRACTION_FIRED = 0.2
MAX_CYTOTOX_AC50_UM = 100.0


def filter_applicable(info: pd.DataFrame) -> pd.Series:
    """Bool per (OASIS_ID, endpoint): could the cytotoxicity filter apply to this pair (consensus cytotox AC50 exists)?"""
    d = info.drop_duplicates(["OASIS_ID", "assay_component_endpoint_name"]).set_index(["OASIS_ID", "assay_component_endpoint_name"])
    fired = d["cytotox_nhit"] / d["cytotox_ntested"].where(d["cytotox_ntested"] > 0)
    ok = (fired >= MIN_FRACTION_FIRED) & d["cytotox_median_ac50"].notna() & (d["cytotox_median_ac50"] <= MAX_CYTOTOX_AC50_UM)
    return ok.rename("filter_applicable")


def load_cellbased_filter_flags() -> pd.Series:
    return filter_applicable(pd.read_parquet(ANNOT / "toxcast_cellbased_info.parquet", columns=[
        "OASIS_ID", "assay_component_endpoint_name", "cytotox_ntested", "cytotox_nhit", "cytotox_median_ac50"]))


def stratum_counts(y: pd.Series, flag: pd.Series) -> dict[str, tuple[int, int]]:
    """(n_active, n_nonhit) among the compounds of ``y`` for pairs where the filter applies / cannot apply."""
    f = flag.reindex(y.index).fillna(False).astype(bool)
    return {"applies": (int((y[f] == 1).sum()), int((y[f] == 0).sum())), "not_applicable": (int((y[~f] == 1).sum()), int((y[~f] == 0).sum()))}
