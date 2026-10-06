"""Split 'no advantage' into *why* morphology did not win: cell count already solves the endpoint, or nobody predicts it.

Descriptive classes (not hypothesis tests). An AUROC counts as 'informative' if it exceeds the 95th percentile of the AUROCs that the same
model obtains on label-permuted endpoints (the permutation control), so the cut-off reflects cross-validation noise at this sample size.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CLASSES = ("morphology_advantage", "count_explained", "morphology_signal_uncertified", "no_detectable_signal", "indeterminate", "positive_control")


def null_auroc_thresholds(null: pd.DataFrame, q: float = 0.95, min_active: int = 15, min_inactive: int = 15) -> dict:
    n = null[(null["n_active"] >= min_active) & (null["n_inactive"] >= min_inactive)]
    return {"full": float(np.quantile(n["full_AUROC"], q)), "strong": float(np.quantile(n["strong_cc_AUROC"], q)), "q": q, "n_null_runs": int(len(n))}


def classify_signal(audit: pd.DataFrame, thr_full: float, thr_strong: float) -> pd.Series:
    """Per-endpoint class (see CLASSES). Needs columns verdict, strong_cc_AUROC, full_AUROC."""
    v = audit["verdict"]
    cc_informative = audit["strong_cc_AUROC"] > thr_strong
    full_informative = audit["full_AUROC"] > thr_full
    out = pd.Series(np.select(
        [v == "morphology_advantage", v == "indeterminate", v == "positive_control", cc_informative, full_informative],
        ["morphology_advantage", "indeterminate", "positive_control", "count_explained", "morphology_signal_uncertified"],
        default="no_detectable_signal"), index=audit.index)
    return out
