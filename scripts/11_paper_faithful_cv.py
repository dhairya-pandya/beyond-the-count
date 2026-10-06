#!/usr/bin/env python
"""Reproduction check: re-run the full and scalar-cell-count models with the SOURCE PAPER's protocol (unshuffled
StratifiedKFold on the aggregated-profile row order, single run, pooled AUROC) and compare with the paper's compiled
per-endpoint AUROCs. Shows how much of the gap between our pipeline and the paper is CV protocol (shuffling + repeats)
rather than data processing.

Row order replicated: for ``allpod`` the paper concatenates compounds that have wells above their POD (sorted by id) with the rest.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cpsa import RESULTS  # noqa: E402
from cpsa.data.build_dataset import BASELINE_SCALAR, build_compound_table, load_pods, profile_columns  # noqa: E402
from cpsa.data.load_labels import load_all_labels  # noqa: E402
from cpsa.models.train_eval import auroc, oof_predict  # noqa: E402
from importlib import import_module  # noqa: E402

paper_metrics = import_module("06_compare_to_paper").paper_metrics


def paper_order(table: pd.DataFrame, agg: str, feature_set: str) -> list[str]:
    if agg != "allpod":
        return sorted(table.index)
    pod_cp, _ = load_pods(feature_set)
    first = sorted(i for i in table.index if i in pod_cp.index)
    return first + sorted(i for i in table.index if i not in pod_cp.index)


def one(e, table, labels, order, cols_full):
    y = labels[e].dropna()
    ids = [i for i in order if i in y.index]
    yv = y.loc[ids].astype(int).to_numpy()
    k = 5
    out = {"endpoint_id": e}
    for name, cols in (("full", cols_full), ("scalar_cc", BASELINE_SCALAR)):
        p = oof_predict(table.loc[ids, cols], yv, np.asarray(ids), k, 1, 0, shuffle=False)[0]
        out[f"pf_{name}_AUROC"] = auroc(yv, p)
    return out


def main() -> None:
    sys.path.insert(0, str(Path(__file__).parent))
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="cellprofiler_allpod")
    ap.add_argument("--n-jobs", type=int, default=2)
    a = ap.parse_args()
    fs, agg = a.config.split("_", 1)
    table = build_compound_table(fs, agg)
    labels, _ = load_all_labels()
    audit = pd.read_csv(RESULTS / a.config / "audit_table.csv").set_index("endpoint_id")
    todo = audit.index[audit["powered"] & (audit["verdict"] != "positive_control") & (audit["category"] != "native")].tolist()
    order = paper_order(table, agg, fs)
    rows = Parallel(n_jobs=a.n_jobs)(delayed(one)(e, table, labels, order, profile_columns(table)) for e in todo)
    df = pd.DataFrame(rows).set_index("endpoint_id").join(audit[["full_AUROC", "scalar_cc_AUROC"]]).join(paper_metrics(fs, agg)).dropna(subset=["paper_Actual_AUROC"])
    df.to_csv(RESULTS / a.config / "paper_faithful_cv.csv")
    for tag, c_ours, c_paper in (("full", "pf_full_AUROC", "paper_Actual_AUROC"), ("scalar_cc", "pf_scalar_cc_AUROC", "paper_Cellcount_baseline_AUROC")):
        print(f"{tag}: paper-protocol vs paper  r={pearsonr(df[c_ours], df[c_paper])[0]:.3f}  mean diff={(df[c_ours] - df[c_paper]).mean():+.3f} | "
              f"our default protocol vs paper  r={pearsonr(df[tag + '_AUROC'], df[c_paper])[0]:.3f}  mean diff={(df[tag + '_AUROC'] - df[c_paper]).mean():+.3f}   (n={len(df)})")


if __name__ == "__main__":
    main()
