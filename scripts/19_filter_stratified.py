#!/usr/bin/env python
"""Does the cell-count shortcut depend on whether the ToxCast cytotoxicity filter could apply to a label?

Cell-based labels are hits set to 0 when the endpoint AC50 exceeds half the matched consensus cytotoxicity AC50 -- but that consensus exists for
only ~35 % of compound-endpoint pairs. Where it cannot apply, cytotoxic compounds can remain 'active', which a cell-count model can exploit.
Re-scores the saved out-of-fold predictions separately for the two strata (no retraining): per endpoint and stratum, AUROC of the scalar / strong
cell-count baselines and of the full model. Labels always come from the *_binary parquet; the info file only supplies the stratum flag.
-> results/<config>/filter_stratified.csv, filter_stratified_summary.csv
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cpsa import RESULTS  # noqa: E402
from cpsa.data.label_context import load_cellbased_filter_flags  # noqa: E402
from cpsa.models.train_eval import auroc  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="cellprofiler_allpod")
    ap.add_argument("--min-class", type=int, default=10, help="min actives and non-hits within a stratum")
    a = ap.parse_args()
    d = RESULTS / a.config
    audit = pd.read_csv(d / "audit_table.csv")
    oof = pd.read_parquet(d / "oof_predictions.parquet")
    flags = load_cellbased_filter_flags()
    eps = audit.loc[(audit["category"] == "cellbased") & audit["delta_AUROC"].notna(), "endpoint_id"]
    rows = []
    for e, g in oof[oof["endpoint_id"].isin(set(eps))].groupby("endpoint_id"):
        f = pd.Series([bool(flags.get((o, e), False)) for o in g["OASIS_ID"]], index=g.index)
        for name, m in (("applies", f), ("not_applicable", ~f)):
            s = g[m]
            n1, n0 = int((s["y"] == 1).sum()), int((s["y"] == 0).sum())
            r = dict(endpoint_id=e, stratum=name, n_active=n1, n_nonhit=n0, active_rate=n1 / max(n1 + n0, 1))
            if n1 >= a.min_class and n0 >= a.min_class:
                for col, lab in (("p_scalar_cc", "scalar_cc"), ("p_strong_cc", "strong_cc"), ("p_full", "full")):
                    r[f"{lab}_AUROC"] = auroc(s["y"].to_numpy(), s[col].to_numpy())
            rows.append(r)
    t = pd.DataFrame(rows)
    t.to_csv(d / "filter_stratified.csv", index=False)
    ok = t.dropna(subset=["strong_cc_AUROC"]) if "strong_cc_AUROC" in t else t.iloc[0:0]
    pair = ok.pivot(index="endpoint_id", columns="stratum", values=["strong_cc_AUROC", "full_AUROC", "active_rate"]).dropna()
    summ = {
        "endpoints_scored_in_both_strata": len(pair),
        "share_of_pairs_where_filter_applies": float(t.groupby("stratum")[["n_active", "n_nonhit"]].sum().sum(axis=1).pipe(lambda s: s["applies"] / s.sum())),
        "median_active_rate_applies": float(pair[("active_rate", "applies")].median()), "median_active_rate_not_applicable": float(pair[("active_rate", "not_applicable")].median()),
        "median_strong_cc_AUROC_applies": float(pair[("strong_cc_AUROC", "applies")].median()), "median_strong_cc_AUROC_not_applicable": float(pair[("strong_cc_AUROC", "not_applicable")].median()),
        "median_full_AUROC_applies": float(pair[("full_AUROC", "applies")].median()), "median_full_AUROC_not_applicable": float(pair[("full_AUROC", "not_applicable")].median()),
        "median_delta_applies": float((pair[("full_AUROC", "applies")] - pair[("strong_cc_AUROC", "applies")]).median()),
        "median_delta_not_applicable": float((pair[("full_AUROC", "not_applicable")] - pair[("strong_cc_AUROC", "not_applicable")]).median()),
    }
    pd.Series(summ).round(4).to_csv(d / "filter_stratified_summary.csv", header=["value"])
    print(pd.Series(summ).round(3).to_string())


if __name__ == "__main__":
    main()
