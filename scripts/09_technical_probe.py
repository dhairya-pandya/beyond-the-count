#!/usr/bin/env python
"""Technical-confound probe: can plate / well-position / batch alone predict each powered endpoint's label?

Complements the cell-count baselines (the original plan's 'plate/well/batch' baseline is undefined at compound level, so
we probe it separately). Writes results/<config>/technical_probe.csv with AUROC for tech-only and tech+strong-cc models.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cpsa import RESULTS  # noqa: E402
from cpsa.data.build_dataset import baseline_strong_columns, build_compound_table  # noqa: E402
from cpsa.data.load_labels import load_all_labels  # noqa: E402
from cpsa.data.technical import TECH_COLUMNS, technical_features  # noqa: E402
from cpsa.models.train_eval import auroc, oof_predict  # noqa: E402


def one(e, X, labels, cc_cols, n_repeats, seed):
    y = labels[e].dropna()
    ids = y.index.intersection(X.index)
    yv = y.loc[ids].astype(int).to_numpy()
    k = int(min(5, yv.sum(), (1 - yv).sum()))
    out = {"endpoint_id": e}
    for name, cols in (("tech", TECH_COLUMNS), ("tech_plus_strong_cc", TECH_COLUMNS + cc_cols)):
        p = oof_predict(X.loc[ids, cols], yv, np.asarray(ids), k, n_repeats, seed).mean(0)
        out[f"{name}_AUROC"] = auroc(yv, p)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="cellprofiler_allpod")
    ap.add_argument("--n-repeats", type=int, default=3)
    ap.add_argument("--n-jobs", type=int, default=8)
    a = ap.parse_args()
    fs, agg = a.config.split("_", 1)
    table = build_compound_table(fs, agg)
    cc_cols = baseline_strong_columns(table)
    X = table[cc_cols].join(technical_features(), how="inner")
    labels, _ = load_all_labels()
    audit = pd.read_csv(RESULTS / a.config / "audit_table.csv")
    todo = audit.loc[audit["powered"] & (audit["verdict"] != "positive_control"), "endpoint_id"].tolist()
    rows = Parallel(n_jobs=a.n_jobs)(delayed(one)(e, X, labels, cc_cols, a.n_repeats, 0) for e in todo)
    df = pd.DataFrame(rows).merge(audit[["endpoint_id", "strong_cc_AUROC", "full_AUROC", "verdict"]], on="endpoint_id")
    df.to_csv(RESULTS / a.config / "technical_probe.csv", index=False)
    print(df[["tech_AUROC", "tech_plus_strong_cc_AUROC", "strong_cc_AUROC", "full_AUROC"]].describe().round(3))
    print("frac endpoints tech AUROC > 0.6:", (df["tech_AUROC"] > 0.6).mean().round(3))
    adv = df[df["verdict"] == "morphology_advantage"]
    print("advantage endpoints where tech+cc >= full:", int((adv["tech_plus_strong_cc_AUROC"] >= adv["full_AUROC"]).sum()), "of", len(adv))


if __name__ == "__main__":
    main()
