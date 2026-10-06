#!/usr/bin/env python
"""Validate our pipeline against the paper's own compiled per-endpoint metrics, and quantify what multiple-testing
correction + power flagging change relative to a naive 'Actual > cell-count baseline' reading.

Paper metrics: data/external/ewald_repo/2_downstream_analysis/compiled_results/compiled_*_metrics.parquet
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cpsa import RESULTS  # noqa: E402
from cpsa.data.load_labels import REPO  # noqa: E402

COMPILED = REPO / "2_downstream_analysis" / "compiled_results"
RENAME = {"MTT": "MT"}


def paper_metrics(feature_set: str, agg: str) -> pd.DataFrame:
    frames = []
    for name in ("cellbased", "cellfree", "cytotox", "axiom"):
        f = COMPILED / f"compiled_{'toxcast_' if name != 'axiom' else ''}{name}_metrics.parquet"
        frames.append(pd.read_parquet(f))
    d = pd.concat(frames)
    d = d[(d["Feat_type"] == feature_set) & (d["Metadata_AggType"] == agg)]
    d["endpoint_id"] = d["Metadata_Label"].replace(RENAME)
    w = d.pivot_table(index="endpoint_id", columns="Model_type", values="AUROC")
    w.columns = [f"paper_{c}_AUROC" for c in w.columns]
    n = d.drop_duplicates("endpoint_id").set_index("endpoint_id")[["Metadata_Count_0", "Metadata_Count_1"]]
    return w.join(n)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="cpcnn_allpod")
    a = ap.parse_args()
    fs, agg = a.config.split("_", 1)
    ours = pd.read_csv(RESULTS / a.config / "audit_table.csv").set_index("endpoint_id")
    paper = paper_metrics(fs, agg)
    m = ours.join(paper, how="inner").dropna(subset=["paper_Actual_AUROC", "full_AUROC"])
    print(f"{a.config}: {len(m)} endpoints in common with the paper's compiled metrics")
    rows = []
    for ours_col, paper_col, label in (("full_AUROC", "paper_Actual_AUROC", "full model AUROC"),
                                       ("scalar_cc_AUROC", "paper_Cellcount_baseline_AUROC", "scalar cell-count AUROC")):
        x, y = m[ours_col], m[paper_col]
        rows.append(dict(quantity=label, n=len(m), pearson=pearsonr(x, y)[0], spearman=spearmanr(x, y)[0], mean_ours_minus_paper=(x - y).mean(), mad=(x - y).abs().mean()))
    cmp_ = pd.DataFrame(rows)
    print(cmp_.round(3).to_string(index=False))

    ctl = m.drop(index=[i for i in ("cell_count",) if i in m.index])
    paper_delta = ctl["paper_Actual_AUROC"] - ctl["paper_Cellcount_baseline_AUROC"]
    our_delta = ctl["full_AUROC"] - ctl["scalar_cc_AUROC"]
    summ = dict(
        endpoints=len(ctl),
        paper_naive_actual_gt_cc=int((paper_delta > 0).sum()),
        ours_naive_delta_gt0_scalar=int((our_delta > 0).sum()),
        ours_powered=int(ctl["powered"].sum()),
        ours_indeterminate=int((ctl["verdict"] == "indeterminate").sum()),
        ours_advantage_vs_scalar_BH=int((ctl["verdict_scalar"] == "morphology_advantage").sum()),
        ours_advantage_vs_strong_BH=int((ctl["verdict"] == "morphology_advantage").sum()),
        direction_agreement=float(((paper_delta > 0) == (our_delta > 0)).mean()),
    )
    print(pd.Series(summ).to_string())
    m.to_csv(RESULTS / a.config / "paper_comparison.csv")
    cmp_.assign(**summ).to_csv(RESULTS / a.config / "paper_comparison_summary.csv", index=False)


if __name__ == "__main__":
    main()
