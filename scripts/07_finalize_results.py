#!/usr/bin/env python
"""Promote one configuration to the headline results, summarise sensitivity across configurations, draw report figures.

Writes results/{audit_table.csv, oof_predictions.parquet, enrichment.csv}, results/sensitivity_summary.csv and results/figures/*.png
"""
import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cpsa import RESULTS  # noqa: E402
from cpsa.stats.calibration import platt_oof  # noqa: E402
from cpsa.viz import plots  # noqa: E402


def summarise(cfg_dir: Path, primary_adv: set[str]) -> dict:
    a = pd.read_csv(cfg_dir / "audit_table.csv")
    a = a[a["verdict"] != "positive_control"]
    adv = set(a.loc[a["verdict"] == "morphology_advantage", "endpoint_id"])
    jac = len(adv & primary_adv) / max(len(adv | primary_adv), 1)
    return dict(
        config=cfg_dir.name, endpoints=len(a), powered=int(a["powered"].sum()), indeterminate=int((a["verdict"] == "indeterminate").sum()),
        naive_delta_gt0_strong=int((a["delta_AUROC"] > 0).sum()), adv_vs_strong=len(adv),
        adv_vs_scalar=int((a["verdict_scalar"] == "morphology_advantage").sum()),
        adv_vs_strong_raw_bootstrap=int((a["verdict_uncalibrated"] == "morphology_advantage").sum()),
        adv_vs_scalar_raw_bootstrap=int((a["verdict_scalar_uncalibrated"] == "morphology_advantage").sum()),
        null_sd_used=round(float(a["null_sd_strong"].iloc[0]), 3),
        median_full_AUROC=a.loc[a["powered"], "full_AUROC"].median(), median_strong_cc_AUROC=a.loc[a["powered"], "strong_cc_AUROC"].median(),
        median_scalar_cc_AUROC=a.loc[a["powered"], "scalar_cc_AUROC"].median(), jaccard_vs_primary_advantage=jac,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--primary", default="cellprofiler_allpod")
    a = ap.parse_args()
    src = RESULTS / a.primary
    for f in ("audit_table.csv", "oof_predictions.parquet", "enrichment.csv"):
        if (src / f).exists():
            shutil.copy(src / f, RESULTS / f)
    audit = pd.read_csv(RESULTS / "audit_table.csv")
    oof_raw = pd.read_parquet(RESULTS / "oof_predictions.parquet")  # add cross-validated Platt-recalibrated probabilities for the demo
    for col in ("p_full", "p_strong_cc"):
        oof_raw[col + "_cal"] = oof_raw.groupby("endpoint_id", group_keys=False).apply(
            lambda g, c=col: pd.Series(platt_oof(g["y"].to_numpy(), g[c].to_numpy()), index=g.index))
    oof_raw.to_parquet(RESULTS / "oof_predictions.parquet")
    prim_adv = set(audit.loc[audit["verdict"] == "morphology_advantage", "endpoint_id"])
    cfgs = sorted(d for d in RESULTS.iterdir() if (d / "audit_table.csv").exists() and d.name != "pilot")
    pd.DataFrame([summarise(d, prim_adv) for d in cfgs]).to_csv(RESULTS / "sensitivity_summary.csv", index=False)
    fig = RESULTS / "figures"
    oof = pd.read_parquet(RESULTS / "oof_predictions.parquet")
    plots.pipeline_diagram(fig / "pipeline.png")
    plots.scatter_baseline_vs_full(audit, fig / "scatter_strong.png")
    plots.scatter_baseline_vs_full(audit, fig / "scatter_scalar.png", x="scalar_cc_AUROC", xlabel="AUROC, scalar cell-count baseline")
    plots.delta_forest(audit, fig / "delta_forest.png")
    plots.correction_funnel(audit[audit["verdict"] != "positive_control"], fig / "correction_funnel.png")
    plots.verdicts_by_category(audit, fig / "verdicts_by_category.png")
    plots.pooled_reliability(oof, audit, fig / "pooled_reliability.png")
    if (RESULTS / "enrichment.csv").exists():
        plots.enrichment_bars(pd.read_csv(RESULTS / "enrichment.csv"), fig / "enrichment_family.png")
    cmp_ = RESULTS / a.primary / "paper_comparison.csv"
    if cmp_.exists():
        plots.paper_agreement(pd.read_csv(cmp_), fig / "paper_agreement.png")
    null = RESULTS / a.primary / "null_control.csv"
    if null.exists():
        plots.null_pvalues(pd.read_csv(null), fig / "null_control.png")
    print(pd.read_csv(RESULTS / "sensitivity_summary.csv").round(3).to_string(index=False))


if __name__ == "__main__":
    main()
