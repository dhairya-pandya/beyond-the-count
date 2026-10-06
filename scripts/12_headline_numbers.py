#!/usr/bin/env python
"""Every number quoted in the technical report, computed from a config's audit_table.csv -> results/<config>/headline_numbers.json."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cpsa import RESULTS  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="cellprofiler_allpod")
    a = ap.parse_args()
    d = RESULTS / a.config
    A = pd.read_csv(d / "audit_table.csv")
    ctl = A[A["verdict"] == "positive_control"]
    T = A[A["verdict"] != "positive_control"]
    P = T[T["powered"]]
    I = T[~T["powered"]]
    adv = T[T["verdict"] == "morphology_advantage"]
    out = {
        "config": a.config,
        "n_endpoints_tested": len(T),
        "n_by_category": T["category"].value_counts().to_dict(),
        "n_powered": len(P), "n_indeterminate": len(I), "frac_indeterminate": round(len(I) / len(T), 3),
        "powered_by_category": P["category"].value_counts().to_dict(),
        "verdict_by_category": T.groupby(["category", "verdict"]).size().unstack(fill_value=0).to_dict("index"),
        "funnel": {
            "tested": len(T),
            "naive_delta_gt0_all": int((T["delta_AUROC"] > 0).sum()),
            "naive_delta_gt0_powered": int((P["delta_AUROC"] > 0).sum()),
            "powered_raw_p_lt_0.05_uncorrected": int(((P["delta_AUROC"] > 0) & (P["bootstrap_p"] < 0.05)).sum()),
            "bh_q_lt_0.05_raw_bootstrap_p": int((T["verdict_uncalibrated"] == "morphology_advantage").sum()),
            "bh_q_lt_0.05_null_calibrated_p": len(adv),
        },
        "advantage_vs_strong_raw": int((T["verdict_uncalibrated"] == "morphology_advantage").sum()),
        "advantage_vs_scalar_raw": int((T["verdict_scalar_uncalibrated"] == "morphology_advantage").sum()),
        "frac_powered_delta_gt0": round(float((P["delta_AUROC"] > 0).mean()), 3),
        "n_powered_delta_gt0": int((P["delta_AUROC"] > 0).sum()),
        "null_sd": {"strong": round(float(P["null_sd_strong"].iloc[0]), 3), "scalar": round(float(P["null_sd_scalar"].iloc[0]), 3)},
        "advantage_vs_scalar_baseline": int((T["verdict_scalar"] == "morphology_advantage").sum()),
        "advantage_vs_scalar_but_not_vs_strong": int(((T["verdict_scalar"] == "morphology_advantage") & (T["verdict"] != "morphology_advantage")).sum()),
        "advantage_vs_strong_but_not_vs_scalar": int(((T["verdict_scalar"] != "morphology_advantage") & (T["verdict"] == "morphology_advantage")).sum()),
        "median_AUROC_powered": {k: round(float(P[f"{k}_AUROC"].median()), 3) for k in ("scalar_cc", "strong_cc", "full")},
        "median_delta_powered": round(float(P["delta_AUROC"].median()), 3),
        "median_delta_advantage": round(float(adv["delta_AUROC"].median()), 3),
        "AUROC_sd_powered_full": round(float(P["full_AUROC"].std()), 3), "AUROC_sd_indeterminate_full": round(float(I["full_AUROC"].std()), 3),
        "indeterminate_full_AUROC_below_0.4_or_above_0.8": int(((I["full_AUROC"] < 0.4) | (I["full_AUROC"] > 0.8)).sum()),
        "indeterminate_full_AUROC_below_0.4": int((I["full_AUROC"] < 0.4).sum()),
        "calibration_powered": {
            "median_brier_skill": round(float(P["brier_skill"].median()), 3), "median_ece": round(float(P["ece"].median()), 3),
            "median_calib_slope": round(float(P["calib_slope"].median()), 3),
            "frac_slope_lt_0.8": round(float((P["calib_slope"] < 0.8).mean()), 3),
            "frac_brier_skill_gt0": round(float((P["brier_skill"] > 0).mean()), 3),
            "median_brier_raw_vs_recal": [round(float(P["calibration_brier"].median()), 4), round(float(P["brier_recalibrated"].median()), 4)],
        },
        "native": A[A["endpoint_id"].isin(["MT", "LDH", "cell_count"])].set_index("endpoint_id")[
            ["scalar_cc_AUROC", "strong_cc_AUROC", "full_AUROC", "delta_AUROC", "delta_z", "calibrated_p", "fdr_q", "fdr_q_scalar", "verdict", "verdict_scalar", "verdict_uncalibrated"]].round(4).to_dict("index"),
        "top_advantage": adv.sort_values("delta_AUROC", ascending=False).head(10)[["endpoint_id", "category", "assay_target_family", "n_active", "strong_cc_AUROC", "full_AUROC", "delta_AUROC", "fdr_q"]].round(3).to_dict("records"),
        "cellfree_powered_endpoints": P[P["category"] == "cellfree"][["endpoint_id", "n_active", "strong_cc_AUROC", "full_AUROC", "verdict"]].round(3).to_dict("records"),
    }
    if (d / "null_control.csv").exists():
        from cpsa.stats.null_calibration import calibrated_p, z_scores
        N = pd.read_csv(d / "null_control.csv")
        N = N[(N["n_active"] >= 15) & (N["n_inactive"] >= 15)]
        z = z_scores(N["delta_AUROC"], N["delta_ci_lo"], N["delta_ci_hi"])
        bins = pd.cut(N["n_active"], [14, 29, 59, 10**6], labels=["15-29", "30-59", "60+"])
        held = N["endpoint_id"].str.contains("__seed")  # split-half check: sd from the first batch of runs, error rates on the held-out batch
        split = None
        if held.any() and (~held).any():
            sd0 = float(np.std(z[~held.to_numpy()], ddof=1))
            split = {"sd_from_first_batch": round(sd0, 3), "n_first": int((~held).sum()), "n_heldout": int(held.sum()),
                     "heldout_type1_raw_at_0.05": round(float((N.loc[held, "bootstrap_p"] < 0.05).mean()), 3),
                     "heldout_type1_calibrated_at_0.05": round(float((calibrated_p(z[held.to_numpy()], sd0) < 0.05).mean()), 3),
                     "heldout_type1_calibrated_at_0.01": round(float((calibrated_p(z[held.to_numpy()], sd0) < 0.01).mean()), 3)}
        out["null_control"] = {
            "split_half": split,
            "powered_runs": len(N), "mean_AUROC_full": round(float(N["full_AUROC"].mean()), 3), "mean_AUROC_strong_cc": round(float(N["strong_cc_AUROC"].mean()), 3),
            "mean_delta": round(float(N["delta_AUROC"].mean()), 4), "sd_z": round(float(np.std(z, ddof=1)), 3),
            "sd_z_by_n_active": {str(k): round(float(np.std(z[bins == k], ddof=1)), 3) for k in bins.cat.categories},
            "type1_raw_at_0.05": round(float((N["bootstrap_p"] < 0.05).mean()), 3), "type1_raw_at_0.01": round(float((N["bootstrap_p"] < 0.01).mean()), 3),
            "type1_calibrated_at_0.05_in_sample": round(float((calibrated_p(z, float(np.std(z, ddof=1))) < 0.05).mean()), 3),
            "bh_advantage_calls_raw_pipeline": int((N["verdict"] == "morphology_advantage").sum()),
        }
    (d / "headline_numbers.json").write_text(json.dumps(out, indent=2, default=lambda o: o.item() if isinstance(o, np.generic) else str(o)))
    print(json.dumps(out, indent=2, default=lambda o: o.item() if isinstance(o, np.generic) else str(o)))


if __name__ == "__main__":
    main()
