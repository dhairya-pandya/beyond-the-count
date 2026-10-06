#!/usr/bin/env python
"""Sensitivity of the headline count to analysis choices, recomputed from saved audit_table.csv / oof_predictions.parquet
(no re-training): power threshold, alpha, baseline (strong vs scalar), raw vs null-calibrated p-values, and a chance-floored
baseline. Uses the null-calibrated p-values (columns calibrated_p*) when the table has them."""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cpsa import RESULTS  # noqa: E402
from cpsa.stats.multiple_testing import assign_verdicts  # noqa: E402
from cpsa.stats.null_calibration import calibrated_p, z_scores  # noqa: E402


def count_adv(df: pd.DataFrame, dcol: str, pcol: str, alpha: float, k: int) -> tuple[int, int]:
    v = assign_verdicts(df.assign(delta_auroc=df[dcol], bootstrap_p=df[pcol]), alpha, k, k)
    return int(v["powered"].sum()), int((v["verdict"] == "morphology_advantage").sum())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="cellprofiler_allpod")
    a = ap.parse_args()
    d = RESULTS / a.config
    A = pd.read_csv(d / "audit_table.csv")
    T = A[(A["verdict"] != "positive_control") & A["delta_AUROC"].notna()].copy()
    rows = []
    for k in (10, 15, 20, 30):
        for alpha in (0.05, 0.10):
            for tag, dcol, pcol in (("strong_cc", "delta_AUROC", "calibrated_p"), ("scalar_cc", "delta_AUROC_scalar", "calibrated_p_scalar")):
                n_pow, n_adv = count_adv(T, dcol, pcol, alpha, k)
                rows.append(dict(analysis="power_threshold_grid", min_per_class=k, alpha=alpha, baseline=tag, p_values="null-calibrated", n_powered=n_pow, n_advantage=n_adv))
            if k == 15:
                for tag, dcol, pcol in (("strong_cc", "delta_AUROC", "bootstrap_p"), ("scalar_cc", "delta_AUROC_scalar", "bootstrap_p_scalar")):
                    n_pow, n_adv = count_adv(T, dcol, pcol, alpha, k)
                    rows.append(dict(analysis="power_threshold_grid", min_per_class=k, alpha=alpha, baseline=tag, p_values="raw bootstrap", n_powered=n_pow, n_advantage=n_adv))

    # chance-floored strong baseline: delta = full - max(strong_cc, 0.5), z uses the unfloored bootstrap SE (conservative)
    sd = float(T["null_sd_strong"].iloc[0])
    F = T.copy()
    F["delta_floor"] = F["full_AUROC"] - F["strong_cc_AUROC"].clip(lower=0.5)
    se = (F["delta_ci_hi"] - F["delta_ci_lo"]) / 3.919928
    F["p_floor"] = calibrated_p(F["delta_floor"] / se, sd)
    n_pow, n_adv = count_adv(F, "delta_floor", "p_floor", 0.05, 15)
    base_adv = set(T.loc[T["verdict"] == "morphology_advantage", "endpoint_id"])
    v = assign_verdicts(F.assign(delta_auroc=F["delta_floor"], bootstrap_p=F["p_floor"]), 0.05, 15, 15)
    floor_adv = set(F.loc[v["verdict"] == "morphology_advantage", "endpoint_id"])
    rows.append(dict(analysis="chance_floored_baseline", min_per_class=15, alpha=0.05, baseline="strong_cc(floor 0.5)", p_values="null-calibrated", n_powered=n_pow, n_advantage=n_adv,
                     retained_of_primary=len(base_adv & floor_adv), n_primary=len(base_adv)))
    out = pd.DataFrame(rows)
    out.to_csv(d / "sensitivity_postprocess.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
