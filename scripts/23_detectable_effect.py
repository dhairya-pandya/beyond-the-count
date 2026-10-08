#!/usr/bin/env python
"""Precision view of the verdicts: the smallest advantage (in AUROC) each endpoint's data could have detected.

  python scripts/23_detectable_effect.py --config cellprofiler_allpod   # -> results/<config>/detectable_effect.csv + detectable_effect.json

Per scored endpoint: bootstrap SE of the AUROC difference, SE widened by the permutation-null sd, and the minimum detectable effect
(one-sided alpha 0.05, 80% power). Sensitivity: how many powered endpoints are informative at a given detectable-effect threshold,
and how many 'no detectable advantage' verdicts rule out an advantage larger than that threshold.
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cpsa import RESULTS  # noqa: E402
from cpsa.stats.null_calibration import Z95  # noqa: E402
from cpsa.stats.precision import detectable_effect  # noqa: E402

THRESHOLDS = (0.05, 0.10, 0.15, 0.20)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="cellprofiler_allpod")
    a = ap.parse_args()
    d = RESULTS / a.config
    audit = pd.read_csv(d / "audit_table.csv")
    sd0 = json.loads((d / "null_calibration.json").read_text())["sd_strong"]
    t = audit[audit["delta_AUROC"].notna() & (audit["verdict"] != "positive_control")].copy()
    t["delta_se_bootstrap"] = (t["delta_ci_hi"] - t["delta_ci_lo"]) / (2 * Z95)
    t["delta_se_calibrated"] = t["delta_se_bootstrap"] * sd0
    t["min_detectable_effect"] = detectable_effect(t["delta_ci_lo"], t["delta_ci_hi"], sd0)
    keep = ["endpoint_id", "n_active", "n_inactive", "powered", "verdict", "delta_AUROC", "delta_se_bootstrap", "delta_se_calibrated", "min_detectable_effect"]
    t[keep].to_csv(d / "detectable_effect.csv", index=False)
    pw = t[t["powered"]]
    summ = dict(config=a.config, null_sd_used=sd0, n_powered=int(len(pw)), median_mde_powered=float(pw["min_detectable_effect"].median()),
                median_mde_underpowered=float(t.loc[~t["powered"], "min_detectable_effect"].median()))
    for thr in THRESHOLDS:
        prec = t["min_detectable_effect"] <= thr
        summ[f"mde<={thr}"] = dict(
            powered_and_precise=int((prec & t["powered"]).sum()),
            powered_but_wider=int((~prec & t["powered"]).sum()),
            underpowered_but_precise=int((prec & ~t["powered"]).sum()),
            certified_precise=int((prec & (t["verdict"] == "morphology_advantage")).sum()),
            no_advantage_that_rule_out_larger_effects=int((prec & (t["verdict"] == "no_advantage")).sum()),
        )
    (d / "detectable_effect.json").write_text(json.dumps(summ, indent=2))
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
