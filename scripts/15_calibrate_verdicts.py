#!/usr/bin/env python
"""Apply the label-permutation empirical null to a configuration's p-values and recompute BH q-values and verdicts.

Raw bootstrap p-values/verdicts are preserved as `bootstrap_p` / `*_uncalibrated`; `verdict`, `fdr_q` and `calibrated_p` become the
null-calibrated ones (the primary results). Idempotent: re-running restarts from the raw columns.

  python scripts/15_calibrate_verdicts.py --config cellprofiler_allpod                    # null from the same config
  python scripts/15_calibrate_verdicts.py --config dino_allpod --null-from cellprofiler_allpod   # borrow another config's null sd
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cpsa import RESULTS  # noqa: E402
from cpsa.stats.null_calibration import calibrate_table, estimate_null  # noqa: E402
from cpsa.stats.signal_class import classify_signal, null_auroc_thresholds  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--null-from", default=None, help="config whose null_control.csv defines the null sd (default: same config)")
    ap.add_argument("--alpha", type=float, default=0.05)
    a = ap.parse_args()
    src = a.null_from or a.config
    null = pd.read_csv(RESULTS / src / "null_control.csv")
    est = estimate_null(null)
    est["null_from"] = src
    d = RESULTS / a.config
    audit = pd.read_csv(d / "audit_table.csv")
    for col in ("verdict", "verdict_scalar", "fdr_q", "fdr_q_scalar"):  # restart from the raw columns if already calibrated
        if f"{col}_uncalibrated" in audit:
            audit[col] = audit[f"{col}_uncalibrated"]
    audit = audit.drop(columns=[c for c in audit.columns if c.endswith("_uncalibrated") or c in ("delta_z", "delta_z_scalar", "calibrated_p", "calibrated_p_scalar", "null_sd_strong", "null_sd_scalar", "signal_class")])
    out = calibrate_table(audit, est["sd_strong"], est["sd_scalar"], alpha=a.alpha)
    thr = null_auroc_thresholds(null)
    est["auroc_null_q95"] = thr
    out["signal_class"] = classify_signal(out, thr["full"], thr["strong"])
    out.to_csv(d / "audit_table.csv", index=False)
    (d / "null_calibration.json").write_text(json.dumps(est, indent=2))
    T = out[~out["verdict"].isin(["positive_control"])]
    print("signal classes:", out["signal_class"].value_counts().to_dict(), "| null AUROC q95 thresholds:", {k: round(v, 3) for k, v in thr.items() if k in ("full", "strong")})
    print(json.dumps(est, indent=2))
    print(f"{a.config}: advantage vs strong  uncalibrated {int((T.verdict_uncalibrated == 'morphology_advantage').sum())}  ->  calibrated {int((T.verdict == 'morphology_advantage').sum())}"
          f" | vs scalar {int((T.verdict_scalar_uncalibrated == 'morphology_advantage').sum())} -> {int((T.verdict_scalar == 'morphology_advantage').sum())}")


if __name__ == "__main__":
    main()
