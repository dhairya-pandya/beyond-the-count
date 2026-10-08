#!/usr/bin/env python
"""Nested comparison: does morphology add information on top of cell count?  (full + count model vs the count-only model)

  python scripts/22_nested_audit.py audit     --config cellprofiler_allpod     # nested model (+ logistic-regression comparator) on every powered endpoint
  python scripts/22_nested_audit.py null      --config cellprofiler_allpod --seed 0    # label-permutation runs at the same settings (repeat with --seed 1 ...)
  python scripts/22_nested_audit.py calibrate --config cellprofiler_allpod     # null-calibrated p, BH q, verdict_nested -> nested_table.csv

Outputs (results/<config>/): nested_audit.csv, nested_null.csv, nested_table.csv, nested_calibration.json
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cpsa import RESULTS  # noqa: E402
from cpsa.nested import calibrate_nested, estimate_nested_null, nested_audit, nested_null  # noqa: E402


def split_config(config: str) -> tuple[str, str]:
    fs, agg = config.split("_", 1)
    return fs, agg


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["audit", "null", "calibrate"])
    ap.add_argument("--config", default="cellprofiler_allpod")
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--n-repeats", type=int, default=3)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--endpoints", nargs="*", help="restrict to these endpoint ids (pilot runs)")
    ap.add_argument("--no-logreg", action="store_true")
    ap.add_argument("--n-endpoints", type=int, default=100, help="null: endpoints drawn per seed")
    ap.add_argument("--n-perms", type=int, default=3, help="null: permutations per endpoint")
    ap.add_argument("--out-suffix", default="", help="append to output file names (pilot runs)")
    a = ap.parse_args()
    d = RESULTS / a.config
    d.mkdir(parents=True, exist_ok=True)
    fs, agg = split_config(a.config)
    if a.step == "audit":
        df = nested_audit(fs, agg, endpoints=a.endpoints, n_repeats=a.n_repeats, n_boot=a.n_boot, n_jobs=a.n_jobs, seed=a.seed, with_logreg=not a.no_logreg)
        df.to_csv(d / f"nested_audit{a.out_suffix}.csv", index=False)
        print(f"{len(df)} endpoints -> {d / f'nested_audit{a.out_suffix}.csv'}")
    elif a.step == "null":
        df = nested_null(fs, agg, n_endpoints=a.n_endpoints, n_perms=a.n_perms, n_repeats=a.n_repeats, n_boot=a.n_boot, n_jobs=a.n_jobs, seed=a.seed)
        out = d / f"nested_null{a.out_suffix}.csv"
        if out.exists():
            df = pd.concat([pd.read_csv(out), df], ignore_index=True)
        df.to_csv(out, index=False)
        print(f"{len(df)} permutation runs -> {out}")
    else:
        rows = pd.read_csv(d / "nested_audit.csv")
        est = estimate_nested_null(pd.read_csv(d / "nested_null.csv"))
        out = calibrate_nested(rows, est["sd"])
        audit = pd.read_csv(d / "audit_table.csv")[["endpoint_id", "category", "assay_target_family", "full_AUROC", "strong_cc_AUROC", "delta_AUROC", "verdict"]]
        table = audit.merge(out.drop(columns=["strong_cc_AUROC"]), on="endpoint_id", how="inner").rename(columns={"verdict": "verdict_main"})
        table.to_csv(d / "nested_table.csv", index=False)
        summ = dict(est, n_powered=int(len(table)), adds_beyond_count=int((table.verdict_nested == "adds_beyond_count").sum()),
                    main_advantage=int((table.verdict_main == "morphology_advantage").sum()),
                    both=int(((table.verdict_nested == "adds_beyond_count") & (table.verdict_main == "morphology_advantage")).sum()),
                    median_delta_nested=float(table.delta_nested.median()), median_delta_main=float(table.delta_AUROC.median()))
        if "logreg_AUROC" in table:
            summ.update(median_full_AUROC=float(table.full_AUROC.median()), median_logreg_AUROC=float(table.logreg_AUROC.median()),
                        median_nested_AUROC=float(table.nested_AUROC.median()))
        (d / "nested_calibration.json").write_text(json.dumps(summ, indent=2))
        print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
