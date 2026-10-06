#!/usr/bin/env python
"""Where does morphology help, by COMPOUND class? Per-compound morphology benefit (ranking accuracy of the full model minus the cell-count
baseline, averaged over powered endpoints) broken down by (i) the paper's hit pattern across readouts -- a mechanism-free test of 'morphology reads
sub-lethal stress before cells are lost' -- and (ii) vendor pathway / target / research area / clinical phase.
-> results/<config>/compound_benefit.csv, compound_classes_<grouping>.csv"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cpsa import RESULTS  # noqa: E402
from cpsa.biology.compound_classes import class_table, compound_benefit  # noqa: E402
from cpsa.data.compound_annotations import hit_pattern, vendor_classes  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="cellprofiler_allpod")
    ap.add_argument("--min-n", type=int, default=10)
    a = ap.parse_args()
    d = RESULTS / a.config
    audit = pd.read_csv(d / "audit_table.csv")
    oof = pd.read_parquet(d / "oof_predictions.parquet")
    eps = audit.loc[audit["powered"] & (audit["verdict"] != "positive_control"), "endpoint_id"].tolist()
    ben = compound_benefit(oof, eps)
    ben.to_csv(d / "compound_benefit.csv")
    print(f"{a.config}: {len(ben)} compounds, {len(eps)} powered endpoints; mean benefit {ben['benefit'].mean():+.3f} (>0 = morphology ranks compounds better than cell count)")
    groupings = {"hit_pattern": hit_pattern(), **vendor_classes()}
    for name, g in groupings.items():
        for col in ("benefit", "benefit_active"):
            t = class_table(ben, g, min_n=a.min_n, col=col)
            t.insert(0, "measure", col)
            t.to_csv(d / f"compound_classes_{name}_{col}.csv", index=False)
            if name == "hit_pattern" or (len(t) and (t["q"] < 0.1).any()):
                print(f"\n== {name} / {col}")
                print(t[t["q"] < 0.1].round(4).to_string(index=False) if name != "hit_pattern" else t.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
