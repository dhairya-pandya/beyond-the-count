#!/usr/bin/env python
"""Fisher-exact enrichment of 'morphology advantage' endpoints by assay / target family (one config's audit_table.csv)."""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cpsa import RESULTS  # noqa: E402
from cpsa.biology.enrichment import enrichment_table  # noqa: E402

BY = ["category", "assay_target_family", "assay_design_type", "cell_short_name", "tissue", "assay_mode"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="cpcnn_allpod", help="results/<config>/audit_table.csv")
    ap.add_argument("--min-group-size", type=int, default=5)
    a = ap.parse_args()
    d = RESULTS / a.config
    audit = pd.read_csv(d / "audit_table.csv")
    # exploratory grouping: signal-decrease (antagonist-mode) assays are classically cytotoxicity-confounded
    audit["assay_mode"] = np.where(audit["endpoint_id"].str.contains("Antagonist", case=False), "antagonist-mode",
                                   np.where(audit["endpoint_id"].str.contains("Agonist", case=False), "agonist-mode", "other"))
    parts = []
    for col, tag in (("verdict", "vs_strong_cc"), ("verdict_scalar", "vs_scalar_cc")):
        t = enrichment_table(audit, by=BY, verdict_col=col, min_group_size=a.min_group_size)
        t.insert(0, "baseline", tag)
        parts.append(t)
    out = pd.concat(parts, ignore_index=True)
    out.to_csv(d / "enrichment.csv", index=False)
    print(out[out.q < 0.1].to_string() if (out.q < 0.1).any() else "no group with q < 0.1")
    print(f"wrote {d / 'enrichment.csv'}  ({len(out)} group tests)")


if __name__ == "__main__":
    main()
