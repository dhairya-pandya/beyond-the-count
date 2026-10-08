#!/usr/bin/env python
"""Enrichment of 'morphology advantage' endpoints by assay / target family (one config's audit_table.csv).

Fisher exact p / q per group, plus a cluster-level permutation p / q (p_cluster, q_cluster) in which whole assays are the exchangeable units, so
correlated endpoints from one assay count once."""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cpsa import RESULTS  # noqa: E402
from cpsa.biology.enrichment import assay_cluster, enrichment_table  # noqa: E402

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
    audit["assay_cluster"] = assay_cluster(audit["endpoint_id"])
    parts = []
    for col, tag in (("verdict", "vs_strong_cc"), ("verdict_scalar", "vs_scalar_cc")):
        t = enrichment_table(audit, by=BY, verdict_col=col, min_group_size=a.min_group_size, cluster_col="assay_cluster")
        t.insert(0, "baseline", tag)
        parts.append(t)
    out = pd.concat(parts, ignore_index=True)
    out.to_csv(d / "enrichment.csv", index=False)
    sig = out[(out.q < 0.1) | (out.q_cluster < 0.1)]
    print(sig.to_string() if len(sig) else "no group with q < 0.1")
    print(f"wrote {d / 'enrichment.csv'}  ({len(out)} group tests)")


if __name__ == "__main__":
    main()
