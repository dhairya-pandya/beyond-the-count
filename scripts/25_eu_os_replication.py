#!/usr/bin/env python
"""Second morphology source: audit the EU-OPENSCREEN HepG2 profiles against the same ToxCast / Tox21 / MT / LDH labels.

  python scripts/25_eu_os_replication.py              # -> results/eu_os/

Requires data/external/eu_os (Zenodo 10.5281/zenodo.19347244 plus the v1.0.0 annotation files; see src/cpsa/data/eu_os.py).
Compounds present in both resources (InChIKey skeleton match) keep the labels of the main audit. Two audits run on identical compounds,
endpoints and settings, each with its own label-permutation null:

  hepg2   EU-OPENSCREEN HepG2 profiles (median over four imaging sites and replicates, 10 uM) against the single-number cell-count baseline
  oasis   primary-hepatocyte CellProfiler profiles of the main audit against the same single-number baseline

Only one concentration exists in HepG2, so the dose-response baseline of the main audit cannot be built: both audits use the paper-style
scalar count. Exploratory by design: different cell system, different imaging, 4 sites.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cpsa import PROCESSED, RESULTS  # noqa: E402
from cpsa.api import estimate_null_sd, permutation_null, shortcut_audit  # noqa: E402
from cpsa.data.build_dataset import build_compound_table, profile_columns  # noqa: E402
from cpsa.data.eu_os import build_overlap_tables  # noqa: E402
from cpsa.data.load_labels import load_all_labels  # noqa: E402


def load_inputs():
    pp, bp = PROCESSED / "eu_os_profile.parquet", PROCESSED / "eu_os_baseline.parquet"
    if pp.exists() and bp.exists():
        prof, base = pd.read_parquet(pp), pd.read_parquet(bp)
    else:
        prof, base = build_overlap_tables()
        PROCESSED.mkdir(parents=True, exist_ok=True)
        prof.to_parquet(pp)
        base.to_parquet(bp)
    table = build_compound_table("cellprofiler", "allpod")
    ids = prof.index.intersection(table.index)
    labels, meta = load_all_labels()
    lab = labels.loc[labels.index.intersection(ids)]
    n1, n0 = (lab == 1).sum(), (lab == 0).sum()
    keep = [e for e in lab.columns if n1[e] >= 15 and n0[e] >= 15]
    return prof.loc[ids], base.loc[ids], table.loc[ids], lab[keep], meta


def run(name, profile, baseline, labels, a, out):
    kw = dict(n_repeats=3, n_boot=1000, seed=a.seed, n_jobs=a.n_jobs)
    null = permutation_null(profile, baseline, labels, n_runs=a.n_null_runs, **kw)
    null.to_csv(out / f"null_{name}.csv", index=False)
    sd = estimate_null_sd(null)
    res = shortcut_audit(profile, baseline, labels, controls=("cell_count",), null_sd=sd, **kw).table
    raw = shortcut_audit(profile, baseline, labels, controls=("cell_count",), **kw).table
    res["verdict_uncalibrated"] = raw["verdict"].to_numpy()
    res["fdr_q_uncalibrated"] = raw["fdr_q"].to_numpy()
    res["null_sd"] = sd
    res.to_csv(out / f"audit_{name}.csv", index=False)
    return res, null, sd


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-null-runs", type=int, default=300)
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = RESULTS / "eu_os"
    out.mkdir(parents=True, exist_ok=True)
    prof, base, table, labels, meta = load_inputs()
    print(f"compounds {len(prof)}  endpoints powered on the overlap {labels.shape[1]}  features HepG2 {prof.shape[1]}")
    oasis_prof = table[profile_columns(table)]
    oasis_base = table[["Cell_Count"]]
    h, h_null, h_sd = run("hepg2", prof, base, labels, a, out)
    o, o_null, o_sd = run("oasis", oasis_prof, oasis_base, labels, a, out)
    m = h.merge(o, on="endpoint_id", suffixes=("_hepg2", "_oasis"))
    m = m.merge(meta[["category", "assay_target_family"]], left_on="endpoint_id", right_index=True, how="left")
    m.to_csv(out / "comparison.csv", index=False)
    tested = m[~m["verdict_hepg2"].isin(["positive_control"])]
    both = (tested.verdict_hepg2 == "morphology_advantage") & (tested.verdict_oasis == "morphology_advantage")
    summ = dict(
        n_compounds=int(len(prof)), n_endpoints=int(len(tested)), n_features_hepg2=int(prof.shape[1]), n_features_oasis=int(oasis_prof.shape[1]),
        null_sd_hepg2=h_sd, null_sd_oasis=o_sd,
        null_runs_hepg2=int(len(h_null)), null_runs_oasis=int(len(o_null)),
        certified_hepg2=int((tested.verdict_hepg2 == "morphology_advantage").sum()),
        certified_oasis=int((tested.verdict_oasis == "morphology_advantage").sum()),
        certified_both=int(both.sum()),
        raw_wins_hepg2=int((tested.verdict_uncalibrated_hepg2 == "morphology_advantage").sum()),
        raw_wins_oasis=int((tested.verdict_uncalibrated_oasis == "morphology_advantage").sum()),
        median_full_AUROC_hepg2=float(tested.full_AUROC_hepg2.median()), median_full_AUROC_oasis=float(tested.full_AUROC_oasis.median()),
        median_baseline_AUROC_hepg2=float(tested.baseline_AUROC_hepg2.median()), median_baseline_AUROC_oasis=float(tested.baseline_AUROC_oasis.median()),
        median_delta_hepg2=float(tested.delta_AUROC_hepg2.median()), median_delta_oasis=float(tested.delta_AUROC_oasis.median()),
        spearman_full_AUROC=float(tested[["full_AUROC_hepg2", "full_AUROC_oasis"]].corr(method="spearman").iloc[0, 1]),
        spearman_delta=float(tested[["delta_AUROC_hepg2", "delta_AUROC_oasis"]].corr(method="spearman").iloc[0, 1]),
    )
    (out / "summary.json").write_text(json.dumps(summ, indent=2))
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
