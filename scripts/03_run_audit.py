#!/usr/bin/env python
"""Entry point: run the cell-count shortcut audit for one feature set / aggregation.

Example:  python scripts/03_run_audit.py --feature-set cellprofiler --agg allpod
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cpsa.audit import run_audit  # noqa: E402
from cpsa.data.load_profiles import FEATURE_SETS  # noqa: E402
from cpsa.data.preprocess import AGG_METHODS  # noqa: E402


def chem_groups():
    from cpsa.data.chem_groups import cluster_smiles, load_smiles
    return cluster_smiles(load_smiles(), similarity=0.4)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--feature-set", default="cpcnn", choices=FEATURE_SETS)
    ap.add_argument("--agg", default="allpod", choices=AGG_METHODS)
    ap.add_argument("--categories", nargs="*", help="subset of: cellbased cellfree cytotox native")
    ap.add_argument("--endpoints", nargs="*", help="restrict to these endpoint ids (pilot runs)")
    ap.add_argument("--n-repeats", type=int, default=3)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--group-by", choices=["compound", "chem_cluster"], default="compound",
                    help="CV groups: one per compound (default) or ECFP4/Butina chemical clusters (similarity >= 0.4) to keep analog series together")
    ap.add_argument("--powered-only", action="store_true", help="skip endpoints with <15 actives or <15 inactives (sensitivity runs)")
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args()
    run_audit(a.feature_set, a.agg, categories=tuple(a.categories) if a.categories else None, endpoints=a.endpoints,
              n_repeats=a.n_repeats, n_boot=a.n_boot, n_jobs=a.n_jobs, seed=a.seed, out_dir=a.out_dir, powered_only=a.powered_only,
              group_map=chem_groups() if a.group_by == "chem_cluster" else None)


if __name__ == "__main__":
    main()
