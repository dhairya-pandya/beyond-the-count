#!/usr/bin/env python
"""Label-permutation control: re-run the audit on shuffled labels. Expect AUROC ~0.5, p ~ uniform, ~0 BH calls."""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cpsa import RESULTS  # noqa: E402
from cpsa.audit import null_control  # noqa: E402


def chem_groups():
    from cpsa.data.chem_groups import cluster_smiles, load_smiles
    return cluster_smiles(load_smiles(), similarity=0.4)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--feature-set", default="cellprofiler")
    ap.add_argument("--agg", default="allpod")
    ap.add_argument("--n-endpoints", type=int, default=60)
    ap.add_argument("--n-perms", type=int, default=2)
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0, help="different seeds draw different endpoints / permutations")
    ap.add_argument("--n-repeats", type=int, default=2, help="CV repeats per run (the audit itself uses 3; fewer repeats make sd0 slightly conservative)")
    ap.add_argument("--n-boot", type=int, default=500, help="bootstrap resamples per run (the audit uses 1000)")
    ap.add_argument("--out-name", default="null_control.csv")
    ap.add_argument("--group-by", choices=["compound", "chem_cluster"], default="compound")
    ap.add_argument("--append", action="store_true", help="add to an existing null_control.csv instead of overwriting")
    a = ap.parse_args()
    df = null_control(a.feature_set, a.agg, n_endpoints=a.n_endpoints, n_perms=a.n_perms, n_jobs=a.n_jobs, seed=a.seed,
                      n_repeats=a.n_repeats, n_boot=a.n_boot,
                      group_map=chem_groups() if a.group_by == "chem_cluster" else None)
    out = RESULTS / f"{a.feature_set}_{a.agg}" / a.out_name
    out.parent.mkdir(parents=True, exist_ok=True)
    if a.append and out.exists():
        df = pd.concat([pd.read_csv(out), df.assign(endpoint_id=df["endpoint_id"] + f"__seed{a.seed}")], ignore_index=True)
    df.to_csv(out, index=False)
    print(f"runs={len(df)}  mean full AUROC={df.full_AUROC.mean():.3f}  mean strong_cc AUROC={df.strong_cc_AUROC.mean():.3f}  "
          f"frac p<0.05={(df.bootstrap_p < 0.05).mean():.3f}  BH advantage calls={(df.verdict == 'morphology_advantage').sum()}  -> {out}")


if __name__ == "__main__":
    main()
