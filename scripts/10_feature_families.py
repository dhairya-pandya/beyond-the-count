#!/usr/bin/env python
"""Which CellProfiler feature families (compartment / feature type / channel) carry the full model's signal, for endpoints
with vs without a morphology advantage? Gain importance from one full-data fit per powered endpoint (interpretability aid;
not used for any verdict)."""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cpsa import RESULTS  # noqa: E402
from cpsa.biology.feature_families import aggregate_importance, parse_feature_name  # noqa: E402
from cpsa.data.build_dataset import build_compound_table, build_endpoint_dataset, profile_columns  # noqa: E402
from cpsa.data.load_labels import load_all_labels  # noqa: E402
from cpsa.models.train_eval import make_model  # noqa: E402


def one(e, table, labels, cols):
    d = build_endpoint_dataset(table, labels, e)
    y = d.pop("label").to_numpy()
    m = make_model(y, 0).fit(d[cols].to_numpy("float32"), y)
    return e, m.feature_importances_


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="cellprofiler_allpod")
    ap.add_argument("--n-jobs", type=int, default=8)
    a = ap.parse_args()
    fs, agg = a.config.split("_", 1)
    if fs != "cellprofiler":
        sys.exit("feature-name parsing is only defined for CellProfiler features")
    table = build_compound_table(fs, agg)
    labels, _ = load_all_labels()
    audit = pd.read_csv(RESULTS / a.config / "audit_table.csv").set_index("endpoint_id")
    todo = audit.index[audit["powered"] & (audit["verdict"] != "positive_control")]
    cols = profile_columns(table)
    res = Parallel(n_jobs=a.n_jobs)(delayed(one)(e, table, labels, cols) for e in todo)
    imp = pd.DataFrame(np.vstack([r[1] for r in res]), index=[r[0] for r in res], columns=cols)
    out = RESULTS / a.config
    imp.to_parquet(out / "feature_importance.parquet")
    for level in ("compartment", "group", "channel"):
        fam = aggregate_importance(imp, level)
        grp = audit.loc[fam.index, "verdict"]
        summ = fam.groupby(grp).mean().T
        idx = {"compartment": 0, "group": 1, "channel": 2}[level]
        share = pd.Series([parse_feature_name(x)[idx] for x in cols]).value_counts(normalize=True)
        summ["share_of_features"] = share.reindex(summ.index).fillna(0.0)
        summ.round(4).to_csv(out / f"feature_family_{level}.csv")
        print(f"\n== {level} (mean importance share per verdict; share_of_features = what chance would give)")
        print(summ.round(3).sort_values("morphology_advantage", ascending=False).to_string())


if __name__ == "__main__":
    main()
