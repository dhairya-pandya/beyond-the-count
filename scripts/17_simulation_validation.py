#!/usr/bin/env python
"""Validate the audit on synthetic data with known ground truth (src/cpsa/simulate.py).

For several simulated datasets: estimate the empirical-null sd from label-permutation runs (exactly as for the real data), audit every endpoint
and report, by endpoint kind, how often morphology is credited with an advantage -- raw bootstrap p-values vs null-calibrated p-values.
Expected: null and shortcut-only endpoints are (almost) never credited; endpoints that truly carry morphology information are detected
increasingly often with more actives and larger effects. -> results/simulation/{endpoints.csv,summary.csv,power.png}
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cpsa import RESULTS  # noqa: E402
from cpsa.api import estimate_null_sd, permutation_null, shortcut_audit  # noqa: E402
from cpsa.simulate import make_dataset  # noqa: E402
from cpsa.stats.multiple_testing import assign_verdicts  # noqa: E402


def one_dataset(seed: int, n_jobs: int, n_null_runs: int, n_samples: int) -> pd.DataFrame:
    d = make_dataset(n=n_samples, n_features=120, endpoints={"null": 100, "shortcut": 60, "adds": 120, "morph_only": 40}, seed=seed)
    null = permutation_null(d.profile, d.baseline, d.labels, n_runs=n_null_runs, n_repeats=2, n_boot=500, seed=seed, n_jobs=n_jobs)
    sd = estimate_null_sd(null)
    raw = shortcut_audit(d.profile, d.baseline, d.labels, n_repeats=3, n_boot=1000, seed=seed, n_jobs=n_jobs).table
    cal = shortcut_audit(d.profile, d.baseline, d.labels, n_repeats=3, n_boot=1000, seed=seed, n_jobs=n_jobs, null_sd=sd).table
    t = raw[["endpoint_id", "n_active", "powered", "delta_AUROC", "baseline_AUROC", "full_AUROC", "verdict"]].rename(columns={"verdict": "verdict_raw"})
    t["verdict_calibrated"] = cal["verdict"].to_numpy()
    t = t.merge(d.truth, on="endpoint_id")
    t["dataset_seed"], t["null_sd"], t["null_type1_raw"] = seed, sd, float((null["bootstrap_p"] < 0.05).mean())
    return t


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--n-null-runs", type=int, default=150)
    ap.add_argument("--n-samples", type=int, default=600)
    a = ap.parse_args()
    out = RESULTS / "simulation"
    out.mkdir(parents=True, exist_ok=True)
    t = pd.concat([one_dataset(s, a.n_jobs, a.n_null_runs, a.n_samples) for s in a.seeds], ignore_index=True)
    t.to_csv(out / "endpoints.csv", index=False)
    pw = t[t["powered"]].copy()
    pw["act_bin"] = pd.cut(pw["n_active"], [14, 29, 59, 10**6], labels=["15-29", "30-59", "60+"])
    rows = []
    for (kind, eff), g in pw.groupby(["kind", "effect"]):
        rows.append(dict(kind=kind, effect=eff, n_endpoints=len(g), credited_raw=(g.verdict_raw == "morphology_advantage").mean(),
                         credited_calibrated=(g.verdict_calibrated == "morphology_advantage").mean()))
    s = pd.DataFrame(rows)
    for (kind, eff, b), g in pw[pw.kind.isin(["adds", "morph_only"])].groupby(["kind", "effect", "act_bin"], observed=True):
        rows.append(dict(kind=kind, effect=eff, act_bin=b, n_endpoints=len(g), credited_raw=(g.verdict_raw == "morphology_advantage").mean(),
                         credited_calibrated=(g.verdict_calibrated == "morphology_advantage").mean()))
    pd.DataFrame(rows).to_csv(out / "summary.csv", index=False)
    print(s.round(3).to_string(index=False))
    print("null sd per dataset:", t.groupby("dataset_seed")["null_sd"].first().round(2).to_dict(),
          "| raw type-I in permutation runs:", t.groupby("dataset_seed")["null_type1_raw"].first().round(3).to_dict())
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
        for a_, col, title in zip(ax, ("credited_raw", "credited_calibrated"), ("raw bootstrap p", "null-calibrated p")):
            for kind, ls in (("adds", "-"), ("morph_only", "--")):
                for eff, c in zip((0.5, 1.0, 1.5), ("#9ecae1", "#4292c6", "#08519c")):
                    g = pd.DataFrame(rows).dropna(subset=["act_bin"])
                    g = g[(g.kind == kind) & (g.effect == eff)]
                    a_.plot(g["act_bin"], g[col], ls, color=c, marker="o", label=f"{kind}, effect {eff}")
            base = s[s.kind.isin(["null", "shortcut"])]
            for kind, c in (("null", "#d95f02"), ("shortcut", "#7f7f7f")):
                a_.axhline(float(base.loc[base.kind == kind, col].iloc[0]), color=c, ls=":", label=f"{kind} endpoints (should be ~0)")
            a_.set_title(title, fontsize=9); a_.set_xlabel("actives per endpoint"); a_.set_ylim(0, 1.02)
        ax[0].set_ylabel("fraction credited with a morphology advantage"); ax[1].legend(fontsize=6, frameon=False)
        fig.tight_layout(); fig.savefig(out / "power.png", dpi=200)
    except Exception as e:  # plotting is a convenience
        print("plot skipped:", e)


if __name__ == "__main__":
    main()
