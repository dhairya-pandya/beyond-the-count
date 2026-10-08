#!/usr/bin/env python
"""Chip-scale view of the audit (simulation with known ground truth) and a planning curve from the real endpoints.

  python scripts/24_chip_scale_study.py            # -> results/chip_scale/

* endpoints.csv / summary.csv: audit of simulated datasets with 20-200 compounds (one row per endpoint / per size): how many endpoints can be
  judged, how often endpoints without a morphology advantage are credited, power to credit real ones, minimum detectable effect.
* leakage.csv: compounds replicated over chips with noise labels; folds grouped by compound vs treating each chip as independent.
* planner_fit.json: log-log fit of the minimum detectable effect against effective sample size over the real CellProfiler endpoints.
Nothing here is chip data: it shows what the method needs at chip scale.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cpsa import RESULTS  # noqa: E402
from cpsa.chip import chip_scale_study, fit_planning_curve, planned_detectable_effect, replicate_leakage_demo, summarise_study  # noqa: E402


def make_figures(summary: pd.DataFrame, leak: pd.DataFrame, curve: dict, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, grey, mag, blue = "#17222E", "#8F9BA7", "#B01E82", "#2358B8"
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
    n = summary["n_compounds"]
    ax[0].plot(n, summary["indeterminate_strict"], "-o", color=grey, label="needs 15 actives and 15 non-hits")
    ax[0].plot(n, summary["indeterminate_chip"], "-o", color=mag, label="needs 5 and 5 (exploratory)")
    ax[0].set(title="Endpoints that cannot be judged", xlabel="compounds tested", ylabel="share of endpoints", ylim=(0, 1.02))
    ax[0].legend(fontsize=7, frameon=False)
    ax[1].plot(n, summary["power_calibrated"], "-o", color=mag, label="true advantage credited")
    ax[1].plot(n, summary["credited_raw_on_null"], "--o", color=grey, label="no advantage credited, raw p")
    ax[1].plot(n, summary["credited_calibrated_on_null"], "-o", color=blue, label="no advantage credited, calibrated p")
    ax[1].axhline(0.05, color=ink, ls=":", lw=0.8)
    ax[1].set(title="Power and false credit", xlabel="compounds tested", ylim=(0, 1.02))
    ax[1].legend(fontsize=7, frameon=False)
    grid = np.arange(20, 401, 5)
    mid = [planned_detectable_effect(int(g), 0.3, curve) for g in grid]
    ax[2].fill_between(grid, [m[1] for m in mid], [m[2] for m in mid], color=mag, alpha=0.15)
    ax[2].plot(grid, [m[0] for m in mid], color=mag, label="planning curve (real endpoints, 30% actives)")
    ax[2].plot(n, summary["median_min_detectable_effect"], "o", color=blue, label="simulation")
    ax[2].set(title="Smallest detectable advantage", xlabel="compounds tested", ylabel="AUROC", ylim=(0, 0.6))
    ax[2].legend(fontsize=7, frameon=False)
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "chip_scale.png", dpi=200)
    plt.close(fig)
    fig, a = plt.subplots(figsize=(4.2, 3.4))
    for i, (name, g) in enumerate(leak.groupby("cv")):
        a.boxplot(g["AUROC"], positions=[i], widths=0.5, patch_artist=True, boxprops=dict(facecolor=[mag, grey][i], alpha=0.6), medianprops=dict(color=ink))
    a.axhline(0.5, color=ink, ls=":", lw=0.8)
    a.set_xticks([0, 1], sorted(leak["cv"].unique()), fontsize=8)
    a.set(title="Replicate chips, noise labels", ylabel="AUROC (true value 0.5)")
    a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "replicate_leakage.png", dpi=200)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sizes", type=int, nargs="*", default=[20, 30, 40, 60, 100, 200])
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3])
    ap.add_argument("--n-null-runs", type=int, default=100)
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--config", default="cellprofiler_allpod", help="real configuration whose detectable_effect.csv defines the planning curve")
    a = ap.parse_args()
    out = RESULTS / "chip_scale"
    out.mkdir(parents=True, exist_ok=True)
    t = chip_scale_study(tuple(a.sizes), tuple(a.seeds), n_null_runs=a.n_null_runs, n_jobs=a.n_jobs)
    t.to_csv(out / "endpoints.csv", index=False)
    summary = summarise_study(t)
    summary.to_csv(out / "summary.csv", index=False)
    leak = replicate_leakage_demo()
    leak.to_csv(out / "leakage.csv", index=False)
    curve = fit_planning_curve(pd.read_csv(RESULTS / a.config / "detectable_effect.csv"))
    curve["config"] = a.config
    curve["examples"] = {f"{n}c_{int(p * 100)}pct": [round(v, 3) for v in planned_detectable_effect(n, p, curve)] for n in (20, 40, 60, 100, 200, 400) for p in (0.2, 0.3, 0.5)}
    (out / "planner_fit.json").write_text(json.dumps(curve, indent=2))
    make_figures(summary, leak, curve, out)
    print(summary.round(3).to_string(index=False))
    print(leak.groupby("cv")["AUROC"].agg(["mean", "std"]).round(3))
    print({k: round(v, 3) if isinstance(v, float) else v for k, v in curve.items() if k != "examples"})


if __name__ == "__main__":
    main()
