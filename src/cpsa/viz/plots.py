"""Static figures for the technical report (matplotlib, colour-blind-safe palette)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = {"morphology_advantage": "#1b9e77", "no_advantage": "#7f7f7f", "indeterminate": "#e6ab02", "positive_control": "#d95f02"}
LABELS = {"morphology_advantage": "morphology advantage", "no_advantage": "no advantage", "indeterminate": "indeterminate", "positive_control": "positive control"}


def _save(fig, path: Path | None):
    fig.tight_layout()
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=200)
        plt.close(fig)
    return fig


def scatter_baseline_vs_full(audit: pd.DataFrame, path: Path | None = None, x: str = "strong_cc_AUROC", xlabel: str = "AUROC, cell-count-only baseline"):
    d = audit[audit[x].notna()]
    fig, ax = plt.subplots(figsize=(5.2, 5))
    ax.plot([0.3, 1], [0.3, 1], "--", color="#bbb", lw=1)
    for v, g in d.groupby("verdict"):
        ax.scatter(g[x], g["full_AUROC"], s=18 + 3 * np.sqrt(g["n_active"]), c=COLORS.get(v, "k"), alpha=0.75, label=f"{LABELS.get(v, v)} (n={len(g)})", edgecolor="white", lw=0.4)
    ax.set_xlabel(xlabel); ax.set_ylabel("AUROC, full morphology profile")
    ax.legend(frameon=False, fontsize=8, loc="upper left"); ax.set_aspect("equal")
    return _save(fig, path)


def delta_forest(audit: pd.DataFrame, path: Path | None = None, top: int = 40):
    d = audit[audit["verdict"].isin(["morphology_advantage", "no_advantage"])].sort_values("delta_AUROC", ascending=False)
    d = pd.concat([d.head(top // 2), d.tail(top // 2)]).drop_duplicates("endpoint_id") if len(d) > top else d
    fig, ax = plt.subplots(figsize=(8, max(3, 0.2 * len(d))))
    y = np.arange(len(d))[::-1]
    ax.hlines(y, d["delta_ci_lo"], d["delta_ci_hi"], color=[COLORS[v] for v in d["verdict"]], lw=1.5)
    ax.scatter(d["delta_AUROC"], y, c=[COLORS[v] for v in d["verdict"]], s=14, zorder=3)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(y); ax.set_yticklabels(d["endpoint_id"], fontsize=6)
    ax.set_xlabel("ΔAUROC (full − cell-count baseline), 95% bootstrap CI  [green = BH q < 0.05]")
    ax.set_ylim(-1, len(d))
    return _save(fig, path)


def correction_funnel(audit: pd.DataFrame, path: Path | None = None):
    """How many 'wins' survive: naive Δ>0 → powered → p<0.05 uncorrected → BH (raw bootstrap) → BH (null-calibrated)."""
    d = audit[~audit["verdict"].isin(["positive_control"])]
    powered = d[d["powered"]]
    raw_col = "verdict_uncalibrated" if "verdict_uncalibrated" in d else "verdict"
    steps = [("tested endpoints", len(d)), ("Δ AUROC > 0 (naive 'win')", int((d["delta_AUROC"] > 0).sum())),
             ("… and powered (≥15 / ≥15)", int((powered["delta_AUROC"] > 0).sum())),
             ("… and raw p < 0.05, uncorrected", int(((powered["delta_AUROC"] > 0) & (powered["bootstrap_p"] < 0.05)).sum())),
             ("… and BH q < 0.05 (raw bootstrap p)", int((d[raw_col] == "morphology_advantage").sum()))]
    if "verdict_uncalibrated" in d:
        steps.append(("… and BH q < 0.05 (null-calibrated p)", int((d["verdict"] == "morphology_advantage").sum())))
    fig, ax = plt.subplots(figsize=(6.6, 3.0))
    cols = ["#9aa0a6"] * (len(steps) - 1) + ["#1b9e77"]
    ax.barh([s[0] for s in steps][::-1], [s[1] for s in steps][::-1], color=cols[::-1])
    for i, (_, v) in enumerate(steps[::-1]):
        ax.text(v + 2, i, str(v), va="center", fontsize=9)
    ax.set_xlabel("number of endpoints")
    return _save(fig, path)


def verdicts_by_category(audit: pd.DataFrame, path: Path | None = None):
    t = audit.groupby(["category", "verdict"]).size().unstack(fill_value=0)
    t = t[[c for c in COLORS if c in t.columns]]
    fig, ax = plt.subplots(figsize=(6, 3))
    t.plot.barh(stacked=True, ax=ax, color=[COLORS[c] for c in t.columns], width=0.7)
    ax.set_xlabel("endpoints"); ax.set_ylabel(""); ax.legend(frameon=False, fontsize=8)
    return _save(fig, path)


def pooled_reliability(oof: pd.DataFrame, audit: pd.DataFrame, path: Path | None = None, n_bins: int = 10):
    ok = audit.loc[audit["powered"] & ~audit["endpoint_id"].isin(["cell_count"]), "endpoint_id"]
    o = oof[oof["endpoint_id"].isin(ok)]
    p, y = o["p_full"].to_numpy(), o["y"].to_numpy()
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    xs = [p[idx == b].mean() for b in range(n_bins) if (idx == b).any()]
    ys = [y[idx == b].mean() for b in range(n_bins) if (idx == b).any()]
    fig, ax = plt.subplots(figsize=(4.2, 4))
    ax.plot([0, 1], [0, 1], "--", color="#bbb"); ax.plot(xs, ys, "o-", color=COLORS["morphology_advantage"])
    ax.set_xlabel("predicted probability (out-of-fold)"); ax.set_ylabel("observed frequency"); ax.set_title("Full model, all powered endpoints pooled", fontsize=9)
    return _save(fig, path)


def enrichment_bars(enr: pd.DataFrame, path: Path | None = None, by: str = "assay_target_family", baseline: str = "vs_strong_cc"):
    e = enr[(enr["by"] == by) & (enr["baseline"] == baseline)].sort_values("frac_adv_in")
    fig, ax = plt.subplots(figsize=(6, max(2.5, 0.28 * len(e))))
    cols = ["#1b9e77" if q < 0.05 else "#9aa0a6" for q in e["q"]]
    ax.barh(e["group"], e["frac_adv_in"], color=cols)
    for i, (n, a, q) in enumerate(zip(e["n_in_group"], e["n_adv_in_group"], e["q"])):
        ax.text(e["frac_adv_in"].iloc[i] + 0.01, i, f"{a}/{n}  q={q:.2g}", va="center", fontsize=7)
    base = (e["n_adv_in_group"].sum() + 0) / max(e["n_in_group"].sum(), 1)
    ax.axvline(base, color="k", ls=":", lw=1)
    ax.set_xlabel("fraction of powered endpoints with morphology advantage"); ax.set_xlim(0, 1.15)
    return _save(fig, path)


def null_pvalues(null: pd.DataFrame, path: Path | None = None):
    """Label-permutation control (powered runs only): raw p-value histogram, z-scores vs. N(0,1) and the empirical null."""
    from scipy.stats import norm

    from cpsa.stats.null_calibration import z_scores

    n = null[(null["n_active"] >= 15) & (null["n_inactive"] >= 15)]
    z = z_scores(n["delta_AUROC"], n["delta_ci_lo"], n["delta_ci_hi"])
    sd = float(np.std(z, ddof=1))
    fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.1))
    ax[0].hist(n["bootstrap_p"], bins=np.linspace(0, 1, 11), color="#7f7f7f", edgecolor="white")
    ax[0].axhline(len(n) / 10, color="k", ls="--", lw=1)
    ax[0].set_xlabel("raw bootstrap p (permuted labels)"); ax[0].set_ylabel("runs")
    ax[0].set_title(f"type-I at 0.05: {(n['bootstrap_p'] < 0.05).mean():.1%}", fontsize=9)
    xs = np.linspace(-5, 5, 200)
    ax[1].hist(z, bins=30, density=True, color="#bbbbbb", edgecolor="white")
    ax[1].plot(xs, norm.pdf(xs), color="#1b9e77", label="N(0, 1)")
    ax[1].plot(xs, norm.pdf(xs, scale=sd), color="#d95f02", label=f"N(0, {sd:.2f}²)")
    ax[1].set_xlabel("z = ΔAUROC / bootstrap SE"); ax[1].legend(frameon=False, fontsize=8)
    ax[2].hist(n["full_AUROC"], bins=20, color="#7f7f7f", alpha=0.8, label="full"); ax[2].hist(n["strong_cc_AUROC"], bins=20, color="#e6ab02", alpha=0.6, label="strong cc")
    ax[2].axvline(0.5, color="k", lw=1); ax[2].set_xlabel("AUROC (permuted labels)"); ax[2].legend(frameon=False, fontsize=8)
    return _save(fig, path)


def paper_agreement(cmp_: pd.DataFrame, path: Path | None = None):
    fig, ax = plt.subplots(1, 2, figsize=(7.5, 3.6))
    for a, (o, p, t) in zip(ax, (("full_AUROC", "paper_Actual_AUROC", "full model"), ("scalar_cc_AUROC", "paper_Cellcount_baseline_AUROC", "cell-count baseline"))):
        a.scatter(cmp_[p], cmp_[o], s=10, alpha=0.6, color="#1b9e77"); a.plot([0.3, 1], [0.3, 1], "--", color="#bbb")
        a.set_xlabel("paper AUROC"); a.set_ylabel("this pipeline AUROC"); a.set_title(t, fontsize=9)
    return _save(fig, path)


def pipeline_diagram(path: Path | None = None):
    """Schematic of the audit pipeline (for the report and the demo video)."""
    from matplotlib.patches import FancyBboxPatch

    fig, ax = plt.subplots(figsize=(10, 4.4))
    ax.set_xlim(0, 10); ax.set_ylim(0, 4.4); ax.axis("off")

    def box(x, y, w, h, text, fc="#eef3f7", ec="#456", fs=8.5):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.08", fc=fc, ec=ec, lw=1.1))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", color="#456", lw=1.2))

    box(0.1, 3.2, 2.0, 0.9, "Well-level profiles\n(CellProfiler / CP-CNN / DINOv2)\n+ cell counts", "#fdf2e0")
    box(0.1, 1.9, 2.0, 0.9, "ToxCast / Tox21 hit-calls\n402 endpoints + MT, LDH\n(authors' repo)", "#fdf2e0")
    box(2.6, 2.7, 1.9, 1.0, "DMSO-MAD normalisation\ncorrelation filter\nper-compound aggregation")
    box(5.0, 3.35, 2.1, 0.75, "FULL: morphology profile", "#d9f0e6", "#1b9e77")
    box(5.0, 2.45, 2.1, 0.75, "STRONG cell-count baseline\n(dose–response curve)", "#f3e2c8", "#b8860b")
    box(5.0, 1.55, 2.1, 0.75, "SCALAR cell-count baseline\n(paper's one number)", "#f3e2c8", "#b8860b")
    box(7.6, 2.45, 2.2, 1.0, "grouped 5-fold CV ×3\nΔAUROC + paired\ncompound bootstrap")
    box(7.6, 0.2, 2.2, 0.9, "power check\n≥15 actives & inactives\nelse INDETERMINATE", "#fbe3e3", "#b33")
    box(5.0, 0.2, 2.1, 0.9, "BH-FDR over powered\nendpoints → verdict", "#d9f0e6", "#1b9e77")
    box(2.6, 0.2, 2.0, 0.9, "calibration · enrichment ·\nnull control · demo")
    arrow(2.1, 3.65, 2.6, 3.3); arrow(2.1, 2.35, 2.6, 2.9)
    for y in (3.72, 2.82, 1.92):
        arrow(4.5, 3.2, 5.0, y)
    for y in (3.72, 2.82, 1.92):
        arrow(7.1, y, 7.6, 2.95)
    arrow(8.7, 2.45, 8.7, 1.1); arrow(7.6, 0.65, 7.1, 0.65)
    arrow(5.0, 0.65, 4.6, 0.65)
    ax.text(5, 4.3, "Cell Painting Shortcut Audit", ha="center", fontsize=11, fontweight="bold")
    return _save(fig, path)
