"""Cell Painting Shortcut Audit — interactive demo over precomputed results (no live training, CPU only).

Run:  streamlit run demo/app.py
Reads: results/audit_table.csv, results/oof_predictions.parquet, results/enrichment.csv (optional),
       demo/assets/compounds.parquet, demo/assets/dose_response.parquet
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
ASSETS = ROOT / "demo" / "assets"

VERDICT_COLORS = {
    "morphology_advantage": "#2a9d8f",
    "no_advantage": "#8d99ae",
    "indeterminate": "#e9c46a",
    "positive_control": "#e76f51",
}
VERDICT_LABEL = {
    "morphology_advantage": "Morphology advantage",
    "no_advantage": "No advantage over cell count",
    "indeterminate": "Indeterminate (under-powered)",
    "positive_control": "Positive control (cell count)",
}

st.set_page_config(page_title="Cell Painting Shortcut Audit", layout="wide")


@st.cache_data
def load_audit() -> pd.DataFrame:
    return pd.read_csv(RES / "audit_table.csv")


@st.cache_data
def load_oof() -> pd.DataFrame:
    return pd.read_parquet(RES / "oof_predictions.parquet")


@st.cache_data
def load_enrichment():
    p = RES / "enrichment.csv"
    return pd.read_csv(p) if p.exists() else None


@st.cache_data
def load_compounds() -> pd.DataFrame:
    return pd.read_parquet(ASSETS / "compounds.parquet")


@st.cache_data
def load_dose_response() -> pd.DataFrame:
    return pd.read_parquet(ASSETS / "dose_response.parquet")


def badge(verdict: str) -> str:
    return f"<span style='background:{VERDICT_COLORS.get(verdict, '#999')};color:white;padding:3px 10px;border-radius:12px;font-weight:600'>{VERDICT_LABEL.get(verdict, verdict)}</span>"


def reliability_figure(y: np.ndarray, p: np.ndarray, n_bins: int = 8) -> go.Figure:
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    xs, ys, ns = [], [], []
    for b in range(n_bins):
        m = idx == b
        if m.sum() >= 3:
            xs.append(p[m].mean()); ys.append(y[m].mean()); ns.append(int(m.sum()))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="#aaa"), name="perfect"))
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", text=[f"n={n}" for n in ns], name="full model (out-of-fold)"))
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10), xaxis_title="predicted probability", yaxis_title="observed frequency",
                      title="Reliability diagram")
    return fig


audit = load_audit()
config = f"{audit['feature_set'].iloc[0]} features · {audit['agg_method'].iloc[0]} aggregation"

st.title("Cell Painting Shortcut Audit")
st.caption(
    "Does a Cell Painting morphology model beat a model that only *counts surviving cells* — after correcting for testing hundreds of "
    "endpoints at once, and without over-interpreting endpoints that have too few positives? "
    f"Data: Ewald et al. 2026 (primary human hepatocytes, 1,085 compounds). Configuration: **{config}**."
)

counts = audit["verdict"].value_counts()
c = st.columns(4)
for col, v in zip(c, ["morphology_advantage", "no_advantage", "indeterminate", "positive_control"]):
    col.metric(VERDICT_LABEL[v], int(counts.get(v, 0)))

tab_ep, tab_cmp, tab_tbl, tab_enr, tab_img, tab_about = st.tabs(
    ["Endpoint explorer", "Compound explorer", "Audit table", "Where does morphology win?", "Example images", "Method"])

# ------------------------------------------------------------------ endpoint explorer
with tab_ep:
    labels = {r.endpoint_id: f"{r.endpoint_id}  ·  {r.verdict}" for r in audit.itertuples()}
    ep = st.selectbox("Endpoint", list(labels), format_func=lambda e: labels[e], key="ep_sel")
    r = audit[audit["endpoint_id"] == ep].iloc[0]
    st.markdown(badge(r["verdict"]), unsafe_allow_html=True)
    desc = r.get("endpoint_description")
    st.write(f"**Category:** {r.get('category')} · **Target family:** {r.get('assay_target_family')} · **Assay design:** {r.get('assay_design_type')}")
    if isinstance(desc, str):
        st.caption(desc[:400])
    left, right = st.columns([1, 1])
    with left:
        st.write(f"**{int(r['n_compounds'])} compounds** — {int(r['n_active'])} active, {int(r['n_inactive'])} non-hit "
                 f"({'powered' if r['powered'] else 'UNDER-POWERED: needs ≥15 of each class'}).")
        if pd.notna(r.get("delta_AUROC")):
            bars = pd.DataFrame({
                "model": ["cell count (scalar)", "cell count (dose–response curve)", "full morphology profile"],
                "AUROC": [r["scalar_cc_AUROC"], r["strong_cc_AUROC"], r["full_AUROC"]],
            })
            fig = px.bar(bars, x="AUROC", y="model", orientation="h", range_x=[0.4, 1.0], text=bars["AUROC"].round(3))
            fig.update_layout(height=250, margin=dict(l=10, r=10, t=10, b=10), yaxis_title=None)
            st.plotly_chart(fig, width="stretch")
            st.write(f"**ΔAUROC (full − strong cell-count baseline) = {r['delta_AUROC']:+.3f}** "
                     f"(95% bootstrap CI {r['delta_ci_lo']:+.3f} to {r['delta_ci_hi']:+.3f}); z = {r['delta_z']:.2f}; "
                     f"null-calibrated one-sided p = {r['calibrated_p']:.4f} (raw bootstrap p = {r['bootstrap_p']:.4f}); "
                     f"BH-FDR q = {'n/a' if pd.isna(r['fdr_q']) else format(r['fdr_q'], '.4f')}.")
            st.caption("The raw bootstrap p-value is anti-conservative (label-permutation control: 13% false positives at the nominal 5% for CellProfiler features), "
                       "so verdicts use p-values calibrated against that empirical null.")
            st.caption(f"Versus the paper's simpler scalar cell-count baseline: Δ = {r['delta_AUROC_scalar']:+.3f}, verdict = {r['verdict_scalar']}.")
        else:
            st.info("Too few compounds in one class to run cross-validation — reported as indeterminate.")
    with right:
        oof = load_oof()
        o = oof[oof["endpoint_id"] == ep]
        if len(o):
            st.plotly_chart(reliability_figure(o["y"].to_numpy(), o["p_full"].to_numpy()), width="stretch")
            st.caption(f"Brier {r['calibration_brier']:.3f} (skill vs. prevalence {r['brier_skill']:+.2f}) · ECE {r['ece']:.3f} · "
                       f"calibration slope {r['calib_slope']:.2f} (<1 = over-confident). Raw probabilities use class re-weighting, "
                       "so the raw scores are shown here; the compound view uses recalibrated probabilities.")

# ------------------------------------------------------------------ compound explorer
with tab_cmp:
    comps = load_compounds().dropna(subset=["name"]).sort_values("name")
    options = {f"{n}  ({i})": i for n, i in zip(comps["name"], comps["OASIS_ID"])}
    pick = st.selectbox("Compound (type to search)", list(options), key="cmp_sel")
    oid = options[pick]
    crow = comps[comps["OASIS_ID"] == oid].iloc[0]
    st.write(f"**{crow['name']}** · CASRN {crow['CASRN']} · {crow['DTXSID']}")
    hits = {k.replace("_", " "): crow[k] for k in ["Cell_count_hit", "MT_hit", "LDH_hit", "Cell_Painting_hit"]}
    st.write(" · ".join(f"{k}: **{v}**" for k, v in hits.items()))

    dr = load_dose_response()
    d = dr[dr["OASIS_ID"] == oid].sort_values("concentration_uM")
    if len(d):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=d["concentration_uM"], y=d["cell_count_ratio"], name="cell count (÷ DMSO)", mode="lines+markers"))
        fig.add_trace(go.Scatter(x=d["concentration_uM"], y=d["mt"], name="MT (metabolic activity)", mode="lines+markers"))
        fig.add_trace(go.Scatter(x=d["concentration_uM"], y=1 - d["ldh"].clip(upper=1), name="1 − LDH release", mode="lines+markers"))
        fig.update_xaxes(type="log", title="concentration (µM)")
        fig.update_layout(height=330, margin=dict(l=10, r=10, t=30, b=10), title="Measured readouts across the 8-point concentration series")
        st.plotly_chart(fig, width="stretch")

    oof = load_oof()
    pc = oof[oof["OASIS_ID"] == oid].merge(audit[["endpoint_id", "category", "assay_target_family", "verdict", "delta_AUROC", "fdr_q", "full_AUROC", "ece"]], on="endpoint_id")
    st.subheader("Out-of-fold model predictions for this compound")
    st.caption("Each probability comes from a model that never saw this compound (compound-grouped cross-validation), then recalibrated "
               "per endpoint with cross-validated Platt scaling (raw model scores are over-confident). "
               "The verdict describes the *endpoint*: whether morphology reliably beats cell count for it.")
    vf = st.multiselect("Show verdicts", list(VERDICT_LABEL), default=["morphology_advantage", "no_advantage"], format_func=lambda v: VERDICT_LABEL[v])
    show = pc[pc["verdict"].isin(vf)].copy()
    show["measured"] = show["y"].map({1: "active", 0: "non-hit"})
    mcol, bcol = ("p_full_cal", "p_strong_cc_cal") if "p_full_cal" in show else ("p_full", "p_strong_cc")
    show = show.rename(columns={mcol: "P(active) morphology", bcol: "P(active) cell-count only"})
    cols = ["endpoint_id", "category", "assay_target_family", "measured", "P(active) morphology", "P(active) cell-count only", "verdict", "delta_AUROC", "fdr_q", "ece"]
    st.dataframe(show[cols].sort_values("fdr_q"), width="stretch", hide_index=True,
                 column_config={"P(active) morphology": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f"),
                                "P(active) cell-count only": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f")})

# ------------------------------------------------------------------ audit table
with tab_tbl:
    scored = audit[audit["delta_AUROC"].notna()]
    fig = px.scatter(scored, x="strong_cc_AUROC", y="full_AUROC", color="verdict", color_discrete_map=VERDICT_COLORS,
                     hover_name="endpoint_id", hover_data=["category", "assay_target_family", "n_active", "delta_AUROC", "fdr_q"], height=520)
    fig.add_trace(go.Scatter(x=[0.3, 1], y=[0.3, 1], mode="lines", line=dict(dash="dash", color="#aaa"), showlegend=False))
    fig.update_layout(xaxis_title="AUROC — cell-count-only baseline", yaxis_title="AUROC — full morphology profile")
    st.plotly_chart(fig, width="stretch")
    f1, f2 = st.columns(2)
    cat = f1.multiselect("Category", sorted(audit["category"].dropna().unique()), default=sorted(audit["category"].dropna().unique()))
    ver = f2.multiselect("Verdict", list(VERDICT_LABEL), default=list(VERDICT_LABEL), format_func=lambda v: VERDICT_LABEL[v])
    t = audit[audit["category"].isin(cat) & audit["verdict"].isin(ver)]
    show_cols = ["endpoint_id", "category", "assay_target_family", "n_active", "n_inactive", "scalar_cc_AUROC", "strong_cc_AUROC", "full_AUROC",
                 "delta_AUROC", "delta_z", "bootstrap_p", "calibrated_p", "fdr_q", "verdict", "verdict_uncalibrated", "verdict_scalar", "calibration_brier", "ece"]
    st.dataframe(t[show_cols].sort_values("fdr_q"), width="stretch", hide_index=True)
    st.download_button("Download audit_table.csv", audit.to_csv(index=False), "audit_table.csv", "text/csv")

# ------------------------------------------------------------------ enrichment
with tab_enr:
    enr = load_enrichment()
    st.write("One-sided Fisher exact tests: is an assay / target family over-represented among 'morphology advantage' endpoints "
             "(vs. powered endpoints with no advantage)? q-values are BH-adjusted within each grouping.")
    st.caption("Caution: cytotoxicity endpoints have about four times more actives than cell-based endpoints (median 97.5 vs 25), so they are easier to certify. "
               "Restricted to endpoints with at least 50 actives the cytotoxicity enrichment disappears for CellProfiler and CP-CNN (odds ratio ≈ 1) and is undecided for DINOv2; "
               "read these tables as descriptive, not as evidence of a mechanism.")
    if enr is None:
        st.info("Run `python scripts/04_run_enrichment.py` to generate results/enrichment.csv.")
    else:
        base = st.radio("Baseline used for the verdicts", sorted(enr["baseline"].unique()), horizontal=True)
        by = st.selectbox("Group by", sorted(enr["by"].unique()), index=sorted(enr["by"].unique()).index("assay_target_family") if "assay_target_family" in set(enr["by"]) else 0)
        e = enr[(enr["baseline"] == base) & (enr["by"] == by)].sort_values("p")
        st.dataframe(e.drop(columns=["baseline", "by"]), width="stretch", hide_index=True)
        if (e["q"] < 0.05).any():
            st.success(f"{int((e['q'] < 0.05).sum())} group(s) significantly enriched at q < 0.05.")
        else:
            st.warning("No group is significantly enriched at q < 0.05 — morphology's advantage is not explained by this grouping.")

    sens = RES / "sensitivity_summary.csv"
    if sens.exists():
        st.subheader("Robustness across representations and aggregation rules")
        st.caption("Number of endpoints certified as 'morphology advantage' (BH q < 0.05, null-calibrated p) per configuration, versus the strong and the paper's scalar "
                   "cell-count baseline, next to the counts a plain bootstrap would give.")
        sm = pd.read_csv(sens)[["config", "powered", "adv_vs_strong", "adv_vs_scalar", "adv_vs_strong_raw_bootstrap", "adv_vs_scalar_raw_bootstrap", "null_sd_used"]]
        st.dataframe(sm, width="stretch", hide_index=True)

# ------------------------------------------------------------------ example images
with tab_img:
    manifest_path = ASSETS / "images" / "manifest.csv"
    st.write("Cell Painting Gallery images (cpg0037-oasis, CC0) for a few featured compounds: a well at the highest tested concentration next to a "
             "DMSO control from the same plate. Fetched individually via `index.parquet` (no bulk download) by `scripts/14_fetch_example_images.py`. "
             "Channels: DNA blue, ER green, AGP red, Mito magenta, RNA cyan.")
    if not manifest_path.exists():
        st.info("Run `python scripts/14_fetch_example_images.py` to fetch the example images.")
    else:
        man = pd.read_csv(manifest_path)
        names = load_compounds().set_index("OASIS_ID")
        feat = man.drop_duplicates("OASIS_ID")
        group_label = {"hit_in_all_assays": "active in every assay", "cell_painting_only": "Cell Painting active only (no cell loss, MT or LDH change)", "inactive": "inactive in all readouts"}
        opt = {f"{names.loc[o, 'name'] if o in names.index else o} — {group_label[g]}": o for o, g in zip(feat["OASIS_ID"], feat["group"])}
        pick_img = st.selectbox("Featured compound", list(opt), key="img_sel")
        o = opt[pick_img]
        c1, c2 = st.columns(2)
        for col, kind, title in ((c1, "treated", "Treated"), (c2, "dmso", "DMSO control")):
            r = man[(man["OASIS_ID"] == o) & (man["kind"] == kind)]
            if len(r) and (ASSETS / "images" / r.iloc[0]["file"]).exists():
                conc = f" · {r.iloc[0]['concentration_uM']:.3g} µM" if kind == "treated" else ""
                col.image(str(ASSETS / "images" / r.iloc[0]["file"]), caption=f"{title}{conc} · plate {r.iloc[0]['plate']} well {r.iloc[0]['well']}")

# ------------------------------------------------------------------ method
with tab_about:
    st.markdown(
        """
**Question per endpoint.** (1) Does the full morphological profile beat a cell-count-only model, after Benjamini–Hochberg correction across all
endpoints tested? (2) Is there enough data to say (≥15 actives and ≥15 non-hits) — otherwise the endpoint is *indeterminate*, never a
silent AUROC? (3) Where morphology wins, is the advantage concentrated in a biologically coherent assay / target family?

**Models.** XGBoost (150 trees, lr 0.05, class-reweighted — as in the source paper). *Full*: normalised morphology features. *Cell-count baselines*: the
paper's scalar mean cell count, and a stronger dose–response version (cell count at each of 8 concentrations, minimum, AUC, cell-count POD) — still
carrying **only** cell-count information.

**Validation.** Compound-grouped, stratified 5-fold CV repeated 3×; ΔAUROC from pooled out-of-fold predictions; one-sided compound-bootstrap z-scores. A **label-permutation control**
(hundreds of runs with shuffled labels) shows the raw bootstrap p-values are anti-conservative, so p-values are **calibrated against the empirical null** (Efron-style) before
BH-FDR over powered endpoints only; calibration (Brier, ECE, slope) of the full model is reported per endpoint.

**Reading the labels.** A ToxCast 0 means *not a (filtered) hit* — no hit, a hit removed by the cytotoxicity filter, or a tie — never "tested negative"; untested pairs are missing, not 0; for the cytotoxicity columns 1 = cytotoxic in that cell type or tissue. The cytotoxicity filter applies to only ~35 % of cell-based records.

**Not claimed.** This is a 2D plate assay, not an organ-on-chip; single 44 h time point; donor variability is not modeled. The audit method is dataset-agnostic and is
meant to be re-run on any phenotypic-profiling toxicology dataset that has a cell-count-like baseline.
"""
    )
