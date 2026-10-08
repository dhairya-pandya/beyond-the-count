"""Cell Painting Shortcut Audit: interactive demo over precomputed results (no live training, CPU only).

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
REPO_URL = "https://github.com/dhairya-pandya/beyond-the-count"

# Palette: two stain colours from the Cell Painting dyes (mitochondria magenta, DNA blue) on a cool paper tone.
INK, SLATE, RULE, PAPER = "#17222E", "#5E6B78", "#D5DBE0", "#F4F6F7"
MAGENTA, BLUE, GREY, RING = "#B01E82", "#2358B8", "#8F9BA7", "#AEB7C0"
SERIF = "Newsreader, Iowan Old Style, Georgia, serif"
SANS = "Hanken Grotesk, Helvetica Neue, Arial, sans-serif"

VERDICT_ORDER = ["morphology_advantage", "positive_control", "no_advantage", "indeterminate"]
VERDICT_COLORS = {"morphology_advantage": MAGENTA, "positive_control": BLUE, "no_advantage": GREY, "indeterminate": RING}
VERDICT_LABEL = {
    "morphology_advantage": "Morphology advantage",
    "positive_control": "Positive control",
    "no_advantage": "No advantage over cell count",
    "indeterminate": "Indeterminate, too few positives",
}
VERDICT_SHORT = {"morphology_advantage": "advantage", "positive_control": "control", "no_advantage": "no advantage",
                 "indeterminate": "indeterminate"}

st.set_page_config(page_title="Cell Painting Shortcut Audit", page_icon="🔬", layout="wide")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Hanken+Grotesk:wght@400;500;600;700&family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;0,6..72,600;1,6..72,400&display=swap');

html, body, .stApp, [data-testid="stAppViewContainer"] { background: %PAPER%; }
html, body, .stApp, .stApp p, .stApp li, .stApp label, .stApp span, .stApp div { font-family: %SANS%; }
[data-testid="stIconMaterial"], .stApp [data-testid="stIconMaterial"] { font-family: 'Material Symbols Rounded', 'Material Symbols Outlined' !important; }
.block-container { max-width: 1160px; padding: 3.2rem 1.5rem 4rem; }
@media (max-width: 640px) { .block-container { padding: 2rem 1rem 3rem; } }

/* type scale: serif for headings and the lede, grotesk for everything you operate */
.stApp h1, .stApp h2, .stApp h3, .stApp h1 span, .stApp h2 span, .stApp h3 span { font-family: %SERIF% !important; color: %INK%; letter-spacing: -0.012em; font-weight: 500; }
.hero h1 { font-size: clamp(2.1rem, 4.6vw, 3.3rem); line-height: 1.08; margin: 0 0 1rem; max-width: 18ch; }
.stApp .hero .lede { font-family: %SERIF%; font-size: 1.28rem; line-height: 1.55; color: %INK%; max-width: 60ch; margin: 0 0 0.4rem; }
h2, h3 { margin-top: 1.6rem; }
h3 { font-size: 1.45rem; }
.stApp p, .stApp li { font-size: 1.02rem; line-height: 1.6; }
[data-testid="stMarkdownContainer"] p { max-width: 72ch; }
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p { font-size: 0.93rem; color: %SLATE%; line-height: 1.55; }

/* finding line + legend that doubles as the headline numbers */
.stApp .finding { font-family: %SANS%; font-weight: 600; font-size: 1.08rem; line-height: 1.5; margin: 2.2rem 0 0.4rem; max-width: 64ch; }
.legend { display: flex; flex-wrap: wrap; gap: 0.5rem 1.8rem; margin: 0.4rem 0 0.1rem; }
.legend .item { display: flex; align-items: baseline; gap: 0.5rem; font-size: 0.98rem; }
.legend .num { font-weight: 700; font-size: 1.35rem; font-variant-numeric: tabular-nums; }
.dot { display: inline-block; width: 0.8rem; height: 0.8rem; border-radius: 50%; flex: none; transform: translateY(1px); }
.dot.ring { background: transparent; border: 2px solid %RING%; }

/* verdict marker */
.verdict { display: inline-flex; align-items: center; gap: 0.55rem; font-weight: 600; font-size: 1.05rem; margin: 0.2rem 0 0.4rem; }
.verdict i { width: 0.85rem; height: 0.85rem; border-radius: 50%; display: inline-block; }
.stApp .answer { font-family: %SERIF%; font-size: 1.2rem; line-height: 1.5; max-width: 58ch; margin: 0.2rem 0 1rem; }
.stApp .facts { color: %SLATE%; font-size: 0.97rem; margin: 0 0 0.9rem; }
.facts b { color: %INK%; font-weight: 600; }
.chips { display: flex; flex-wrap: wrap; gap: 0.5rem; margin: 0.3rem 0 1rem; }
.chip { border: 1.5px solid %RULE%; border-radius: 999px; padding: 0.3rem 0.9rem; font-size: 0.95rem; background: white; }
.chip.on { border-color: %MAGENTA%; color: %MAGENTA%; font-weight: 600; }

/* controls: 44px minimum hit area, readable text, visible focus */
.stApp [role="tab"] { min-height: 48px; padding: 0.6rem 1.15rem; }
.stApp [role="tab"] p { font-size: 1.02rem; font-weight: 600; }
[data-baseweb="tab-list"] { gap: 0.2rem; overflow-x: auto; }
[data-baseweb="tab-highlight"], .react-aria-SelectionIndicator { display: none !important; }
.stApp [role="tab"] { border-bottom: 3px solid transparent; }
.stApp [role="tab"][aria-selected="true"] { border-bottom-color: %MAGENTA%; }
[data-baseweb="select"] > div { min-height: 46px; font-size: 1rem; background: white; }
[data-testid="stButtonGroup"] button, .stButton button, .stDownloadButton button {
  min-height: 44px; padding: 0.55rem 1.1rem; font-size: 0.98rem; font-weight: 600; border-radius: 8px; }
[data-testid="stButtonGroup"] { gap: 0.4rem; flex-wrap: wrap; }
.stDownloadButton button { border: 1.5px solid %INK%; background: white; color: %INK%; }
.stDownloadButton button:hover { background: %INK%; color: white; border-color: %INK%; }
[data-testid="stWidgetLabel"] p { font-size: 0.97rem; font-weight: 600; color: %INK%; }
button:focus-visible, [data-baseweb="select"] input:focus-visible { outline: 3px solid %BLUE% !important; outline-offset: 2px; }
[data-testid="stExpander"] summary { min-height: 46px; font-weight: 600; }
[data-testid="stExpander"] { border-color: %RULE%; background: white; }

.footer { margin-top: 3rem; padding-top: 1rem; border-top: 1px solid %RULE%; color: %SLATE%; font-size: 0.93rem; }
.footer a { color: %INK%; }
@media (prefers-reduced-motion: reduce) { * { transition: none !important; animation: none !important; } }
</style>
"""
for _k, _v in {"PAPER": PAPER, "SANS": SANS, "SERIF": SERIF, "INK": INK, "SLATE": SLATE, "RULE": RULE, "RING": RING,
               "MAGENTA": MAGENTA, "BLUE": BLUE}.items():
    CSS = CSS.replace(f"%{_k}%", _v)
st.markdown(CSS, unsafe_allow_html=True)


# ------------------------------------------------------------------ data
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


# ------------------------------------------------------------------ figure helpers
def style(fig: go.Figure, height: int, **layout) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=8, r=8, t=36, b=8), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=SANS, color=INK, size=13), title_font=dict(family=SERIF, size=17, color=INK),
        hoverlabel=dict(font_family=SANS, bgcolor="white", font_color=INK), legend=dict(orientation="h", y=0, yref="container", yanchor="bottom"), **layout)
    fig.update_xaxes(gridcolor=RULE, zerolinecolor=RULE, linecolor=RULE, tickfont=dict(color=SLATE))
    fig.update_yaxes(gridcolor=RULE, zerolinecolor=RULE, linecolor=RULE, tickfont=dict(color=SLATE), automargin=True)
    fig.update_xaxes(automargin=True)
    return fig


def plate_map(audit: pd.DataFrame, current: str) -> tuple[go.Figure, list[str]]:
    """One dot per endpoint, laid out like a multi-well plate and grouped by verdict."""
    parts = []
    for v in VERDICT_ORDER:
        d = audit[audit["verdict"] == v]
        parts.append(d.sort_values("n_active" if v == "indeterminate" else "delta_AUROC", ascending=False, na_position="last"))
    o = pd.concat(parts).reset_index(drop=True)
    rows = 15
    cols = -(-len(o) // rows)
    x, y = o.index % cols, o.index // cols
    detail = [f"change in AUROC {d:+.2f}" if pd.notna(d) else f"{int(a)} active compounds" for d, a in zip(o["delta_AUROC"], o["n_active"])]
    fill = [PAPER if v == "indeterminate" else VERDICT_COLORS[v] for v in o["verdict"]]
    line = [VERDICT_COLORS[v] for v in o["verdict"]]
    size = [11 if v in ("morphology_advantage", "positive_control") else 10 if v == "no_advantage" else 9 for v in o["verdict"]]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="markers", name="endpoints", showlegend=False,
        marker=dict(color=fill, size=size, line=dict(color=line, width=1.5)),
        customdata=np.stack([o["endpoint_id"], o["verdict"].map(VERDICT_LABEL), detail], axis=1),
        hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]}<br>%{customdata[2]}<extra></extra>",
        selected=dict(marker=dict(opacity=1)), unselected=dict(marker=dict(opacity=1))))
    cur = o.index[o["endpoint_id"] == current]
    if len(cur):
        fig.add_trace(go.Scatter(x=x[cur], y=y[cur], mode="markers", hoverinfo="skip", showlegend=False,
                                 marker=dict(size=20, color="rgba(0,0,0,0)", line=dict(color=INK, width=2.5)),
                                 selected=dict(marker=dict(opacity=1)), unselected=dict(marker=dict(opacity=1))))
    fig.update_xaxes(visible=False, range=[-0.8, cols - 0.2], fixedrange=True)
    fig.update_yaxes(visible=False, range=[rows - 0.3, -0.7], fixedrange=True)
    fig.update_layout(height=330, margin=dict(l=0, r=0, t=6, b=6), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      hoverlabel=dict(font_family=SANS, bgcolor="white", font_color=INK), dragmode=False, clickmode="event+select")
    return fig, o["endpoint_id"].tolist()


def reliability_figure(y: np.ndarray, p: np.ndarray, n_bins: int = 8) -> go.Figure:
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    xs, ys, ns = [], [], []
    for b in range(n_bins):
        m = idx == b
        if m.sum() >= 3:
            xs.append(p[m].mean()); ys.append(y[m].mean()); ns.append(int(m.sum()))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color=GREY), name="perfectly calibrated"))
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", text=[f"{n} compounds" for n in ns], name="morphology model",
                             line=dict(color=MAGENTA, width=2.5), marker=dict(size=9, color=MAGENTA)))
    style(fig, 360, title="Do probabilities match reality?")
    fig.update_layout(margin=dict(l=8, r=8, t=36, b=64))
    fig.update_xaxes(title="predicted probability of active", range=[0, 1])
    fig.update_yaxes(title="share actually active", range=[0, 1])
    return fig


def verdict_marker(verdict: str) -> str:
    color = VERDICT_COLORS.get(verdict, GREY)
    ring = f"background:transparent;border:2px solid {color}" if verdict == "indeterminate" else f"background:{color}"
    return f"<div class='verdict'><i style='{ring}'></i>{VERDICT_LABEL.get(verdict, verdict)}</div>"


def answer_sentence(r: pd.Series, n_powered: int) -> str:
    v = r["verdict"]
    if v == "indeterminate":
        return (f"Too few examples to judge. Only {int(r['n_active'])} compounds are active and {int(r['n_inactive'])} are non-hits, "
                "and we need at least 15 of each, so we make no claim for this endpoint.")
    q = "" if pd.isna(r["fdr_q"]) else f" (q = {r['fdr_q']:.3f})"
    if v == "morphology_advantage":
        return (f"Morphology adds real information here. The full profile scores AUROC {r['full_AUROC']:.2f}; "
                f"a model that only knows cell counts scores {r['strong_cc_AUROC']:.2f}. "
                f"The gap survives correction for testing {n_powered} endpoints{q}.")
    if v == "no_advantage":
        return (f"Morphology does not reliably beat cell counts here. The full profile scores AUROC {r['full_AUROC']:.2f} "
                f"and the cell-count-only model {r['strong_cc_AUROC']:.2f}; the difference is not certified after correcting "
                f"for testing {n_powered} endpoints{q}.")
    return (f"This endpoint is defined from cell count itself, so a cell-count model should score near 1.00 and does "
            f"({r['strong_cc_AUROC']:.2f}). It checks that the audit can spot a pure shortcut.")


# ------------------------------------------------------------------ header
audit = load_audit()
counts = audit["verdict"].value_counts()
n_all = len(audit)
n_adv, n_no, n_ind = (int(counts.get(v, 0)) for v in ("morphology_advantage", "no_advantage", "indeterminate"))
n_powered = n_adv + n_no
config = f"{audit['feature_set'].iloc[0]} features, {audit['agg_method'].iloc[0]} aggregation"

st.markdown(
    f"""
<div class="hero">
<h1>Does a Cell Painting model see more than a cell count?</h1>
<p class="lede">Dying cells are easy to count, and a model that only counts them can look impressive. We tested {n_all} toxicity
endpoints from primary human liver cells to find where the full morphology profile adds something a cell count cannot.</p>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    f"<p class='finding'>Most endpoints cannot support a claim. {n_ind} have too few positive compounds to judge, "
    f"{n_no} show no certified gain over cell counts, and only {n_adv} show a certified morphology advantage.</p>",
    unsafe_allow_html=True,
)

ids = audit["endpoint_id"].tolist()
st.session_state.setdefault("ep_sel", "MT" if "MT" in ids else ids[0])
fig_map, plate_ids = plate_map(audit, st.session_state["ep_sel"])
event = st.plotly_chart(fig_map, width="stretch", on_select="rerun", selection_mode="points", key="platemap",
                        config={"displayModeBar": False})
points = (event.selection.points if event and getattr(event, "selection", None) else []) or []
if points:
    cd = points[0].get("customdata")
    clicked = cd[0] if isinstance(cd, (list, tuple)) else None
    if clicked in ids and clicked != st.session_state.get("_last_click"):
        st.session_state["_last_click"] = clicked
        st.session_state["ep_sel"] = clicked
        st.rerun()

st.markdown(
    f"""
<div class="legend">
<span class="item"><span class="dot" style="background:{MAGENTA}"></span><span class="num">{n_adv}</span>morphology advantage</span>
<span class="item"><span class="dot" style="background:{GREY}"></span><span class="num">{n_no}</span>no advantage over cell count</span>
<span class="item"><span class="dot ring"></span><span class="num">{n_ind}</span>indeterminate</span>
<span class="item"><span class="dot" style="background:{BLUE}"></span><span class="num">{int(counts.get('positive_control', 0))}</span>positive control</span>
</div>
""",
    unsafe_allow_html=True,
)
st.caption(f"Each dot is one endpoint (one assay readout). Click a dot to open it below. Configuration: {config}. "
           "Data: Ewald et al. 2026, 1,085 compounds.")

with st.expander("New to Cell Painting?"):
    st.markdown(
        "Cell Painting stains cells with six fluorescent dyes, photographs them in five channels, and measures thousands of "
        "shape, texture and brightness features per cell. Each compound is tested at eight concentrations, and the features "
        "become that compound's morphology profile. The risk we audit: when a compound kills cells, simply *fewer cells* "
        "predicts many toxicity labels, so a morphology model can look good without reading any morphology."
    )

tab_ep, tab_cmp, tab_tbl, tab_enr, tab_img, tab_about = st.tabs(
    ["Endpoints", "Compounds", "All results", "Where morphology wins", "Example images", "Method"])

# ------------------------------------------------------------------ endpoint explorer
with tab_ep:
    ep = st.selectbox("Endpoint (type to search)", ids, key="ep_sel",
                      format_func=lambda e: f"{e}  ({VERDICT_SHORT[audit.loc[audit['endpoint_id'] == e, 'verdict'].iloc[0]]})")
    r = audit[audit["endpoint_id"] == ep].iloc[0]
    st.markdown(verdict_marker(r["verdict"]), unsafe_allow_html=True)
    st.markdown(f"<p class='answer'>{answer_sentence(r, n_powered)}</p>", unsafe_allow_html=True)
    st.markdown(
        f"<p class='facts'><b>{int(r['n_compounds'])}</b> compounds tested, <b>{int(r['n_active'])}</b> active and "
        f"<b>{int(r['n_inactive'])}</b> non-hits. Category <b>{r.get('category')}</b>, target family "
        f"<b>{r.get('assay_target_family')}</b>, assay design <b>{r.get('assay_design_type')}</b>.</p>", unsafe_allow_html=True)
    desc = r.get("endpoint_description")
    if isinstance(desc, str):
        st.caption(desc[:400])

    left, right = st.columns([1, 1], gap="large")
    with left:
        if pd.notna(r.get("delta_AUROC")):
            names = ["Full morphology profile", "Cell count, dose-response curve", "Cell count, single number"]
            vals = [r["full_AUROC"], r["strong_cc_AUROC"], r["scalar_cc_AUROC"]]
            fig = go.Figure(go.Bar(x=vals, y=names, orientation="h", marker_color=[MAGENTA, GREY, RING], text=[f"{v:.2f}" for v in vals],
                                   textposition="outside", cliponaxis=False,
                                   hovertemplate="%{y}<br>AUROC %{x:.3f}<extra></extra>"))
            style(fig, 250, title="Ranking accuracy by model", showlegend=False)
            fig.update_yaxes(title=None, autorange="reversed")
            fig.update_xaxes(title="AUROC (0.5 = chance)", range=[0.4, 1.05])
            st.plotly_chart(fig, width="stretch", theme=None, config={"displayModeBar": False})
            with st.expander("Statistics for this endpoint"):
                stats = pd.DataFrame({
                    "measure": ["AUROC difference (full minus cell-count curve)", "95% bootstrap interval", "z-score",
                                "Calibrated p-value (used for the verdict)", "Raw bootstrap p-value", "BH-FDR q-value",
                                "Difference against the paper's single-number baseline"],
                    "value": [f"{r['delta_AUROC']:+.3f}", f"{r['delta_ci_lo']:+.3f} to {r['delta_ci_hi']:+.3f}", f"{r['delta_z']:.2f}",
                              f"{r['calibrated_p']:.4f}", f"{r['bootstrap_p']:.4f}",
                              "n/a" if pd.isna(r["fdr_q"]) else f"{r['fdr_q']:.4f}",
                              f"{r['delta_AUROC_scalar']:+.3f} ({VERDICT_SHORT.get(r['verdict_scalar'], r['verdict_scalar'])})"],
                })
                st.dataframe(stats, hide_index=True, width="stretch")
                st.caption("The raw bootstrap p-value is too optimistic. When we shuffled labels it flagged about 10% of endpoints at a "
                           "nominal 5% (18% for CP-CNN features), so verdicts use p-values calibrated against that shuffled-label null.")
        else:
            st.info("There are too few compounds in one class to run cross-validation, so this endpoint stays indeterminate.")
    with right:
        o = load_oof()
        o = o[o["endpoint_id"] == ep]
        if len(o):
            st.plotly_chart(reliability_figure(o["y"].to_numpy(), o["p_full"].to_numpy()), width="stretch", theme=None,
                            config={"displayModeBar": False})
            st.caption(f"Brier score {r['calibration_brier']:.3f}, expected calibration error {r['ece']:.3f}, calibration slope "
                       f"{r['calib_slope']:.2f} (below 1 means over-confident). This chart shows raw model scores; the Compounds tab "
                       "uses recalibrated probabilities.")

# ------------------------------------------------------------------ compound explorer
with tab_cmp:
    comps = load_compounds().dropna(subset=["name"]).sort_values("name")
    options = {f"{n}  ({i})": i for n, i in zip(comps["name"], comps["OASIS_ID"])}
    pick = st.selectbox("Compound (type to search)", list(options), key="cmp_sel")
    oid = options[pick]
    crow = comps[comps["OASIS_ID"] == oid].iloc[0]
    st.markdown(f"### {crow['name']}")
    st.caption(f"CASRN {crow['CASRN']}, {crow['DTXSID']}")
    chips = "".join(
        f"<span class='chip{' on' if crow[k] == 'Yes' else ''}'>{lab}: {'hit' if crow[k] == 'Yes' else 'no hit'}</span>"
        for k, lab in [("Cell_count_hit", "Cell count"), ("MT_hit", "Metabolic activity (MT)"), ("LDH_hit", "LDH release"),
                       ("Cell_Painting_hit", "Cell Painting")])
    st.markdown(f"<div class='chips'>{chips}</div>", unsafe_allow_html=True)

    dr = load_dose_response()
    d = dr[dr["OASIS_ID"] == oid].sort_values("concentration_uM")
    if len(d):
        fig = go.Figure()
        for y, name, color in [(d["cell_count_ratio"], "Cell count (relative to DMSO)", BLUE), (d["mt"], "Metabolic activity (MT)", MAGENTA),
                               (1 - d["ldh"].clip(upper=1), "1 minus LDH release", GREY)]:
            fig.add_trace(go.Scatter(x=d["concentration_uM"], y=y, name=name, mode="lines+markers", line=dict(color=color, width=2.5),
                                     marker=dict(size=8, color=color)))
        style(fig, 340, title="Readouts across 8 concentrations")
        fig.update_xaxes(type="log", title="concentration (µM)")
        st.plotly_chart(fig, width="stretch", theme=None, config={"displayModeBar": False})

    oof = load_oof()
    pc = oof[oof["OASIS_ID"] == oid].merge(
        audit[["endpoint_id", "category", "assay_target_family", "verdict", "delta_AUROC", "fdr_q", "full_AUROC", "ece"]], on="endpoint_id")
    st.markdown("### What the models predicted for this compound")
    st.caption("Each probability comes from a model that never saw this compound (cross-validation grouped by compound), then recalibrated "
               "per endpoint because raw scores are over-confident. The verdict describes the endpoint, not this compound: "
               "whether morphology reliably beats cell count for it.")
    vf = st.pills("Show endpoints with verdict", list(VERDICT_LABEL), selection_mode="multi",
                  default=["morphology_advantage", "no_advantage"], format_func=lambda v: VERDICT_LABEL[v], key="cmp_verdicts")
    show = pc[pc["verdict"].isin(vf or [])].copy()
    if show.empty:
        st.info("Pick at least one verdict above to list endpoints.")
    else:
        show["measured"] = show["y"].map({1: "active", 0: "non-hit"})
        mcol, bcol = ("p_full_cal", "p_strong_cc_cal") if "p_full_cal" in show else ("p_full", "p_strong_cc")
        show = show.rename(columns={mcol: "P(active), morphology", bcol: "P(active), cell count only"})
        cols = ["endpoint_id", "category", "assay_target_family", "measured", "P(active), morphology", "P(active), cell count only",
                "verdict", "delta_AUROC", "fdr_q", "ece"]
        st.dataframe(show[cols].sort_values("fdr_q"), width="stretch", hide_index=True,
                     column_config={"P(active), morphology": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f"),
                                    "P(active), cell count only": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f")})

# ------------------------------------------------------------------ all results
with tab_tbl:
    st.markdown("### Morphology against cell count, endpoint by endpoint")
    st.caption("Dots above the dashed line are endpoints where the full morphology model scored higher than the cell-count-only model. "
               "Colour shows whether that gap is certified after correction.")
    scored = audit[audit["delta_AUROC"].notna()]
    fig = px.scatter(scored, x="strong_cc_AUROC", y="full_AUROC", color="verdict", color_discrete_map=VERDICT_COLORS,
                     hover_name="endpoint_id", hover_data=["category", "assay_target_family", "n_active", "delta_AUROC", "fdr_q"])
    fig.update_traces(marker=dict(size=9, line=dict(width=0)))
    fig.add_trace(go.Scatter(x=[0.3, 1], y=[0.3, 1], mode="lines", line=dict(dash="dash", color=GREY), showlegend=False))
    style(fig, 520, legend_title_text="")
    fig.update_xaxes(title="AUROC, cell-count-only model")
    fig.update_yaxes(title="AUROC, full morphology model")
    fig.for_each_trace(lambda t: t.update(name=VERDICT_LABEL.get(t.name, t.name)))
    st.plotly_chart(fig, width="stretch", theme=None, config={"displayModeBar": False})

    cats = sorted(audit["category"].dropna().unique())
    cat = st.pills("Category", cats, selection_mode="multi", default=cats, key="tbl_cat")
    ver = st.pills("Verdict", list(VERDICT_LABEL), selection_mode="multi", default=list(VERDICT_LABEL),
                   format_func=lambda v: VERDICT_LABEL[v], key="tbl_ver")
    t = audit[audit["category"].isin(cat or []) & audit["verdict"].isin(ver or [])]
    show_cols = ["endpoint_id", "category", "assay_target_family", "n_active", "n_inactive", "scalar_cc_AUROC", "strong_cc_AUROC", "full_AUROC",
                 "delta_AUROC", "delta_z", "bootstrap_p", "calibrated_p", "fdr_q", "verdict", "verdict_uncalibrated", "verdict_scalar",
                 "calibration_brier", "ece"]
    st.caption(f"{len(t)} of {n_all} endpoints shown.")
    st.dataframe(t[show_cols].sort_values("fdr_q"), width="stretch", hide_index=True)
    st.download_button("Download the audit table (CSV)", audit.to_csv(index=False), "audit_table.csv", "text/csv")

# ------------------------------------------------------------------ enrichment
with tab_enr:
    enr = load_enrichment()
    st.markdown("### Is the advantage concentrated in a type of assay?")
    st.write("For each assay or target family we ask whether it is over-represented among the endpoints with a certified morphology "
             "advantage, compared with powered endpoints that have none (one-sided Fisher exact test, BH-adjusted within each grouping).")
    st.warning("Read these tables as descriptive. Cytotoxicity endpoints have about four times more actives than cell-based ones "
               "(median 97.5 against 25), so they are easier to certify. With at least 50 actives the cytotoxicity enrichment disappears "
               "for CellProfiler and CP-CNN features and stays undecided for DINOv2.")
    if enr is None:
        st.info("Run `python scripts/04_run_enrichment.py` to create results/enrichment.csv.")
    else:
        baselines = ["vs_strong_cc", "vs_scalar_cc"]
        base_label = {"vs_strong_cc": "Cell-count curve (strong)", "vs_scalar_cc": "Single cell-count number (paper)"}
        base = st.segmented_control("Certified advantage measured against", baselines, default="vs_strong_cc", key="enr_base",
                                    format_func=lambda b: base_label[b]) or "vs_strong_cc"
        groupings = sorted(enr["by"].unique())
        by = st.selectbox("Group endpoints by", groupings,
                          index=groupings.index("assay_target_family") if "assay_target_family" in groupings else 0)
        e = enr[(enr["baseline"] == base) & (enr["by"] == by)].sort_values("p")
        e = e.drop(columns=["baseline", "by"]).rename(columns={
            "group": "Group", "n_in_group": "Endpoints in group", "n_adv_in_group": "Advantage in group", "n_adv_out": "Advantage elsewhere",
            "n_out": "Endpoints elsewhere", "frac_adv_in": "Share in group", "frac_adv_out": "Share elsewhere", "odds_ratio": "Odds ratio",
            "p": "p-value", "q": "q-value"})
        st.dataframe(e, width="stretch", hide_index=True, column_config={
            "Share in group": st.column_config.NumberColumn(format="%.2f"), "Share elsewhere": st.column_config.NumberColumn(format="%.2f"),
            "Odds ratio": st.column_config.NumberColumn(format="%.2f"), "p-value": st.column_config.NumberColumn(format="%.4f"),
            "q-value": st.column_config.NumberColumn(format="%.4f")})
        if (e["q-value"] < 0.05).any():
            st.success(f"{int((e['q-value'] < 0.05).sum())} group(s) are significantly enriched at q < 0.05.")
        else:
            st.info("No group is significantly enriched at q < 0.05, so this grouping does not explain where morphology wins.")

    sens = RES / "sensitivity_summary.csv"
    if sens.exists():
        st.markdown("### Does the answer survive other choices?")
        st.caption("Endpoints certified as a morphology advantage (BH q < 0.05, calibrated p-value) for each image representation and "
                   "aggregation rule, against the strong and the paper's single-number cell-count baseline, next to what a plain "
                   "bootstrap would have reported.")
        sm = pd.read_csv(sens)[["config", "powered", "adv_vs_strong", "adv_vs_scalar", "adv_vs_strong_raw_bootstrap",
                                "adv_vs_scalar_raw_bootstrap", "null_sd_used"]].rename(columns={
            "config": "Configuration", "powered": "Powered endpoints", "adv_vs_strong": "Certified vs curve baseline",
            "adv_vs_scalar": "Certified vs single-number baseline", "adv_vs_strong_raw_bootstrap": "Plain bootstrap vs curve",
            "adv_vs_scalar_raw_bootstrap": "Plain bootstrap vs single number", "null_sd_used": "Null width used"})
        st.dataframe(sm, width="stretch", hide_index=True)

# ------------------------------------------------------------------ example images
with tab_img:
    manifest_path = ASSETS / "images" / "manifest.csv"
    st.markdown("### What the cells look like")
    st.write("A treated well at the highest tested concentration, next to a DMSO control from the same plate. Channels: DNA blue, "
             "ER green, AGP red, mitochondria magenta, RNA cyan.")
    st.caption("Images come from the public Cell Painting Gallery (cpg0037-oasis, CC0), fetched one at a time by "
               "`scripts/14_fetch_example_images.py`.")
    if not manifest_path.exists():
        st.info("Run `python scripts/14_fetch_example_images.py` to fetch the example images.")
    else:
        man = pd.read_csv(manifest_path)
        names = load_compounds().set_index("OASIS_ID")
        feat = man.drop_duplicates("OASIS_ID")
        group_label = {"hit_in_all_assays": "active in every assay",
                       "cell_painting_only": "Cell Painting active only (no cell loss, MT or LDH change)",
                       "inactive": "inactive in all readouts"}
        opt = {f"{names.loc[o, 'name'] if o in names.index else o}: {group_label[g]}": o for o, g in zip(feat["OASIS_ID"], feat["group"])}
        pick_img = st.selectbox("Featured compound", list(opt), key="img_sel")
        o = opt[pick_img]
        c1, c2 = st.columns(2, gap="large")
        for col, kind, title in ((c1, "treated", "Treated"), (c2, "dmso", "DMSO control")):
            rr = man[(man["OASIS_ID"] == o) & (man["kind"] == kind)]
            if len(rr) and (ASSETS / "images" / rr.iloc[0]["file"]).exists():
                conc = f", {rr.iloc[0]['concentration_uM']:.3g} µM" if kind == "treated" else ""
                col.image(str(ASSETS / "images" / rr.iloc[0]["file"]), caption=f"{title}{conc}. Plate {rr.iloc[0]['plate']}, well {rr.iloc[0]['well']}")

# ------------------------------------------------------------------ method
with tab_about:
    st.markdown(
        """
### The question, asked of every endpoint
1. Does the full morphology profile beat a cell-count-only model, after Benjamini–Hochberg correction for testing many endpoints at once?
2. Is there enough data to say? An endpoint needs at least 15 active compounds and 15 non-hits. Otherwise it is *indeterminate*, and we publish no score for it.
3. Where morphology wins, is the advantage concentrated in a coherent assay or target family?

### The models
All three use XGBoost (150 trees, learning rate 0.05, class-reweighted, as in the source paper).
- **Full morphology:** normalised Cell Painting features.
- **Cell count, single number:** the paper's baseline, the mean cell count.
- **Cell count, dose-response curve:** a stronger baseline that sees cell count at each of 8 concentrations, plus its minimum, area under the curve and the concentration where it falls. It still carries only cell-count information.

### How we test the difference
Cross-validation is grouped by compound (5 folds, repeated 3 times), so no compound appears in both training and test data. We compare pooled out-of-fold AUROC with a compound bootstrap. When we shuffled the labels, the raw bootstrap p-values were too optimistic, so we calibrate them against that shuffled-label null before applying Benjamini–Hochberg over the powered endpoints. Brier score, calibration error and calibration slope are reported per endpoint.

### Reading the labels
A ToxCast 0 means *not a (filtered) hit*: a true non-hit, a hit removed by the cytotoxicity filter, or a tie. It never means "tested negative". Untested pairs are missing, not 0. For the cytotoxicity columns, 1 means cytotoxic in that cell type or tissue. The cytotoxicity filter applies to only about 35% of cell-based records.

### What we do not claim
This is a 2D plate assay, not an organ-on-chip, at a single 44-hour time point, and donor variability is not modelled. The count of certified endpoints depends on how wide the shuffled-label null is, so treat it as a range. The audit itself is dataset-agnostic and can be re-run on any phenotypic-profiling toxicology dataset that has a cell-count-like baseline.
"""
    )

st.markdown(
    f"<div class='footer'>Code, report and reproduction steps: <a href='{REPO_URL}'>{REPO_URL.removeprefix('https://')}</a>. "
    "Source data: Ewald et al., Cell Systems 2026 (Zenodo, CC-BY 4.0).</div>",
    unsafe_allow_html=True,
)
