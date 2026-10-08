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
    "no_advantage": "No detectable advantage over cell count",
    "indeterminate": "Indeterminate, too few positives",
}
VERDICT_SHORT = {"morphology_advantage": "advantage", "positive_control": "control", "no_advantage": "no detectable advantage",
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

.stats { margin: 0.2rem 0 0.6rem; }
.stat { display: flex; justify-content: space-between; gap: 1rem; padding: 0.45rem 0; border-bottom: 1px solid %RULE%; font-size: 0.95rem; }
.stat span { color: %SLATE%; max-width: 62%; }
.stat b { font-weight: 600; text-align: right; font-variant-numeric: tabular-nums; }
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


@st.cache_data
def load_optional(name: str):
    p = RES / name
    return pd.read_csv(p) if p.exists() else None


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
        return (f"No detectable advantage over cell counts here. The full profile scores AUROC {r['full_AUROC']:.2f} and the "
                f"cell-count-only model {r['strong_cc_AUROC']:.2f}; the difference is not certified after correcting for testing "
                f"{n_powered} endpoints{q}. This is absence of evidence for an advantage, not proof the two are equivalent.")
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
<span class="item"><span class="dot" style="background:{GREY}"></span><span class="num">{n_no}</span>no detectable advantage over cell count</span>
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

tab_ep, tab_cmp, tab_tbl, tab_enr, tab_img, tab_cells, tab_chip, tab_about = st.tabs(
    ["Endpoints", "Compounds", "All results", "Where morphology wins", "Example images", "Cell lines", "Organ-on-chip", "Method"])

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
            colors = [MAGENTA, GREY, RING]
            nested = load_optional("nested_table.csv")
            nr = nested[nested["endpoint_id"] == ep] if nested is not None else None
            if nr is not None and len(nr):
                nr = nr.iloc[0]
                names.insert(1, "Morphology plus cell count")
                vals.insert(1, nr["nested_AUROC"])
                colors.insert(1, "#D98CC4")
            fig = go.Figure(go.Bar(x=vals, y=names, orientation="h", marker_color=colors, text=[f"{v:.2f}" for v in vals],
                                   textposition="outside", cliponaxis=False,
                                   hovertemplate="%{y}<br>AUROC %{x:.3f}<extra></extra>"))
            style(fig, 250 if len(names) == 3 else 300, title="Ranking accuracy by model", showlegend=False)
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
                mde = load_optional("detectable_effect.csv")
                mr = mde[mde["endpoint_id"] == ep] if mde is not None else None
                extra_m, extra_v = [], []
                if mr is not None and len(mr) and pd.notna(mr.iloc[0]["min_detectable_effect"]):
                    extra_m.append("Smallest advantage this data could detect (80% power)")
                    extra_v.append(f"{mr.iloc[0]['min_detectable_effect']:.2f} AUROC")
                if nr is not None and len(nr):
                    extra_m += ["Morphology plus cell count against cell count alone", "Same comparison: calibrated q-value and verdict"]
                    extra_v += [f"{nr['delta_nested']:+.3f} (95% interval {nr['delta_nested_ci_lo']:+.3f} to {nr['delta_nested_ci_hi']:+.3f})",
                                f"{nr['fdr_q_nested']:.4f}, {str(nr['verdict_nested']).replace('_', ' ')}"]
                    if pd.notna(nr.get("logreg_AUROC")):
                        extra_m.append("Regularised logistic regression on morphology (sanity check), AUROC")
                        extra_v.append(f"{nr['logreg_AUROC']:.3f}")
                if extra_m:
                    stats = pd.concat([stats, pd.DataFrame({"measure": extra_m, "value": extra_v})], ignore_index=True)
                rows_html = "".join(f"<div class='stat'><span>{m}</span><b>{v}</b></div>" for m, v in zip(stats["measure"], stats["value"]))
                st.markdown(f"<div class='stats'>{rows_html}</div>", unsafe_allow_html=True)
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
             "advantage, compared with powered endpoints that have none. The p-value is a one-sided Fisher exact test. Endpoints from one assay are correlated, "
             "so the assay-level p-value repeats the test with whole assays as the unit (a permutation over assay clusters). Both are BH-adjusted within each grouping.")
    st.warning("Read these tables as descriptive. Cytotoxicity endpoints have about four times more actives than cell-based ones "
               "(median 97.5 against 25), so they are easier to certify. With at least 50 actives the cytotoxicity enrichment disappears "
               "for CellProfiler and CP-CNN features and stays undecided for DINOv2. The assay-level test is the more conservative of the two.")
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
            "p": "p-value", "q": "q-value", "p_cluster": "Assay-level p-value", "q_cluster": "Assay-level q-value"})
        st.dataframe(e, width="stretch", hide_index=True, column_config={
            "Share in group": st.column_config.NumberColumn(format="%.2f"), "Share elsewhere": st.column_config.NumberColumn(format="%.2f"),
            "Odds ratio": st.column_config.NumberColumn(format="%.2f"), "p-value": st.column_config.NumberColumn(format="%.4f"),
            "q-value": st.column_config.NumberColumn(format="%.4f"),
            "Assay-level p-value": st.column_config.NumberColumn(format="%.4f"),
            "Assay-level q-value": st.column_config.NumberColumn(format="%.4f")})
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

# ------------------------------------------------------------------ cell lines
with tab_cells:
    st.markdown("### Cell lines: what they look like and where the endpoints come from")
    st.write("The audit runs on primary human hepatocytes, but Cell Painting is used on many cell systems, and the toxicity assays behind the "
             "endpoints use many more. Pick cells and a condition to compare them, then explore which cell systems the endpoints come from.")
    cl_dir = ASSETS / "cell_lines"
    cl_manifest = cl_dir / "manifest.csv"
    SYSTEM_INFO = {
        "Primary hepatocytes": "Freshly plated human liver cells from donors, polygonal and often with two nuclei. The closest of the three to liver physiology, and the cells of the main dataset (OASIS).",
        "HepG2": "A human liver cancer line that grows in tight clusters. Widely used for toxicity screens, and imaged at four sites in the EU-OPENSCREEN bioactives collection.",
        "U2OS": "A human bone cancer line with flat, spread-out cells. The standard reference line of large Cell Painting resources, shown here for contrast.",
    }
    st.markdown("#### See the cells")
    c1, c2 = st.columns([1, 1], gap="large")
    systems = c1.pills("Cell systems", list(SYSTEM_INFO), selection_mode="multi", default=list(SYSTEM_INFO), key="cl_systems")
    cond = c2.pills("Condition", ["DMSO control", "Nocodazole", "Bortezomib"], selection_mode="single", default="DMSO control", key="cl_cond")
    cond = cond or "DMSO control"
    st.caption({"DMSO control": "Untreated cells (solvent only): the reference appearance of each cell system.",
                "Nocodazole": "A microtubule disruptor: cells round up and the cytoskeleton collapses.",
                "Bortezomib": "A proteasome inhibitor that is cytotoxic: the number of healthy cells drops."}[cond])
    if not systems:
        st.info("Pick at least one cell system above.")
    elif not cl_manifest.exists():
        st.info("Run `python scripts/26_fetch_cell_line_images.py` to fetch the example images.")
    else:
        cm = pd.read_csv(cl_manifest)
        cols = st.columns(len(systems), gap="medium")
        for col, system in zip(cols, systems):
            col.markdown(f"**{system}**")
            path, cap = None, None
            om = ASSETS / "images" / "manifest.csv"
            if system == "Primary hepatocytes" and cond == "DMSO control" and om.exists():
                r0 = pd.read_csv(om).query("kind == 'dmso'").iloc[0]
                path, cap = ASSETS / "images" / r0["file"], f"DMSO control, plate {r0['plate']}, well {r0['well']}"
            else:
                r1 = cm[(cm["cell_line"] == system) & (cm["compound"] == cond)]
                if len(r1):
                    path = cl_dir / r1.iloc[0]["file"]
                    conc = "" if cond == "DMSO control" else f", {r1.iloc[0]['concentration_uM']:g} µM"
                    cap = f"{cond}{conc}, plate {r1.iloc[0]['plate']}, well {r1.iloc[0]['well']}"
            if path is not None and path.exists():
                col.image(str(path), caption=cap)
            else:
                col.caption("No image for this combination.")
            col.caption(SYSTEM_INFO[system])
        st.caption("HepG2 and U2OS images: Cell Painting Gallery, cpg0036-EU-OS-bioactives, FMP site (CC0). Channels: DNA blue, ER green, AGP red, mitochondria magenta.")

    st.markdown("#### Where the endpoints come from")
    ce = audit.dropna(subset=["cell_short_name"]).copy()
    ce["cell system"] = ce["cell_short_name"].str.replace("umbilical vein endothelium and peripheral blood mononuclear cells", "HUVEC + PBMC", regex=False) \
                                           .str.replace("umbilical vein endothelium", "HUVEC", regex=False).str.replace("coronary artery smooth muscle cells", "coronary artery SMC", regex=False) \
                                           .str.replace("B and peripheral blood mononuclear cells", "B cells + PBMC", regex=False)
    top = ce["cell system"].value_counts()
    keep_n = st.slider("Cell systems to show (largest first)", 5, min(25, len(top)), 12)
    shown = top.index[:keep_n]
    stack = ce[ce["cell system"].isin(shown)].groupby(["cell system", "verdict"]).size().reset_index(name="endpoints")
    fig = px.bar(stack, y="cell system", x="endpoints", color="verdict", orientation="h", color_discrete_map=VERDICT_COLORS,
                 category_orders={"cell system": list(shown), "verdict": VERDICT_ORDER})
    fig.for_each_trace(lambda t: t.update(name=VERDICT_LABEL.get(t.name, t.name)))
    style(fig, 120 + 28 * keep_n, title="Endpoints by assay cell system", legend_title_text="")
    fig.update_yaxes(title=None)
    fig.update_xaxes(title="endpoints")
    st.plotly_chart(fig, width="stretch", theme=None, config={"displayModeBar": False})
    pick_cl = st.selectbox("Open one cell system", list(shown), key="cl_pick")
    sub = ce[ce["cell system"] == pick_cl]
    n_adv = int((sub["verdict"] == "morphology_advantage").sum())
    n_pow = int(sub["powered"].sum())
    m1, m2, m3 = st.columns(3)
    m1.metric("Endpoints", len(sub))
    m2.metric("Powered", n_pow)
    m3.metric("Certified morphology advantage", n_adv)
    st.dataframe(sub[["endpoint_id", "assay_target_family", "n_active", "n_inactive", "full_AUROC", "strong_cc_AUROC", "delta_AUROC", "fdr_q", "verdict"]]
                 .sort_values("fdr_q"), hide_index=True, width="stretch")
    st.caption("Cell system here means the cells used by the toxicity assay behind each endpoint, which can differ from the cells imaged for Cell Painting.")

    eu_sum = RES / "eu_os" / "summary.json"
    if eu_sum.exists():
        import json
        es = json.loads(eu_sum.read_text())
        st.markdown("#### Same labels, different cells: HepG2 profiles against primary hepatocytes")
        st.write(f"On the {es['n_compounds']} compounds present in both collections and {es['n_endpoints']} endpoints with enough data, the audit was run once "
                 "with HepG2 profiles (four imaging sites, one concentration) and once with primary-hepatocyte profiles, against the same labels and the same "
                 "single-number cell-count baseline.")
        k1, k2, k3 = st.columns(3)
        k1.metric("Certified with HepG2 profiles", es["certified_hepg2"])
        k2.metric("Certified with primary hepatocytes", es["certified_oasis"])
        k3.metric("Certified in both", es["certified_both"])
        cmp_path = RES / "eu_os" / "comparison.csv"
        if cmp_path.exists():
            cp_ = pd.read_csv(cmp_path)
            fig = px.scatter(cp_, x="full_AUROC_oasis", y="full_AUROC_hepg2", hover_name="endpoint_id", color="verdict_hepg2", color_discrete_map=VERDICT_COLORS)
            fig.add_trace(go.Scatter(x=[0.3, 1], y=[0.3, 1], mode="lines", line=dict(dash="dash", color=GREY), showlegend=False))
            fig.for_each_trace(lambda t: t.update(name=VERDICT_LABEL.get(t.name, t.name)))
            style(fig, 400, title="AUROC of the full profile, endpoint by endpoint", legend_title_text="HepG2 verdict")
            fig.update_xaxes(title="primary hepatocytes")
            fig.update_yaxes(title="HepG2")
            st.plotly_chart(fig, width="stretch", theme=None, config={"displayModeBar": False})
        st.caption("Exploratory: HepG2 has a single concentration, so only the single-number cell-count baseline exists, and the labels come from the primary-hepatocyte study.")

# ------------------------------------------------------------------ organ-on-chip
with tab_chip:
    st.markdown("### What the audit says at organ-on-chip scale")
    st.write("A liver chip study tests tens of compounds, not about a thousand, and runs each compound on several chips. This tab shows what the "
             "audit can conclude at that scale and what a chip team would need to change. Every number below comes from simulation with known "
             "ground truth or from the plate data above. No chip data was used.")
    fit_path = RES / "chip_scale" / "planner_fit.json"
    if fit_path.exists():
        import json
        fit = json.loads(fit_path.read_text())
        st.markdown("#### Plan a study")
        c1, c2 = st.columns(2, gap="large")
        n_comp = c1.slider("Compounds tested", 20, 400, 60, step=5)
        share = c2.slider("Share of compounds that are active", 0.10, 0.50, 0.30, step=0.05, format="%.2f")
        n1, n0 = n_comp * share, n_comp * (1 - share)
        n_eff = 4 / (1 / n1 + 1 / n0)
        mid = fit["a"] + fit["b"] * np.log(n_eff)
        typ, lo, hi = np.exp(mid), np.exp(mid - fit["resid_sd"]), np.exp(mid + fit["resid_sd"])
        st.metric("Smallest advantage over cell count this design can detect", f"{typ:.2f} AUROC", help="Typical value; the band is one residual standard deviation of the real endpoints around the fitted curve.")
        st.caption(f"Typical range {lo:.2f} to {hi:.2f}. {int(round(n1))} active and {int(round(n0))} non-hit compounds. "
                   + ("Advantages smaller than this would not be certified; a value above 0.50 means no advantage could be detected at all." if typ > 0.3 else
                      "Advantages smaller than this would not be certified."))
        st.caption(f"Curve fitted on the {fit['n_endpoints']} real endpoints of the plate audit (R² {fit['r2']:.2f}): detectable effect scales roughly with the inverse square root of the effective sample size.")
    sim = RES / "chip_scale" / "summary.csv"
    if sim.exists():
        st.markdown("#### Simulation across study sizes")
        st.image(str(RES / "chip_scale" / "chip_scale.png"))
        sm = pd.read_csv(sim).rename(columns={
            "n_compounds": "Compounds", "indeterminate_strict": "Cannot be judged (15/15 rule)", "indeterminate_chip": "Cannot be judged (5/5 rule)",
            "credited_raw_on_null": "False credit, raw p", "credited_calibrated_on_null": "False credit, calibrated p",
            "power_calibrated": "Power, calibrated p", "median_min_detectable_effect": "Smallest detectable advantage", "null_sd": "Null width"})
        st.dataframe(sm[["Compounds", "Cannot be judged (15/15 rule)", "Cannot be judged (5/5 rule)", "False credit, raw p", "False credit, calibrated p",
                         "Power, calibrated p", "Smallest detectable advantage", "Null width"]], hide_index=True, width="stretch",
                     column_config={c: st.column_config.NumberColumn(format="%.2f") for c in sm.columns if c not in ("Compounds", "n_endpoints")})
    leak_img = RES / "chip_scale" / "replicate_leakage.png"
    if leak_img.exists():
        st.markdown("#### Group by compound when compounds are replicated over chips")
        l1, l2 = st.columns([1, 1], gap="large")
        l1.image(str(leak_img))
        l2.write("Forty compounds with random labels, each run on three chips. If cross-validation treats every chip as independent, a model recognises "
                 "a compound it has already seen on another chip and the AUROC rises above the true 0.5. Grouping folds by compound removes that, "
                 "so the audit keeps compound as the grouping unit and a chip design would add donor or lot as further groups.")
    st.markdown("#### What carries over and what changes")
    k1, k2 = st.columns(2, gap="large")
    k1.markdown(
        """
**Carries over**
- The question: does the full profile beat a cell-count baseline?
- Paired bootstrap of the AUROC difference, with a calibrated null
- The power check, the *indeterminate* verdict and the detectable-effect report
- Label hygiene: untested stays missing, a zero is a non-hit
- Grouped cross-validation
"""
    )
    k2.markdown(
        """
**Changes for a chip**
- **Labels:** albumin, urea, CYP3A4 activity and clinical DILI categories replace the ToxCast and Tox21 battery
- **Baseline:** a 3D cell-count proxy such as nuclei per z-stack or tissue volume
- **Features:** chips are imaged and profiled again, with PDMS autofluorescence and z-stacks in mind
- **Grouping:** compound, chip and donor or lot
- **Sample size:** a pooled or multi-site design, reported as an estimate with an interval
"""
    )
    st.markdown("#### A realistic first pilot")
    st.write("Twenty to sixty compounds spread across the DILIrank categories, two or three functional endpoints chosen in advance, a defined 3D "
             "cell-count proxy, folds grouped by compound and donor, and a result reported as the AUROC difference with its calibrated interval and "
             "its detectable effect, in place of a table of verdicts.")

# ------------------------------------------------------------------ method
with tab_about:
    st.markdown(
        """
### The question, asked of every endpoint
1. Does the full morphology profile beat a cell-count-only model, after Benjamini–Hochberg correction for testing many endpoints at once?
2. Is there enough data to say? An endpoint needs at least 15 active compounds and 15 non-hits. Otherwise it is *indeterminate*, and we publish no score for it.
3. Where morphology wins, is the advantage concentrated in a coherent assay or target family?

### What one row means
Labels are per compound, so each compound is one row. Wells are averaged per compound, and the default rule (`allpod`) keeps only wells at or above the compound's point of departure, so low concentrations where nothing happens do not dilute the profile.

### The models
All use XGBoost (150 trees, learning rate 0.05, class-reweighted, as in the source paper), with these fixed settings so no tuning ever touches the test folds.
- **Full morphology:** normalised Cell Painting features.
- **Cell count, single number:** the paper's baseline, the mean cell count.
- **Cell count, dose-response curve:** a stronger baseline that sees cell count at each of 8 concentrations, plus its minimum, area under the curve and the concentration where it falls. It still carries only cell-count information. No plate, well or batch features are used in any baseline.
- **Morphology plus cell count:** a nested model that sees both feature sets. Comparing it with the cell-count curve alone answers the sharper question: does morphology add information on top of cell count?
- **Regularised logistic regression:** a linear sanity check on the morphology features, with its regularisation strength chosen inside the training folds.

### How we test the difference
Cross-validation is grouped by compound (5 folds, repeated 3 times), so no compound appears in both training and test data. We compare pooled out-of-fold AUROC with a compound bootstrap. When we shuffled the labels, the raw bootstrap p-values were too optimistic, so we calibrate them against that shuffled-label null before applying Benjamini–Hochberg over the powered endpoints. Benjamini–Hochberg is valid under positive dependence, which fits endpoints that share compounds; the effective number of independent tests is smaller than the count, so the correction is conservative. Brier score, calibration error and calibration slope are reported per endpoint, and each endpoint also shows the smallest advantage its data could detect.

### Reading the labels
A ToxCast 0 means *not a (filtered) hit*: a true non-hit, a hit removed by the cytotoxicity filter, or a tie. It never means "tested negative". Untested pairs are missing, not 0. For the cytotoxicity columns, 1 means cytotoxic in that cell type or tissue. The cytotoxicity filter applies to only about 35% of cell-based records.

### How to read "no detectable advantage"
It means the data showed no advantage that survives correction. It is absence of evidence, not proof of equivalence, and it is only as informative as the smallest advantage the endpoint could detect, which is listed for every endpoint.

### What we do not claim
This is a 2D plate assay, not an organ-on-chip, at a single 44-hour time point, and donor variability is not modelled. The count of certified endpoints depends on how wide the shuffled-label null is, so treat it as a range. The audit itself is dataset-agnostic and can be re-run on any phenotypic-profiling toxicology dataset that has a cell-count-like baseline.
"""
    )

st.markdown(
    f"<div class='footer'>Code, report and reproduction steps: <a href='{REPO_URL}'>{REPO_URL.removeprefix('https://')}</a>. "
    "Source data: Ewald et al., Cell Systems 2026 (Zenodo, CC-BY 4.0).</div>",
    unsafe_allow_html=True,
)
