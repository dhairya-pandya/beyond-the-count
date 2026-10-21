#!/usr/bin/env python
"""Build the Team BioBytes finals deck (16:9, pointer style) and export it to PDF.

Each slide is written as its own HTML page, screenshotted with headless Chrome at 1920x1080 and combined with Pillow. One page per
slide, so no pagination surprises.

  python scripts/28_build_finals_deck.py    # -> docs/finals/beyond_the_count_finals.pdf (+ PNG previews in docs/finals/previews)

Tokens match the live demo: paper #F4F6F7, ink #17222E, magenta #B01E82 (mitochondria stain), blue #2358B8 (DNA stain), slate #5E6B78,
grey #8F9BA7, rule #D5DBE0. Newsreader for headlines, Hanken Grotesk for text. Cell images are in docs/finals/assets.
"""
import base64
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "finals"
ASSETS = OUT / "assets"
PREVIEWS = OUT / "previews"
MAG, BLUE, INK, SLATE, GREY, RING, RULE, PAPER = "#B01E82", "#2358B8", "#17222E", "#5E6B78", "#8F9BA7", "#AEB7C0", "#D5DBE0", "#F4F6F7"
N = 10
SANS = "Hanken Grotesk, sans-serif"
SERIF = "Newsreader, Georgia, serif"


def b64(name: str) -> str:
    return "data:image/png;base64," + base64.b64encode((ASSETS / name).read_bytes()).decode()


# ------------------------------------------------------------------ diagrams
def plate_map(unit: int = 30) -> str:
    a = pd.read_csv(ROOT / "results" / "audit_table.csv")
    parts = []
    for v in ["morphology_advantage", "positive_control", "no_advantage", "indeterminate"]:
        d = a[a["verdict"] == v]
        parts.append(d.sort_values("n_active" if v == "indeterminate" else "delta_AUROC", ascending=False, na_position="last"))
    o = pd.concat(parts).reset_index(drop=True)
    rows, cols, top = 15, 27, 62
    w, h = cols * unit, rows * unit + top
    el = []
    for i, v in enumerate(o["verdict"]):
        cx, cy = (i % cols) * unit + unit / 2, (i // cols) * unit + unit / 2 + top
        if v == "morphology_advantage":
            el.append(f'<circle cx="{cx}" cy="{cy}" r="10.5" fill="{MAG}"/>')
        elif v == "positive_control":
            el.append(f'<circle cx="{cx}" cy="{cy}" r="10.5" fill="{BLUE}"/>')
        elif v == "no_advantage":
            el.append(f'<circle cx="{cx}" cy="{cy}" r="9" fill="{GREY}"/>')
        else:
            el.append(f'<circle cx="{cx}" cy="{cy}" r="8" fill="none" stroke="{RING}" stroke-width="2"/>')
    x0, x1 = unit / 2, 14 * unit + unit / 2
    el.insert(0, f'<path d="M{x0 - 8} {top - 14} V{top - 26} H{x1 + 8} V{top - 14}" fill="none" stroke="{MAG}" stroke-width="2.5"/>'
                 f'<text x="{(x0 + x1) / 2}" y="{top - 36}" text-anchor="middle" font-size="24" font-weight="600" fill="{MAG}" font-family="{SANS}">15 certified</text>')
    return (f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="Plate map of 405 endpoints: 15 certified, 166 no detectable advantage, '
            f'223 indeterminate, 1 positive control">{"".join(el)}</svg>')


def diagram_wells() -> str:
    import random
    rnd = random.Random(7)
    el = []
    for r in range(3):
        for c in range(6):
            cx, cy = 70 + c * 100, 60 + r * 100
            el.append(f'<circle cx="{cx}" cy="{cy}" r="42" fill="#FFFFFF" stroke="{RING}" stroke-width="2.5"/>')
            n = rnd.randint(3, 9) if (r + c) % 4 else rnd.randint(10, 15)
            for _ in range(n):
                dx, dy = rnd.uniform(-26, 26), rnd.uniform(-26, 26)
                if dx * dx + dy * dy < 25 * 25:
                    el.append(f'<circle cx="{cx + dx:.1f}" cy="{cy + dy:.1f}" r="3.6" fill="{BLUE}"/>')
    return f'<svg viewBox="0 0 640 310" role="img" aria-label="A plate where only the number of cells per well is measured">{"".join(el)}</svg>'


def arrowhead() -> str:
    return (f'<defs><marker id="ah" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto" markerUnits="strokeWidth">'
            f'<path d="M1 1 L8.5 5 L1 9" fill="none" stroke="{SLATE}" stroke-width="1.3" stroke-linecap="round" stroke-linejoin="round"/></marker></defs>')


def arrow(x1, y1, x2, y2) -> str:
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{SLATE}" stroke-width="1.6" marker-end="url(#ah)"/>'


def rect(x, y, w, h, title, fill="#FFFFFF", stroke=RULE, color=INK, weight=500, size=25, sub=None) -> str:
    s = f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{fill}" stroke="{stroke}" stroke-width="2.5"/>'
    if sub:
        s += f'<text x="{x + 20}" y="{y + 32}" font-size="{size}" font-weight="{weight}" fill="{color}" font-family="{SANS}">{title}</text>'
        s += f'<text x="{x + 20}" y="{y + 58}" font-size="19" fill="{SLATE}" font-family="{SANS}">{sub}</text>'
    else:
        s += f'<text x="{x + 20}" y="{y + h / 2 + 9}" font-size="{size}" font-weight="{weight}" fill="{color}" font-family="{SANS}">{title}</text>'
    return s


def header(x, text) -> str:
    return f'<text x="{x}" y="24" font-size="21" font-weight="600" fill="{SLATE}" font-family="{SANS}" letter-spacing=".02em">{text}</text>'


def diagram_architecture() -> str:
    T = 60  # top of content
    el = [arrowhead()]
    el += [header(0, "Inputs"), header(290, "Prepare"), header(620, "Matched models, same folds"), header(1010, "Statistics"), header(1400, "Four verdicts")]
    inputs = ["Cell Painting profiles", "Cell counts", "Toxicity labels"]
    for i, t in enumerate(inputs):
        y = T + 35 + i * 100
        el.append(rect(0, y, 250, 70, t, size=23))
        el.append(arrow(252, y + 35, 288, T + 170))
    el.append(rect(290, T + 90, 240, 160, "", fill="#FFFFFF", stroke=INK))
    el.append(f'<text x="310" y="{T + 130}" font-size="28" font-weight="600" fill="{INK}" font-family="{SANS}">Prepare</text>')
    el.append(f'<text x="310" y="{T + 164}" font-size="19" fill="{SLATE}" font-family="{SANS}">normalise per plate</text>')
    el.append(f'<text x="310" y="{T + 190}" font-size="19" fill="{SLATE}" font-family="{SANS}">one row per compound</text>')
    el.append(f'<text x="310" y="{T + 216}" font-size="19" fill="{SLATE}" font-family="{SANS}">untested stays missing</text>')
    models = [("Full morphology", MAG, "#FFFFFF", MAG), ("Cell count, one number", RULE, "#FFFFFF", INK),
              ("Cell count, curve", GREY, "#FFFFFF", INK), ("Morphology plus count", MAG, "#F7E3F1", MAG)]
    for i, (t, stroke, fill, col) in enumerate(models):
        y = T + i * 90
        el.append(rect(620, y, 280, 70, t, fill=fill, stroke=stroke, color=col, weight=600 if stroke == MAG else 500, size=23))
        el.append(arrow(510, T + 170, 618, y + 35))
        el.append(arrow(900, y + 35, 1008, T + 170))
    el.append(f'<rect x="1010" y="{T}" width="290" height="340" rx="10" fill="{INK}"/>')
    steps = ["Paired bootstrap", "Calibrated p-value", "Power check 15/15", "BH-FDR across endpoints"]
    for i, t in enumerate(steps):
        el.append(f'<text x="1032" y="{T + 60 + i * 76}" font-size="22" fill="#FFFFFF" font-family="{SANS}" font-weight="500">{t}</text>')
        if i < 3:
            el.append(f'<line x1="1032" y1="{T + 80 + i * 76}" x2="1280" y2="{T + 80 + i * 76}" stroke="#3A4756" stroke-width="1.5"/>')
    el.append(arrow(1302, T + 170, 1398, T + 35))
    el.append(arrow(1302, T + 170, 1398, T + 125))
    el.append(arrow(1302, T + 170, 1398, T + 215))
    el.append(arrow(1302, T + 170, 1398, T + 305))
    verdicts = [("morphology advantage", MAG, "#FFFFFF"), ("no detectable advantage", GREY, "#FFFFFF"),
                ("indeterminate", RING, INK), ("positive control", BLUE, "#FFFFFF")]
    for i, (t, fill, col) in enumerate(verdicts):
        y = T + i * 90
        if t == "indeterminate":
            el.append(f'<rect x="1400" y="{y}" width="320" height="70" rx="35" fill="#FFFFFF" stroke="{RING}" stroke-width="3"/>')
            el.append(f'<text x="1424" y="{y + 44}" font-size="23" font-weight="600" fill="{SLATE}" font-family="{SANS}">{t}</text>')
        else:
            el.append(f'<rect x="1400" y="{y}" width="320" height="70" rx="35" fill="{fill}"/>')
            el.append(f'<text x="1424" y="{y + 44}" font-size="23" font-weight="600" fill="{col}" font-family="{SANS}">{t}</text>')
    return f'<svg viewBox="0 0 1720 430" role="img" aria-label="Architecture: profiles, counts and labels are prepared per compound, fed to four matched models on the same folds, compared through a calibrated statistics pipeline, and sorted into four verdicts">{"".join(el)}</svg>'


def diagram_roadmap() -> str:
    el = [arrowhead()]
    phases = [
        ("Next", "Close the evidence gaps", MAG,
         ["More cell types, donors, time points", "Organ-on-chip data from a partner"],
         "Verdict holds on data we did not build"),
        ("Then", "Put it in other hands", BLUE,
         ["Pilot with a Cell Painting screening group", "Their data, their labels"],
         "Outside team gets a verdict unaided"),
        ("Later", "Make it routine", INK,
         ["Packaged release, docs and tests", "Validation-report export", "Plate, well and batch checks"],
         "Standard pre-flight step"),
    ]
    for i, (when, title, colr, pts, gate) in enumerate(phases):
        x = i * 590
        el.append(f'<rect x="{x}" y="0" width="540" height="400" rx="12" fill="#FFFFFF" stroke="{RULE}" stroke-width="2.5"/>')
        el.append(f'<rect x="{x}" y="0" width="540" height="10" rx="5" fill="{colr}"/>')
        el.append(f'<text x="{x + 32}" y="66" font-size="24" font-weight="600" fill="{SLATE}" font-family="{SANS}">{when}</text>')
        el.append(f'<text x="{x + 32}" y="118" font-size="40" font-weight="500" fill="{INK}" font-family="{SERIF}">{title}</text>')
        for j, p in enumerate(pts):
            y = 170 + j * 50
            el.append(f'<circle cx="{x + 40}" cy="{y - 8}" r="7" fill="{colr}"/>')
            el.append(f'<text x="{x + 62}" y="{y}" font-size="25" fill="{INK}" font-family="{SANS}">{p}</text>')
        el.append(f'<line x1="{x + 32}" y1="310" x2="{x + 508}" y2="310" stroke="{RULE}" stroke-width="2"/>')
        el.append(f'<text x="{x + 32}" y="346" font-size="22" font-weight="600" fill="{colr}" font-family="{SANS}">Gate</text>')
        el.append(f'<text x="{x + 32}" y="374" font-size="22" fill="{SLATE}" font-family="{SANS}">{gate}</text>')
        if i < 2:
            el.append(f'<line x1="{x + 546}" y1="200" x2="{x + 584}" y2="200" stroke="{SLATE}" stroke-width="1.6" marker-end="url(#ah)"/>')
    return f'<svg viewBox="0 0 1720 400" role="img" aria-label="Roadmap in three phases: close the evidence gaps, put it in other hands, make it routine">{"".join(el)}</svg>'


def chart_calibration() -> str:
    bars = [("nominal", 5.0, RING), ("plain bootstrap", 10.3, GREY), ("calibrated", 2.6, MAG)]
    s = []
    for i, (lab, v, col) in enumerate(bars):
        x = 20 + i * 150
        hgt = v * 16
        s.append(f'<rect x="{x}" y="{200 - hgt}" width="100" height="{hgt}" rx="4" fill="{col}"/>')
        s.append(f'<text x="{x + 50}" y="{190 - hgt}" text-anchor="middle" font-size="26" font-weight="600" fill="{INK}" font-family="{SANS}">{v:g}%</text>')
        s.append(f'<text x="{x + 50}" y="236" text-anchor="middle" font-size="20" fill="{SLATE}" font-family="{SANS}">{lab}</text>')
    s.append(f'<line x1="0" y1="200" x2="470" y2="200" stroke="{RULE}" stroke-width="2"/>')
    return f'<svg viewBox="0 0 470 250" role="img" aria-label="Shuffled-label false positives: nominal 5 percent, plain bootstrap 10.3 percent, calibrated 2.6 percent">{"".join(s)}</svg>'


def chart_chip() -> str:
    sizes, power, mde = [20, 40, 60, 100, 200], [1, 11, 6, 33, 83], [0.72, 0.50, 0.37, 0.27, 0.19]
    s, w, x0, y0 = [], 700, 70, 20
    for t in (0, 25, 50, 75, 100):
        y = y0 + (100 - t) / 100 * 250
        s.append(f'<line x1="{x0}" y1="{y}" x2="{w - 10}" y2="{y}" stroke="{RULE}" stroke-width="1.5"/>')
        s.append(f'<text x="{x0 - 12}" y="{y + 7}" text-anchor="end" font-size="20" fill="{SLATE}" font-family="{SANS}">{t}%</text>')
    for i, (n, p, m) in enumerate(zip(sizes, power, mde)):
        x = x0 + 30 + i * 125
        hgt = max(p / 100 * 250, 3)
        s.append(f'<rect x="{x}" y="{y0 + 250 - hgt}" width="80" height="{hgt}" rx="4" fill="{MAG}"/>')
        s.append(f'<text x="{x + 40}" y="{y0 + 240 - hgt}" text-anchor="middle" font-size="24" font-weight="600" fill="{INK}" font-family="{SANS}">{p}%</text>')
        s.append(f'<text x="{x + 40}" y="{y0 + 282}" text-anchor="middle" font-size="24" font-weight="600" fill="{INK}" font-family="{SANS}">{n}</text>')
        s.append(f'<text x="{x + 40}" y="{y0 + 314}" text-anchor="middle" font-size="19" fill="{SLATE}" font-family="{SANS}">detects {m:.2f}</text>')
    return f'<svg viewBox="0 0 {w} 350" role="img" aria-label="Power to certify a true advantage: 1 percent at 20 compounds, 83 percent at 200">{"".join(s)}</svg>'


# ------------------------------------------------------------------ slides (one HTML section each)
def foot(n: int, dark: bool = False, backup: bool = False) -> str:
    dots = "".join(
        f'<circle cx="{7 + i * 20}" cy="7" r="5.5" fill="{MAG if i + 1 == n else "none"}" stroke="{MAG if i + 1 == n else ("#4A586A" if dark else RING)}" stroke-width="1.8"/>'
        for i in range(N))
    return (f'<div class="foot{" dark" if dark else ""}"><div class="fl"><span class="brand">Team BioBytes</span>'
            f'<svg width="{N * 20}" height="14" viewBox="0 0 {N * 20} 14">{dots}</svg></div><span>{"Backup" if backup else f"{n} / {N}"}</span></div>')


def sec(body: str, cls: str = "", n: int = 0, dark: bool = False) -> str:
    return f'<section class="slide {cls}">{body}{foot(n, dark, backup=cls == "bk") if n else ""}</section>'


def slides() -> list[str]:
    S = []
    S.append(f'''<section class="slide s1">
<div class="s1-left">
<p class="kicker">AI for Life Science Challenge · Finals</p>
<h1>Beyond the Count</h1>
<p class="sub">A calibrated audit for AI toxicity models built on Cell Painting</p>
<div class="team"><b>Team BioBytes</b><span>Vrushti · Dhairya</span></div>
</div>
<div class="s1-img"><img src="{b64('hepg2_dmso_1024.png')}" alt="HepG2 cells stained with Cell Painting"><span class="cap">HepG2 cells, Cell Painting Gallery (CC0)</span></div>
</section>''')
    S.append(sec(f'''<h2>A toxicity model can look smart while only counting cells</h2>
<div class="cols">
<div class="c-left">
<div class="bignum">31<span>of</span>49</div>
<p class="bigcap">endpoints well predicted by gene expression were predicted equally well by cell count alone</p>
<p class="src">Seal et al., <i>Nature Communications</i>, 2026</p>
<ul class="pts">
<li>Cell Painting: six dyes, five channels, thousands of features per cell</li>
<li>Many toxicity labels track cytotoxicity, so counting cells can score well</li>
<li>Cell-count-only AUC 0.68 against 0.67 for a full-profile network</li>
<li>FDA's 2026 draft guidance asks for technical characterization</li>
</ul>
</div>
<div class="c-right">
<figure class="panel"><img src="{b64('primary_dmso_1024.png')}" alt="Primary hepatocytes stained with Cell Painting"><figcaption><b>What we hope the model reads</b>shape, texture, organelles</figcaption></figure>
<figure class="panel"><div class="wells">{diagram_wells()}</div><figcaption><b>What it may be reading</b>how many cells are left</figcaption></figure>
</div>
</div>''', "s2", 2))
    S.append(sec(f'''<h2>One audit that tells you, endpoint by endpoint, whether morphology adds anything</h2>
<figure class="diagram">{diagram_architecture()}</figure>
<div class="four">
<div><h4>A strong baseline</h4><p>Cell count sees the whole dose-response curve</p></div>
<div><h4>Calibrated statistics</h4><p>p-values checked against a label-permutation null</p></div>
<div><h4>Power-aware verdicts</h4><p>Too little data is indeterminate, never "no effect"</p></div>
<div><h4>Label provenance</h4><p>An untested pair is missing, never a negative</p></div>
</div>''', "s3", 3))
    S.append(sec(f'''<h2>We tried to break the audit before asking anyone to trust it</h2>
<div class="three">
<div class="g"><div class="gbar" style="background:{MAG}"></div><h3>Calibrated, not assumed</h3>
<ul class="pts sm"><li>Shuffle the labels, rerun everything</li><li>Plain bootstrap: 10.3% false positives</li><li>Calibrated: 2.6% on held-out runs</li></ul>
<figure class="chart">{chart_calibration()}</figure><p class="note">Shuffled endpoints wrongly certified, nominal 5%. CP-CNN 6.5%, nested test 6.2%.</p></div>
<div class="g"><div class="gbar" style="background:{BLUE}"></div><h3>Powered, or silent</h3>
<ul class="pts sm"><li>15 actives and 15 non-hits to get a verdict</li><li>Fewer means indeterminate: 223 of 405 endpoints</li><li>Benjamini-Hochberg FDR across all powered endpoints</li></ul></div>
<div class="g"><div class="gbar" style="background:{INK}"></div><h3>Clean labels</h3>
<ul class="pts sm"><li>Untested is missing, never negative</li><li>107,103 of 281,196 label cells observed: 62% missing</li><li>Filtered binary labels only, found by reading the authors' notebook</li></ul></div>
</div>
<p class="strip">Fixed seeds · MD5-verified downloads · 69 tests · runs on a laptop CPU</p>''', "s4", 4))
    S.append(sec(f'''<h2>Morphology earns its cost on 15 endpoints, and the audit shows which</h2>
<div class="cols s5">
<div class="c-left">
<figure class="plate">{plate_map()}</figure>
<div class="legend"><span><i style="background:{MAG}"></i><b>15</b> morphology advantage</span><span><i style="background:{GREY}"></i><b>166</b> no detectable advantage</span><span><i class="ring"></i><b>223</b> indeterminate</span><span><i style="background:{BLUE}"></i><b>1</b> control</span></div>
<ul class="pts sm two"><li>MT certified with all three representations</li><li>Nested test certifies 23, all 15 included</li></ul>
</div>
<div class="c-right funnel">
<div class="f"><span class="n">405</span><span class="l">endpoints tested</span></div>
<div class="f"><span class="n">181</span><span class="l">enough data to judge</span></div>
<div class="f hi"><span class="n">15</span><span class="l">certified morphology advantage</span></div>
<div class="f"><span class="n">8</span><span class="l">certified under all three representations</span></div>
<p class="note">A plain bootstrap would have claimed 64. A second source (HepG2, exploratory) agrees on MT.</p>
</div>
</div>''', "s5", 5))
    S.append(sec(f'''<h2>A working tool anyone can run, not a one-off analysis</h2>
<div class="cols s6">
<div class="c-left"><div class="browser"><div class="bar"><i></i><i></i><i></i><span>beyond-the-count1.streamlit.app</span></div><img src="{b64('demo_hero.png')}" alt="Live demo"></div></div>
<div class="c-right">
<pre class="code"><span class="k">from</span> cpsa.api <span class="k">import</span> shortcut_audit
res = shortcut_audit(profile, baseline, labels)
res.table   <span class="c"># one verdict per endpoint</span></pre>
<ul class="pts sm"><li>Three inputs: profiles, cell counts, labels</li><li>One call, one verdict per endpoint</li><li>CPU only, open licences, no paid APIs</li></ul>
<p class="walk">Live demo walk-through</p>
<ol class="seq"><li>Verdict table</li><li>Enrichment</li><li>Chip planner</li></ol>
<p class="links">github.com/dhairya-pandya/beyond-the-count</p>
</div></div>''', "s6", 6))
    S.append(sec(f'''<h2>It changes a decision for each team that touches an imaging-based toxicity readout</h2>
<div class="cols s7">
<div class="c-left rows">
<div class="r"><b>Phenotypic AI team</b><span>Reports accuracy and moves on</span><em>Go/no-go check before accuracy is shown</em></div>
<div class="r"><b>Screening lead</b><span>Which endpoints justify full profiling?</span><em>Verdict table: count readout or morphology</em></div>
<div class="r"><b>Organ-on-chip group</b><span>How many compounds does a study need?</span><em>Planner sizes the study before the chip runs</em></div>
<div class="r"><b>Assay validation, regulatory</b><span>What supports a non-animal method?</span><em>A repeatable technical-characterization artifact</em></div>
</div>
<div class="c-right"><h3>At chip scale (simulation)</h3><figure class="chart">{chart_chip()}</figure>
<p class="note">Power to certify a true advantage, by number of compounds tested (20 to 200). False credit stays at 0 to 2%. No chip data used.</p></div>
</div>''', "s7", 7))
    S.append(sec(f'''<h2>What we proved</h2>
<div class="proof">
<div class="pr"><div class="pn">2.6%</div><div><h4>False certification, calibrated</h4><p>Shuffled labels, nominal 5%. The plain bootstrap gives 10.3%</p></div></div>
<div class="pr"><div class="pn">3 of 3</div><div><h4>MT certified in every representation</h4><p>CellProfiler, CP-CNN and DINOv2, with q of 0.001 to 0.039</p></div></div>
<div class="pr"><div class="pn">23</div><div><h4>Endpoints beyond cell count</h4><p>Nested test certifies 23, and all 15 of the main audit are among them</p></div></div>
<div class="pr"><div class="pn">0.51</div><div><h4>Layout and batch carry no signal</h4><p>Plate, well and batch alone score at chance; grouping by compound keeps replicate chips honest</p></div></div>
</div>''', "s8", 8))
    S.append(sec(f'''<h2>The path from a working audit to something labs rely on</h2>
<figure class="diagram road">{diagram_roadmap()}</figure>
<p class="ask">We would value introductions to groups running Cell Painting or organ-on-chip screens.</p>''', "s9", 9))
    dots = "".join(f'<circle cx="{(i % 27) * 22 + 10}" cy="{(i // 27) * 22 + 10}" r="6" fill="{MAG if i < 15 else ("#4A586A" if i < 181 else "#2E3B4B")}"/>' for i in range(405))
    S.append(f'''<section class="slide s10">
<div class="s10-main"><p class="kicker light">Close</p><h2>Before you trust a phenotypic AI readout, check what it is reading</h2>
<div class="links10"><div><b>Live demo</b><span>beyond-the-count1.streamlit.app</span></div><div><b>Code</b><span>github.com/dhairya-pandya/beyond-the-count</span></div><div><b>Data</b><span>Zenodo 10.5281/zenodo.17067683</span></div></div>
<p class="thanks">Team BioBytes. We would like to hear where it breaks.</p></div>
<svg class="dotfield" viewBox="0 0 594 330" aria-hidden="true">{dots}</svg>
{foot(10, dark=True)}</section>''')
    S.append(sec(f'''<p class="kicker">Backup</p><h2>Robustness checks, all consistent with the main result</h2>
<ul class="pts bk">
<li>Power threshold 10 / 15 / 20 / 30: 19 / <b>15</b> / 22 / 26 certified</li>
<li>Aggregation all / allpod / allpodcc: 18 / <b>15</b> / 3</li>
<li>Chemical-similarity folds (921 clusters): all 15 stay certified</li>
<li>Plate, well and batch alone: AUROC 0.51, chance</li>
<li>Null width 1.0 to 1.65: 67 down to 5 certified</li>
<li>Paper's own protocol: AUROC within +0.004 and +0.007</li>
<li>Simulation with known truth: false credit 2 to 4%, true effects found</li>
<li>Logistic regression comparator: median AUROC 0.62 against 0.65</li>
</ul>''', "bk", 11))
    S.append(sec(f'''<p class="kicker">Backup</p><h2>MT, LDH and the second source</h2>
<div class="bk two">
<div><h3>The paper's own readouts</h3><ul class="pts sm"><li>MT: strong count 0.873, full 0.916, q 0.019 / 0.039 / 0.001</li><li>LDH: strong count 0.957, full 0.969, q 0.27 / 0.29 / 0.076</li><li>Against the one-number baseline LDH looks like a win; against the curve it is not</li><li>38 of 51 wins vanish against the strong baseline</li></ul></div>
<div><h3>HepG2 against primary hepatocytes</h3><ul class="pts sm"><li>327 shared compounds, 49 endpoints, one cell-count number</li><li>Certified: 16 with HepG2, 1 with primary</li><li>MT and LDH agree; 10 of 16 also certified in the main audit</li><li>Weaker HepG2 baseline (one 10 µM count), so counts are not comparable</li></ul></div>
</div>''', "bk", 12))
    S.append(sec(f'''<p class="kicker">Backup</p><h2>The audit at chip scale, simulated</h2>
<table class="tbl"><thead><tr><th>Compounds</th><th>Cannot be judged</th><th>False credit, raw / calibrated</th><th>Power</th><th>Smallest detectable advantage</th></tr></thead><tbody>
<tr><td>20</td><td>100%</td><td>0% / 0%</td><td>1%</td><td>0.72</td></tr>
<tr><td>40</td><td>54%</td><td>2% / 0.5%</td><td>11%</td><td>0.50</td></tr>
<tr><td>60</td><td>29%</td><td>4% / 0.5%</td><td>6%</td><td>0.37</td></tr>
<tr><td>100</td><td>1%</td><td>8% / 1.5%</td><td>33%</td><td>0.27</td></tr>
<tr><td>200</td><td>0%</td><td>8% / 2%</td><td>83%</td><td>0.19</td></tr></tbody></table>
<ul class="pts sm bk"><li>Replicate chips: random labels reach AUROC 0.85 if each chip is independent, 0.51 if grouped by compound</li><li>A chip changes labels, the 3D count proxy, features, grouping and sample size</li></ul>''', "bk", 13))
    return S


CSS = f"""
@import url('https://fonts.googleapis.com/css2?family=Hanken+Grotesk:wght@400;500;600;700&family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;0,6..72,600;1,6..72,400&display=swap');
*{{box-sizing:border-box;margin:0;padding:0}}
html,body{{background:{PAPER};width:1920px;height:1080px;overflow:hidden}}
body{{font-family:{SANS};color:{INK};font-size:32px;line-height:1.35;-webkit-font-smoothing:antialiased;font-weight:450}}
.slide{{width:1920px;height:1080px;position:relative;overflow:hidden;background:{PAPER};padding:84px 112px 0}}
h1,h2,h3,h4{{font-family:{SERIF};font-weight:500;letter-spacing:-.012em;color:{INK}}}
h2{{font-size:68px;line-height:1.08;max-width:1640px;font-weight:600}}
h3{{font-size:42px;line-height:1.1;margin-bottom:22px}}
h4{{font-size:36px;line-height:1.15;margin-bottom:12px;font-weight:600}}
p{{font-size:30px;line-height:1.4}}
.kicker{{font-size:21px;font-weight:600;color:{MAG};letter-spacing:.06em;margin-bottom:18px}}
.kicker.light{{color:#E58AC8}}
.foot{{position:absolute;left:112px;right:112px;bottom:38px;display:flex;justify-content:space-between;align-items:center;font-size:24px;color:#3E4A57}}
.foot .fl{{display:flex;align-items:center;gap:24px}}.foot .brand{{font-weight:600;color:{INK}}}
.foot.dark{{color:#9FB0C2}}.foot.dark .brand{{color:#fff}}
ul.pts{{list-style:none;margin-top:24px}}
ul.pts li{{position:relative;padding-left:34px;margin-bottom:16px;font-size:34px;line-height:1.3}}
ul.pts li::before{{content:"";position:absolute;left:0;top:.5em;width:12px;height:12px;border-radius:50%;background:{MAG}}}
ul.pts.sm li{{font-size:30px;margin-bottom:14px}}ul.pts.sm li::before{{width:10px;height:10px}}
ul.pts.two{{columns:1}}
.note{{font-size:25px;color:#3E4A57;line-height:1.35;margin-top:14px}}
.src{{font-size:22px;color:{SLATE};margin-top:8px}}
.cols{{display:flex;gap:72px;margin-top:40px}}
.c-left{{flex:1.2}}.c-right{{flex:1}}
/* 1 title */
.s1{{padding:0}}
.s1-left{{position:absolute;left:112px;top:0;bottom:0;width:880px;display:flex;flex-direction:column;justify-content:center}}
.s1 h1{{font-size:156px;line-height:.95;letter-spacing:-.03em}}
.s1 .sub{{font-size:40px;line-height:1.2;color:{SLATE};margin-top:34px;max-width:780px}}
.s1 .team{{margin-top:70px;display:flex;flex-direction:column;gap:6px;border-left:6px solid {MAG};padding-left:24px}}
.s1 .team b{{font-size:36px;font-weight:600}}.s1 .team span{{font-size:28px;color:{SLATE}}}
.s1-img{{position:absolute;right:0;top:0;width:960px;height:1080px}}.s1-img img{{width:100%;height:100%;object-fit:cover;display:block}}
.s1-img .cap{{position:absolute;right:28px;bottom:26px;color:#fff;font-size:20px;opacity:.9}}
/* 2 */
.bignum{{font-family:{SERIF};font-size:230px;line-height:.9;letter-spacing:-.035em;color:{MAG};font-weight:500;display:flex;align-items:baseline;gap:22px}}
.bignum span{{font-size:64px;letter-spacing:0;color:{SLATE};font-style:italic}}
.bigcap{{font-size:33px;line-height:1.25;max-width:760px;margin-top:12px;font-weight:500}}
.c-right{{display:flex;flex-direction:column;gap:24px}}
figure.panel{{border-radius:12px;overflow:hidden;background:#fff;border:2px solid {RULE}}}
figure.panel img{{width:100%;height:300px;object-fit:cover;display:block}}
.wells svg{{display:block;width:100%;height:300px}}
figure.panel figcaption{{padding:14px 22px 16px;font-size:24px;color:{SLATE};display:flex;flex-direction:column;gap:4px}}
figure.panel figcaption b{{color:{INK};font-weight:600;font-size:26px}}
/* 3 */
.diagram{{margin-top:30px}}.diagram svg{{width:100%;height:auto;display:block}}
.four{{display:grid;grid-template-columns:repeat(4,1fr);gap:44px;margin-top:34px}}
.four h4{{color:{MAG}}}.four p{{font-size:26px}}
/* 4 */
.three{{display:grid;grid-template-columns:repeat(3,1fr);gap:56px;margin-top:40px}}
.g .gbar{{height:9px;border-radius:5px;margin-bottom:22px}}
.g h3{{font-size:42px}}.chart svg{{width:100%;height:auto;display:block;margin-top:8px}}
.strip{{position:absolute;left:112px;bottom:90px;font-size:25px;color:{SLATE}}}
/* 5 */
.s5 .c-left{{flex:1.6}}.s5 .c-right{{flex:1}}
figure.plate{{width:100%}}figure.plate svg{{width:100%;height:auto;display:block;max-height:470px}}
.legend{{display:flex;gap:30px;flex-wrap:wrap;font-size:23px;margin-top:14px}}
.legend b{{font-weight:700;font-size:26px;margin-right:4px}}
.legend i{{display:inline-block;width:16px;height:16px;border-radius:50%;margin-right:10px;vertical-align:-2px}}
.legend i.ring{{border:2.5px solid {RING};background:transparent}}
ul.pts.two{{margin-top:18px}}
.funnel{{display:flex;flex-direction:column;gap:12px;border-left:3px solid {RULE};padding-left:44px;justify-content:center}}
.f{{display:flex;flex-direction:column}}
.f .n{{font-family:{SERIF};font-size:96px;line-height:1;letter-spacing:-.02em;font-weight:500}}
.f .l{{font-size:25px;color:{SLATE};margin-top:2px}}
.f.hi .n{{color:{MAG};font-size:144px}}
/* 6 */
.s6 .c-left{{flex:1.5}}.s6 .c-right{{flex:1}}
.browser{{border-radius:12px;overflow:hidden;border:2px solid {RULE};background:#fff;box-shadow:0 14px 34px rgba(23,34,46,.10)}}
.browser .bar{{height:44px;background:#E8ECEF;display:flex;align-items:center;gap:9px;padding:0 18px;font-size:19px;color:{SLATE}}}
.browser .bar i{{width:13px;height:13px;border-radius:50%;background:#C5CDD5;display:inline-block}}
.browser .bar span{{margin-left:14px}}
.browser img{{width:100%;display:block;height:690px;object-fit:cover;object-position:top}}
pre.code{{background:{INK};color:#E8ECEF;border-radius:10px;padding:26px 30px;font-family:ui-monospace,Menlo,Consolas,monospace;font-size:24px;line-height:1.55;white-space:pre-wrap}}
pre.code .k{{color:#E58AC8}}pre.code .c{{color:#8FA0B3}}
.walk{{margin-top:34px;font-size:21px;font-weight:600;color:{SLATE};letter-spacing:.04em}}
ol.seq{{list-style:none;display:flex;gap:12px;margin-top:14px}}
ol.seq li{{font-size:24px;border:2px solid {RULE};background:#fff;border-radius:999px;padding:8px 20px}}
.links{{margin-top:30px;font-size:26px;color:{BLUE};font-weight:600}}
/* 7 */
.s7 .c-left{{flex:1.35}}.s7 .c-right{{flex:1}}
.s7 .c-right h3{{font-size:38px}}
.rows{{display:flex;flex-direction:column}}
.r{{display:flex;flex-direction:column;gap:4px;padding:20px 0;border-top:2px solid {RULE}}}
.r:last-child{{border-bottom:2px solid {RULE}}}
.r b{{font-size:33px;font-weight:600}}.r span{{font-size:28px;color:#3E4A57}}.r em{{font-style:normal;font-size:27px;color:{MAG};font-weight:500}}
/* 8 */
.lim{{display:flex;flex-direction:column;margin-top:36px}}
.lr{{display:grid;grid-template-columns:5fr 6fr 5fr;gap:44px;padding:26px 0;border-top:2px solid {RULE};align-items:start}}
.lr:last-child{{border-bottom:2px solid {RULE}}}
.lr h4{{font-size:34px;margin:0}}.lr p{{font-size:29px;line-height:1.35}}.lr p.how{{color:{MAG};font-weight:500}}
/* proof slide */
.proof{{display:flex;flex-direction:column;margin-top:44px}}
.pr{{display:grid;grid-template-columns:330px 1fr;gap:48px;align-items:center;padding:30px 0;border-top:2px solid {RULE}}}
.pr:last-child{{border-bottom:2px solid {RULE}}}
.pn{{font-family:{SERIF};font-size:112px;line-height:1;letter-spacing:-.02em;color:{MAG};font-weight:600;white-space:nowrap}}
.pr h4{{font-size:38px;margin:0 0 8px;color:{INK}}}
.pr p{{font-size:30px;line-height:1.35;color:#3E4A57}}
/* 9 */
.road{{margin-top:50px}}.road svg{{width:100%;height:auto;display:block}}
.ask{{margin-top:50px;font-family:{SERIF};font-size:42px;line-height:1.2}}
/* 10 */
.s10{{background:{INK};color:#fff}}
.s10-main{{position:absolute;left:112px;top:150px;width:1400px}}
.s10 h2{{color:#fff;font-size:92px;line-height:1.05}}
.links10{{display:grid;grid-template-columns:repeat(3,auto);gap:70px;margin-top:70px;justify-content:start}}
.links10 div{{display:flex;flex-direction:column;gap:6px;border-left:5px solid {MAG};padding-left:22px}}
.links10 b{{font-size:24px;color:#E58AC8;font-weight:600}}.links10 span{{font-size:28px}}
.thanks{{margin-top:56px;font-size:32px;color:#B9C6D4}}
.dotfield{{position:absolute;right:112px;bottom:110px;width:594px}}
/* backup */
ul.pts.bk li{{font-size:29px}}
.bk.two{{display:grid;grid-template-columns:1fr 1fr;gap:70px;margin-top:40px}}
table.tbl{{border-collapse:collapse;width:100%;font-size:28px;margin-top:36px}}
table.tbl th{{text-align:left;font-weight:600;color:{SLATE};font-size:22px;padding:12px 16px;border-bottom:3px solid {INK}}}
table.tbl td{{padding:14px 16px;border-bottom:2px solid {RULE}}}
table.tbl td:first-child{{font-weight:700}}
"""


def page(section: str) -> str:
    return f'<!doctype html><html lang="en"><head><meta charset="utf-8"><style>{CSS}</style></head><body>{section}</body></html>'


def chrome() -> str:
    for c in (os.environ.get("CHROME"), "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", "/usr/bin/google-chrome", "/usr/bin/chromium"):
        if c and Path(c).exists():
            return c
    sys.exit("Chrome not found; set CHROME=/path/to/binary")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    PREVIEWS.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="finals_"))
    exe = chrome()
    pngs = []
    for i, sec_html in enumerate(slides(), 1):
        html = work / f"slide_{i:02d}.html"
        html.write_text(page(sec_html))
        png = work / f"slide_{i:02d}.png"
        subprocess.run([exe, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--window-size=1920,1080",
                        "--force-device-scale-factor=1", "--virtual-time-budget=12000", f"--screenshot={png}", html.as_uri()],
                       check=True, capture_output=True)
        pngs.append(png)
        Image.open(png).convert("RGB").save(PREVIEWS / f"slide_{i:02d}.png", optimize=True)
    imgs = [Image.open(p).convert("RGB") for p in pngs]
    pdf = OUT / "beyond_the_count_finals.pdf"
    imgs[0].save(pdf, save_all=True, append_images=imgs[1:], resolution=144.0)
    print(f"wrote {pdf} ({len(imgs)} slides)")


if __name__ == "__main__":
    main()
