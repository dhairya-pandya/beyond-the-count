#!/usr/bin/env python
"""Assemble report/kaggle_writeup.md in the structure the competition recommends for the Kaggle Writeup.

Order: category declaration, demo video, code repository, project summary (200-300 words), technical report, optional demo link, team.
The technical report body is report/submission_report.md (its headings are pushed one level down, and the Mermaid diagram is replaced by
the pipeline figure, because Kaggle does not render Mermaid).

  python scripts/27_build_kaggle_writeup.py
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = "https://raw.githubusercontent.com/dhairya-pandya/beyond-the-count/main/"
REPO = "https://github.com/dhairya-pandya/beyond-the-count"
DEMO = "https://beyond-the-count1.streamlit.app/"
VIDEO = "<<VIDEO URL>>"

SUMMARY = """Cell Painting stains cells with six dyes and turns them into thousands of morphology features, and it is becoming a scalable, animal-free readout for toxicology. The risk is a shortcut: many toxicity labels track cytotoxicity, so a model that only knows how many cells survived can score well without reading any morphology. Regulators ask for technical characterization of such methods, and organ-on-chip pipelines that adopt AI image readouts will inherit the same risk.

Beyond the Count is a reusable, dataset-agnostic statistical audit. For every endpoint it compares a full morphology model with a strong cell-count baseline that sees the whole dose-response curve, on identical compound-grouped folds. A paired compound bootstrap, calibrated against a label-permutation null, a power check and Benjamini-Hochberg FDR control give each endpoint one of four verdicts: morphology advantage, no detectable advantage, indeterminate, or positive control.

On the public OASIS primary human hepatocyte data (1,085 compounds, 405 endpoints, three image representations), 181 endpoints have enough data to test and 15 are certified for a morphology advantage, including metabolic activity (MT) in all three representations. A plain bootstrap would have called 64; the permutation calibration corrects that. A nested test (morphology plus cell count against cell count alone) certifies 23 endpoints, including all 15, and a second source, EU-OPENSCREEN HepG2, agrees on MT.

For organ-on-chip, a simulation at 20 to 200 compounds shows what the audit can conclude at chip scale, and the demo's planner sizes a study before it is run. Code, 69 tests and a live Streamlit demo run on a laptop CPU with public data only."""


def push_headings(text: str) -> str:
    out, fence = [], False
    for line in text.splitlines():
        if line.startswith("```"):
            fence = not fence
        out.append("#" + line if (not fence and re.match(r"#{2,5} ", line)) else line)
    return "\n".join(out)


def main() -> None:
    body = (ROOT / "report" / "submission_report.md").read_text()
    body = body.split("\n", 1)[1].lstrip("\n")  # drop the H1
    body = re.sub(r"^\*\*Code:\*\*.*\n+", "", body, count=1, flags=re.M)  # links are in the header of the Writeup
    body = re.sub(r"```mermaid.*?```", f"![System architecture]({RAW}results/figures/pipeline.png)", body, flags=re.S)
    body = body.replace("*Each endpoint runs through the matched models", "*Figure: each endpoint runs through the matched models")
    def figure(path: str, caption: str) -> str:
        return f"![{caption}]({RAW}{path})\n\n*Figure: {caption}*\n\n"

    anchors = {
        "### Robustness checks, all consistent with the main result": figure("results/figures/correction_funnel.png", "from 292 naive wins to 15 certified endpoints (CellProfiler)") +
        figure("results/figures/null_control.png", "label-permutation control: raw and calibrated p-values"),
        "Calibration keeps false credit near zero at every size": figure("results/chip_scale/chip_scale.png", "the audit at organ-on-chip scale (simulation)"),
    }
    for anchor, fig in anchors.items():
        assert anchor in body, anchor
        body = body.replace(anchor, fig + anchor, 1)
    words = len(SUMMARY.split())
    assert 200 <= words <= 300, words
    doc = f"""# Beyond the Count: A Reusable Statistical Audit for Cell-Count Shortcuts in Cell Painting Toxicology

**Category: Model & Algorithm**

## Demo video

{VIDEO} (public, no login, 5 minutes or less)

## Code repository

{REPO} (public; MIT license; `README.md`, `requirements.txt`, 69 unit tests, precomputed results, entry scripts `demo/app.py` and `scripts/03_run_audit.py`)

## Project summary

{SUMMARY}

## Technical report

The full report is below, and as a PDF: {REPO}/blob/main/report/technical_report.pdf

{push_headings(body)}

## Optional demo link

**Live interactive demo:** {DEMO} (public, no login). It has a plate-map view of all 405 endpoints, per-endpoint results with calibration and detectable effect, enrichment, a cell-lines tab with example images, and the organ-on-chip study planner. The app is kept awake by a scheduled GitHub Actions visit every six hours, and its status is shown in the page footer.

If the hosted demo is unavailable, run it locally in two commands (precomputed results are included, no download needed):

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt && .venv/bin/pip install -e .
.venv/bin/streamlit run demo/app.py
```

## Team

Two members, covering both AI/CS and biology:

| Member | Discipline and role |
|---|---|
| Vrushti | Biology and data: label construction and provenance, dataset QA, biological interpretation |
| Dhairya | ML and statistics: models, calibration, multiple-testing framework, Streamlit demo, reproducible pipeline |

The team composition is also declared in Section 1 of the technical report.
"""
    (ROOT / "report" / "kaggle_writeup.md").write_text(doc)
    print(f"wrote report/kaggle_writeup.md ({len(doc.split())} words; summary {words} words)")


if __name__ == "__main__":
    main()
