**Category: Model & Algorithm**

# Cell Painting Shortcut Audit — does morphology really beat counting cells?

**Team:** <<TEAM COMPOSITION — names, disciplines>> · **Code:** <<GITHUB URL>> · **Demo:** <<STREAMLIT URL>> · **Video (≤ 5 min):** <<VIDEO URL>> · **Technical report:** `report/technical_report.md` (PDF attached)

## The problem
Cell Painting is becoming a workhorse of human-relevant, non-animal toxicology (a "new approach methodology", NAM). But phenotypic AI can pass a benchmark for the wrong reason: for cytotoxicity-flavoured readouts, a model that only *counts surviving cells* can score suspiciously well, so a sophisticated morphology model may add nothing beyond that trivial signal. Regulators now ask for exactly this kind of technical characterisation (FDA draft NAM guidance, March 2026: context of use, human biological relevance, technical characterization, fit for purpose).

## What we built
A reusable, dataset-agnostic **statistical audit** that answers, for every assay endpoint:
1. Does the full morphological profile beat a cell-count-only model **after Benjamini–Hochberg correction** for testing hundreds of endpoints at once?
2. Is there **enough data to say** (≥ 15 actives and ≥ 15 inactives)? If not the verdict is *indeterminate* — never a silent AUROC.
3. Where morphology wins, is the advantage **concentrated in a coherent assay / target family** (Fisher exact test on EPA's own annotations) or arbitrary?

Applied to the public primary-human-hepatocyte data of Ewald et al. (*Cell Systems* 2026; 1,085 compounds; 404 endpoints: 292 cell-based + 72 cell-free + 38 cytotoxicity ToxCast/Tox21 endpoints + the study's MT and LDH readouts, plus a cell-count positive control). Everything runs on a CPU laptop; code, tests, and an interactive demo are public.

Methods in one paragraph: per-plate DMSO-MAD normalisation and per-compound aggregation (re-implemented from the authors' pipeline) → three XGBoost models on identical compound-grouped folds (full morphology; the paper's one-number cell-count baseline; and a *stronger* cell-count-only baseline using the whole dose–response curve) → ΔAUROC with a paired compound bootstrap → label-permutation null for p-value calibration → power check → BH-FDR over powered endpoints → calibration and enrichment.

## What we found
*(CellProfiler features, `allpod` aggregation, 404 endpoints + a positive control; CP-CNN and DINOv2 as robustness checks, each with its own matched permutation null)*
* **The audit reproduces the paper.** Under the paper's own CV protocol our AUROCs match its published per-endpoint AUROCs to within 0.01 on average (r ≈ 0.8; 0.9 for well-powered endpoints).
* **55 % of endpoints cannot support any claim** (too few positives). 292 endpoints look like a naive "morphology wins"; 181 are powered; 95 pass an uncorrected test.
* **We audited our own audit — and it over-called.** A label-permutation control (379 powered runs at the audit's own settings) shows plain bootstrap p-values reject 10 % of the time at the nominal 5 % (CP-CNN: 18 %). Calibrating p-values against that null (validated on held-out permutation runs: 2.6 % at 5 %) turns **64 raw BH "wins" into 15 certified endpoints** for CellProfiler; zero false calls in permuted CellProfiler and CP-CNN data.
* **The count is a band, not a point.** It is 67 / 38 / 23 / 15 / 5 for a null sd of 1.0 / 1.2 / 1.33 / 1.45 / 1.65, so we report endpoints and the band, not "how many benefit". Median ΔAUROC over all powered endpoints is +0.094 (88 % positive): the signal is broad, but per-endpoint certainty is limited by power.
* **Overlap across representations.** CellProfiler and CP-CNN certify 15 endpoints each and DINOv2 54; every endpoint certified by the first two is also certified by DINOv2, and 8 are certified under all three (22 under at least two).
* **A weak baseline hides the shortcut.** Against the paper's one-number cell-count baseline 51 endpoints are certified; 38 of those vanish against a baseline that uses the whole dose–response curve. Native readouts: morphology beats cell count for MT with all three representations (q = 0.019 / 0.039 / 0.001) but is **never** certified for LDH at 5 % (q = 0.27 / 0.29 / 0.076) — whereas the one-number baseline would call LDH a win. The positive-control endpoint (label *defined* from cell count) is solved perfectly (AUROC 1.00) by the strong baseline.
* **Biology: honest about what is and is not shown.** Cell-free (purely biochemical) endpoints show no certified advantage (0 of the 8 that can be tested). Cytotoxicity endpoints look enriched among certified endpoints, but they have about four times more actives, and the difference is not significant once endpoint size is controlled, so we report it as descriptive. The mechanistic evidence is at compound level: for compounds active in an endpoint, morphology's ranking benefit over cell count is +0.15 for compounds that change cells without any cell loss, MT or LDH hit, +0.13 for MT/LDH hits without cell loss, and +0.01 for compounds that cause cell loss.
* **Not a layout artefact.** Plate / well / batch / replicate counts alone predict labels at chance (AUROC 0.51).
* **Honest about confidence.** Raw model probabilities are over-confident (median calibration slope 0.33); the demo uses cross-validated recalibration and shows each endpoint's calibration diagnostics.
* **Label semantics handled.** A ToxCast 0 means "not a (filtered) hit", untested pairs stay missing, and a stratified check shows how the cytotoxicity filter (which applies to about a quarter of compound–endpoint pairs) changes where morphology's advantage appears.
* **Sensitivity.** Certified counts for CellProfiler at 10 / 15 / 20 / 30 minimum positives: 19 / 15 / 22 / 26 (15 fixed in advance, not tuned); chance-floored baseline retains 13 of 15; aggregation rules `all` / `allpod` / `allpodcc` certify 18 / 15 / 3 (a stated limitation); chemical-similarity cross-validation folds leave the median AUROC unchanged (0.650 → 0.651) and keep all 15.

## Why it matters
The audit is a drop-in reliability check for any phenotypic-profiling toxicology dataset with a cell-count-like baseline — including organ-on-chip assays that will face the same shortcut risk. We do **not** claim chip results: the data are a 2D plate assay, a single 44 h time point, donor variability unmodeled.

## Reproduce
`README.md`: `scripts/01_download_zenodo_data.py` → `02_fetch_labels.py` → `03_run_audit.py` → `04_run_enrichment.py` → `streamlit run demo/app.py`. All CPU, fixed seeds, MD5-verified downloads (Zenodo CC-BY 4.0; authors' repo BSD-3-Clause).
