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
*(CellProfiler features, `allpod` aggregation, 404 endpoints + a positive control; CP-CNN and DINOv2 as robustness checks)*
* **The audit reproduces the paper.** Under the paper's own CV protocol our AUROCs match its published per-endpoint AUROCs to within 0.01 on average (r ≈ 0.8; 0.9 for well-powered endpoints).
* **55 % of endpoints cannot support any claim** (too few positives). 291 endpoints look like a naive "morphology wins"; 181 are powered; 88 pass an uncorrected test.
* **We audited our own audit — and it over-called.** A label-permutation control (424 runs) shows plain bootstrap p-values reject 13 % of the time at the nominal 5 % (CellProfiler features). Calibrating p-values against that empirical null (validated on held-out permutation runs: 4.9 % at 5 %) turns **62 raw BH "wins" into 9 certified endpoints**; zero false calls in permuted data.
* **The 9 are a stable core**: all are also certified under CP-CNN and DINOv2 (representations certify 9 / 33 / 40 endpoints; 46 certified under any, 27 under at least two). Median ΔAUROC over all powered endpoints is +0.095 (87 % positive) — the signal is broad, but per-endpoint certainty is limited by power.
* **A weak baseline hides the shortcut.** Against the paper's one-number cell-count baseline 27 endpoints are certified; 20 vanish against a baseline that uses the whole dose–response curve. Native readouts: morphology beats cell count for MT with CP-CNN and DINOv2 (q ≈ 0.005; borderline with CellProfiler) but is **never** certified for LDH — whereas the one-number baseline would call LDH a win. The positive-control endpoint (label *defined* from cell count) is solved perfectly (AUROC 1.00) by the strong baseline.
* **Biology: honest about what is and is not shown.** Cell-free (purely biochemical) endpoints show no certified advantage (0 of the 8 that can be tested). Cytotoxicity endpoints look enriched among certified endpoints, but they have about four times more actives, and the difference disappears once endpoint size is controlled (odds ratio ≈ 1 for CellProfiler and CP-CNN; undecided for DINOv2), so we report it as descriptive. The mechanistic evidence is at compound level: morphology adds most for compounds that alter cells without killing them (preliminary).
* **Not a layout artefact.** Plate / well / batch alone predict labels at chance (AUROC 0.51).
* **Honest about confidence.** Raw model probabilities are over-confident (median calibration slope 0.33); the demo uses cross-validated recalibration and shows each endpoint's calibration diagnostics.
* **Sensitivity.** Certified counts for CellProfiler: 8 / 9 / 16 / 24 at 10 / 15 / 20 / 30 minimum positives (pre-specified 15, not tuned); chance-floored baseline retains 9 of 9; aggregation rule changes the strong-baseline count (18 / 9 / 0) — a stated limitation.

## Why it matters
The audit is a drop-in reliability check for any phenotypic-profiling toxicology dataset with a cell-count-like baseline — including organ-on-chip assays that will face the same shortcut risk. We do **not** claim chip results: the data are a 2D plate assay, a single 44 h time point, donor variability unmodeled.

## Reproduce
`README.md`: `scripts/01_download_zenodo_data.py` → `02_fetch_labels.py` → `03_run_audit.py` → `04_run_enrichment.py` → `streamlit run demo/app.py`. All CPU, fixed seeds, MD5-verified downloads (Zenodo CC-BY 4.0; authors' repo BSD-3-Clause).
