# beyond-the-count

**Does Cell Painting see more than cell count?** A statistically controlled *shortcut audit* for image-based toxicology profiling (Python package `cpsa`).

## Cell Painting Shortcut Audit

A statistically rigorous audit of whether Cell Painting morphological profiles actually beat a **cell-count-only baseline**
for predicting toxicology endpoints in primary human hepatocytes — with Benjamini–Hochberg FDR control across hundreds of
endpoints, explicit *indeterminate* verdicts for under-powered endpoints, calibration checks, and an assay/target-family
enrichment analysis of where morphology's advantage lives.

*AI for Life Science Challenge — category: **Model & Algorithm**.*

Source data: Ewald et al. (2026) *Cell Systems* 17(5):101566 — profiles on Zenodo
[10.5281/zenodo.17067683](https://zenodo.org/records/17067683) (CC-BY 4.0), labels/annotations from the authors'
[analysis repo](https://github.com/jessica-ewald/2024_09_09_Axiom_OASIS) (BSD-3-Clause).

> Repository layout: `src/cpsa/` package · `scripts/` numbered pipeline stages · `tests/` · `demo/` Streamlit app · `docs/` (problem statement, plan) · `report/` (technical report, write-up, video script) · `results/` generated outputs.

## Quick start (CPU only, ~1 GB download, laptop-scale)

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt && .venv/bin/pip install -e .
# NB: call .venv/bin/python directly; `source .venv/bin/activate` also works unless your path contains a ':'

.venv/bin/python scripts/01_download_zenodo_data.py            # profiles + metadata (MD5-verified). add --only cpcnn metadata for a quick start
.venv/bin/python scripts/02_fetch_labels.py                    # clones the authors' repo (labels, endpoint annotations, SI tables)

# the audit: raw run -> label-permutation null -> null-calibrated verdicts -> enrichment
.venv/bin/python scripts/03_run_audit.py --feature-set cellprofiler --agg allpod     # -> results/cellprofiler_allpod/audit_table.csv (~30 min on 10 cores)
.venv/bin/python scripts/08_null_control.py --feature-set cellprofiler --agg allpod --n-endpoints 100 --n-perms 2   # -> null_control.csv (repeat with --seed 1 --append)
.venv/bin/python scripts/15_calibrate_verdicts.py --config cellprofiler_allpod        # primary verdicts / q-values use calibrated p-values
.venv/bin/python scripts/04_run_enrichment.py --config cellprofiler_allpod            # -> results/<config>/enrichment.csv

.venv/bin/python scripts/07_finalize_results.py --primary cellprofiler_allpod         # headline results/ + figures/ + calibrated OOF probabilities
.venv/bin/python scripts/05_build_demo_assets.py                                      # small tables the demo reads
.venv/bin/streamlit run demo/app.py                                                   # interactive demo over precomputed results
.venv/bin/python -m pytest                                                            # 30 unit tests
```

All precomputed results needed by the demo are committed under `results/`, so the demo runs without any download or re-computation. `requirements-lock.txt` lists the exact versions used; optional extras for scripts 14/16: `pip install markdown tifffile pillow imagecodecs`.

Additional analyses (optional, all reproducible): `06_compare_to_paper.py` (agreement with the source paper's metrics), `09_technical_probe.py`
(plate/well/batch-only probe), `10_feature_families.py` (CellProfiler feature-family attribution), `11_paper_faithful_cv.py` (re-run under the paper's
own CV protocol), `12_headline_numbers.py` (every number in the report), `13_sensitivity_postprocess.py` (power threshold / alpha / chance-floored
baseline), `14_fetch_example_images.py` (a dozen demo images, individually fetched from the Cell Painting Gallery), `16_build_report_pdf.py`
(report PDF via headless Chrome), `run_all_configs.sh` (all feature-set × aggregation configurations; secondary ones run on powered endpoints only).

`03_run_audit.py` options: `--feature-set {cellprofiler,cpcnn,dino}`, `--agg {all,allpod,allpodcc}`, `--categories`, `--endpoints`
(pilot runs), `--powered-only`, `--n-repeats`, `--n-boot`, `--n-jobs`, `--seed`.

## What it does

For every endpoint (292 cell-based + 72 cell-free + 38 cytotoxicity ToxCast/Tox21 endpoints, plus the paper's native MT / LDH hits):

| step | module |
|---|---|
| per-plate DMSO-MAD normalisation, correlation filter, per-compound aggregation (`all` / `allpod` / `allpodcc`) | `cpsa.data.preprocess` |
| three XGBoost models with identical folds: **full** morphology profile, **scalar_cc** (paper's mean cell count), **strong_cc** (cell-count dose–response curve, min, AUC, cell-count POD — still *only* cell-count information) | `cpsa.models` |
| repeated stratified grouped CV (out-of-fold probabilities), paired compound-bootstrap of ΔAUROC = full − baseline | `cpsa.models.train_eval` |
| label-permutation null → empirical-null calibration of the bootstrap p-values (raw bootstrap p is anti-conservative) | `cpsa.audit.null_control`, `cpsa.stats.null_calibration` |
| power check (≥15 actives and ≥15 non-hits, else **indeterminate**), BH-FDR over powered endpoints | `cpsa.stats` |
| calibration of the full model (Brier, Brier skill, ECE, calibration slope, reliability bins) | `cpsa.stats.calibration` |
| Fisher-exact enrichment of "morphology advantage" endpoints by target family / assay design / cell type | `cpsa.biology.enrichment` |

Verdicts (BH on null-calibrated p-values; raw-bootstrap verdicts kept as `verdict_uncalibrated`): `morphology_advantage` (powered, Δ>0, FDR q<0.05) · `no_advantage` (powered, otherwise) · `indeterminate` (under-powered) ·
`positive_control` (the `cell_count` native endpoint — its label is *defined* from cell count; the strong baseline must score ≈1.0).

## Inputs / outputs

* Input: `data/raw/*.parquet` (Zenodo), `data/external/ewald_repo/` (labels). Both git-ignored.
* Output per configuration `results/<feature_set>_<agg>/`: `audit_table.csv` (one row per endpoint), `oof_predictions.parquet`, `enrichment.csv`.
* `audit_table.csv` columns: endpoint annotations · `n_compounds, n_active, n_inactive` · `scalar_cc_AUROC/PRAUC`, `strong_cc_AUROC/PRAUC`,
  `full_AUROC/PRAUC` · `delta_AUROC` (full − strong_cc), CI, `delta_z`, `bootstrap_p` (raw), `calibrated_p`, `fdr_q`, `verdict`, `verdict_uncalibrated` (+ `_scalar` variants vs the paper's scalar baseline) ·
  `calibration_brier, brier_skill, ece, calib_slope, brier_recalibrated`.

## Compute

Everything runs on CPU. ~1000 compounds × a few hundred to ~5,000 features; one full audit (~400 endpoints × 3 models × 5 folds × 3 repeats)
takes ~15 min (CP-CNN) to ~30 min (CellProfiler) on a 10-core laptop; 200 permutation runs 10–25 min. No GPU, no paid services.

## Headline results (CellProfiler, `allpod`; details in `report/technical_report.pdf`)

404 endpoints → 181 powered (55 % indeterminate) → 62 'wins' under a plain bootstrap → **9 certified** after null calibration (all 9 also certified with CP-CNN and DINOv2; cytotoxicity-burst endpoints enriched, q = 0.007; MT advantage in 2 of 3 representations, LDH never; cell-free 0 of 8). Permuted-label control: raw false-positive rate 13 % at nominal 5 %, calibrated 4.9 % on held-out runs.

## Reading the labels correctly

A ToxCast `0` means **not a (filtered) hit**, never "tested negative"; untested pairs are missing and are never zero-filled; for the 38 cytotoxicity columns `1` = cytotoxic in that cell type/tissue; the cytotoxicity filter that turns some hits into 0 applies only to ~35 % of cell-based records. Labels come only from the repo's `*_binary.parquet` files. Details: `docs/PROBLEM_STATEMENT.md` §3.3b; `scripts/19_filter_stratified.py` re-scores results by whether the filter could apply.

## Scope & limitations

2D hepatocyte plate assay (not organ-on-chip); precomputed profiles only; single 44 h time point; donor variability not modeled;
compound-level aggregation (so plate/well technical covariates are undefined and not used). See the technical report.
