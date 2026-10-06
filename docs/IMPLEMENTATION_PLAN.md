# Cell Painting Shortcut Audit — Implementation Plan

**Competition:** AI for Life Science Challenge (5th Pazhou Algorithm Competition / Kaggle)
**Submission category:** Model & Algorithm
**Deadline:** October 10, 2026
**Plan date:** October 2, 2026 (9 days remaining)

---

## 1. Project Summary

**One-line pitch:** A statistically rigorous audit of whether Cell Painting morphological profiling actually outperforms a trivial cell-counting baseline across toxicology endpoints in primary human hepatocytes — with proper correction for testing hundreds of endpoints at once, explicit handling of underpowered endpoints, and a biological explanation for where morphology's real advantage lies.

**The problem:** Image-based phenotypic profiling assays like Cell Painting are being adopted across pharma and regulatory science as a scalable, human-relevant alternative to animal testing. But phenotypic AI models have a documented failure mode: they can pass benchmarks for the wrong reason. The JUMP Cell Painting consortium found that for many toxicity-related benchmarks, a model that only counts surviving cells can score suspiciously well — meaning an apparently sophisticated morphology model may be adding nothing beyond that trivial signal.

**What we're building:** A reusable statistical audit pipeline that, for any Cell Painting toxicology dataset, answers three questions per assay endpoint:
1. Does the full morphological profile beat a cell-count-only baseline, after correcting for the fact that we're testing hundreds of endpoints simultaneously (so some "wins" are expected by chance alone)?
2. Is there enough data to even answer that question for this endpoint, or should the result be marked indeterminate rather than reported as a false positive/negative?
3. For endpoints where morphology does win, is there a biologically coherent reason (e.g., concentrated in a particular assay/target family), or does the advantage look arbitrary?

**Why this is a real contribution, not a reproduction:** The source paper (Ewald et al., 2026, *Cell Systems*) already compares morphology to a cell-count baseline in its Figure 3 but does not apply multiple-testing correction across the ~412 endpoints it evaluates, does not explicitly flag underpowered endpoints as indeterminate, and does not test whether morphology's advantage clusters in particular target/assay families. Those three gaps are this project's contribution.

**Why it matters beyond this one dataset:** The audit method is dataset-agnostic — it is written to be rerun on any Cell Painting (or similar phenotypic profiling) toxicology dataset with a cell-count-like baseline feature. This is deliberate: the goal is a reusable reliability-check tool for phenotypic AI in life science, not a one-off number for one paper.

---

## 2. Background (for report-writing, condensed)

- **Source paper:** Ewald, J.D. et al. "Cell Painting for cytotoxicity and mode-of-action analysis in primary human hepatocytes." *Cell Systems* 17(5), 101566 (2026). DOI: 10.1016/j.cels.2026.101566. Preprint posted Jan 2025 (bioRxiv), accepted Feb 2026, published online March 27, 2026.
- **Experimental design:** 1,085 compounds (pharmaceuticals, pesticides, industrial chemicals), 8 concentrations each, 2 replicates, primary human hepatocytes, 384-well format, 44-hour exposure. Three parallel readouts: LDH (membrane damage), MT/Realtime-Glo (metabolic activity), and Cell Painting (6-dye, 5-channel imaging). Cell count extracted from images as a fourth readout.
- **Feature extraction compared in the paper:** CellProfiler (classic, hand-engineered), CP-CNN (Cell Painting-specific pretrained CNN), DINO (general-purpose self-supervised vision transformer, no cell-specific tuning).
- **Headline finding:** Cell Painting detects bioactivity at far lower doses than cytotoxicity assays (2.5–16× lower depending on comparison). Morphology beats the cell-count baseline for predicting MT readouts but not LDH readouts. Against ~412 external ToxCast/Tox21 endpoints, Cell Painting predicts cytotoxicity and cell-based endpoints above a random baseline, but not cell-free (pure biochemical) endpoints — and cell-count alone also beats random for cytotoxicity endpoints, which is the shortcut-learning signal this project investigates further.
- **Regulatory relevance:** FDA's 2026 draft guidance on New Approach Methodologies (NAMs) names "technical characterization" and "fit-for-purpose" validation as core requirements for non-animal toxicology methods. An audit of whether a NAM's apparent predictive power is a real signal or a confound speaks directly to that validation requirement.

---

## 3. Scope

**In scope:**
- Statistical audit (baseline vs. full model) across available ToxCast/Tox21 endpoints and the paper's native LDH/MT/cell-count readouts.
- Multiple-testing correction (Benjamini-Hochberg FDR) across all tested endpoints.
- Explicit power-based "indeterminate" flagging for underpowered endpoints.
- Model calibration check (are confidence scores trustworthy).
- Endpoint-level biological enrichment analysis (are "morphology wins" endpoints concentrated in particular assay/target families).
- A lightweight interactive demo over precomputed results.
- Technical report, code repository, and demo video per competition requirements.

**Out of scope (explicitly, and stated as such in the report):**
- This is a 2D hepatocyte plate assay, not an organ-on-a-chip. We are not claiming otherwise. Positioned as a validation-methodology contribution relevant to OoC toxicology assays that will face the same shortcut-learning risk.
- No raw image modeling from scratch — we use the precomputed profile features (CellProfiler/CP-CNN/DINO) already published.
- No new wet-lab data generation.
- Single time point (44h) — no longitudinal/kinetic modeling.
- Donor-to-donor variability in hepatocytes is not modeled (data appears to be single-donor or pooled; confirm during data audit, Section 7, Task A1).

---

## 4. System Architecture

### 4.1 High-level pipeline

```
[Zenodo: profiles + metadata]        [Outcome labels: paper SI or ToxCast pull]
         |                                         |
         v                                         v
   load_profiles.py                          load_labels.py
         |                                         |
         +-------------------+---------------------+
                              v
                      build_dataset.py
            (joins profiles + labels by compound/well,
             enforces compound-level grouping for CV)
                              |
                              v
                   per-endpoint modeling loop
          +-------------------+-------------------+
          |                                       |
          v                                       v
   baseline.py                             full_model.py
 (cell-count + plate/well                 (morphology profile
  technical features, XGBoost)             features, XGBoost)
          |                                       |
          +-------------------+-------------------+
                              v
                      train_eval.py
        (compound-grouped K-fold CV, AUROC/PRAUC,
         bootstrap delta-AUROC + p-value per endpoint)
                              |
                              v
                   multiple_testing.py
            (Benjamini-Hochberg FDR across endpoints)
                              |
                              v
                     power_check.py
       (flags endpoints below n_active threshold as
        "indeterminate", excluded from FDR pool)
                              |
                              v
                     calibration.py
         (reliability diagram / Brier score on full model)
                              |
                              v
                   results/audit_table.csv
     (endpoint, n, n_active, baseline_AUROC, full_AUROC,
      delta, p_value, FDR_q, verdict, calibration_score)
                              |
              +---------------+----------------+
              v                                v
     enrichment.py                        demo/app.py
 (Fisher's exact test: is target/     (interactive: pick compound/
  assay family enriched in             endpoint, see verdict,
  "morphology wins" vs. "no            confidence, and audit table)
  advantage" endpoint groups)
```

### 4.2 Directory structure

```
cell-painting-shortcut-audit/
├── IMPLEMENTATION_PLAN.md          # this document
├── README.md                       # setup + reproduction instructions
├── requirements.txt
├── pyproject.toml
├── .gitignore                      # excludes data/raw, data/external (large files)
├── data/
│   ├── raw/                        # downloaded Zenodo parquet files (gitignored)
│   ├── external/                   # ToxCast pulls / paper SI labels (gitignored)
│   └── processed/                  # joined, modeling-ready tables
├── src/
│   └── cpsa/                       # "Cell Painting Shortcut Audit" package
│       ├── __init__.py
│       ├── data/
│       │   ├── load_profiles.py
│       │   ├── load_labels.py
│       │   └── build_dataset.py
│       ├── models/
│       │   ├── baseline.py
│       │   ├── full_model.py
│       │   └── train_eval.py
│       ├── stats/
│       │   ├── multiple_testing.py
│       │   ├── power_check.py
│       │   └── calibration.py
│       ├── biology/
│       │   └── enrichment.py
│       └── viz/
│           └── plots.py
├── scripts/
│   ├── 01_download_zenodo_data.py
│   ├── 02_fetch_labels.py
│   ├── 03_run_audit.py             # end-to-end orchestration, entry point
│   └── 04_run_enrichment.py
├── notebooks/
│   ├── 00_explore_profiles.ipynb
│   ├── 01_explore_labels.ipynb
│   ├── 02_results_exploration.ipynb
│   └── 03_demo_prototyping.ipynb
├── demo/
│   ├── app.py                      # Streamlit app, reads results/audit_table.csv
│   └── assets/
├── results/
│   ├── audit_table.csv
│   └── figures/
├── report/
│   └── technical_report.md
└── tests/
    ├── test_build_dataset.py
    ├── test_train_eval.py
    └── test_multiple_testing.py
```

### 4.3 Tech stack

| Layer | Choice | Why |
|---|---|---|
| Language | Python 3.10+ | Matches ecosystem of source repo and all data tooling |
| Data I/O | `pandas`, `pyarrow` | Native Parquet support for the Zenodo files |
| Modeling | `xgboost`, `scikit-learn` | Tabular classification; no GPU required at this data scale |
| Stats | `statsmodels` (BH-FDR via `multipletests`), `scipy.stats` (Fisher's exact test, bootstrap) | Standard, well-tested implementations |
| Interpretability | `shap` (optional, time-permitting) | Feature attribution for the full model, supports "interpretable/explainable" judging criterion |
| Visualization | `matplotlib` / `plotly` | Static figures for report; plotly if demo needs interactivity |
| Demo | `streamlit` (or `gradio`) | Fast to build, no frontend code needed, runs on CPU |
| Environment | `venv` + `requirements.txt`, or `conda`/`environment.yml` | Keep it simple — judges must reproduce without paid services |
| Notebooks | Jupyter | Exploration and figure generation only — no production logic lives only in notebooks |

**Compute note for the report:** All modeling runs on CPU. Expected dataset size is on the order of ~17,000–20,000 well-level rows (1,085 compounds × 8 concentrations × 2 replicates, plus controls) with feature vectors in the hundreds-to-low-thousands dimension range (CellProfiler) or a few hundred to ~2048 dimensions (CP-CNN/DINO embeddings). XGBoost training per endpoint is expected to take seconds to low minutes — no GPU, no paid cloud compute needed. State this explicitly in the technical report; it directly supports the "Reproducibility & Implementation Quality" scoring criterion.

---

## 5. Data Pipeline — Detailed Specification

### 5.1 Inputs

| File | Source | Size | Contents |
|---|---|---|---|
| `cellprofiler_raw.parquet` | Zenodo 10.5281/zenodo.17067683 | 413.4 MB | Well-aggregated CellProfiler features |
| `cpcnn_raw.parquet` | same | 48.2 MB | Well-aggregated CP-CNN features |
| `dino_raw.parquet` | same | 399.9 MB | Well-aggregated DINOv2 features |
| `index.parquet` | same | 2.5 MB | Maps wells/compounds to raw image URLs in Cell Painting Gallery |
| `metadata.parquet` | same | 0.7 MB | Compound identity, concentration, plate, well, batch |
| Outcome labels (LDH, MT, cell count, ToxCast hit-calls) | **Unresolved — see Task A1** | Unknown | Required to compute any endpoint AUROC |

### 5.2 Known open question: outcome labels

The Zenodo bundle contains profiles and metadata only — **no LDH/MT/cell-count values or ToxCast hit-calls**. This must be resolved before any modeling work (Track B) can begin. Resolution order:
1. Check the Cell Systems article's Supplementary Information (hosted alongside the paper on ScienceDirect / Cell Press — typically an "mmc" supplementary data archive).
2. Check `0_prepare_data` and `1_snakemake` folders in the GitHub repo (`jessica-ewald/2024_09_09_Axiom_OASIS`) — labels may be compiled there even though not mentioned in the top-level README.
3. Fallback: pull ToxCast/Tox21 hit-call data directly from the EPA CompTox Dashboard (or the `invitrodb`/`ctxR` access routes) using compound identifiers (CASRN/DTXSID) from `metadata.parquet`, and cross-reference against the paper's stated curation (48 cytotoxicity, 292 cell-based, 72 cell-free endpoints) to replicate their endpoint set as closely as possible. This path is slower and should only be used if 1 and 2 fail.

**This is the single highest-risk item in the entire plan.** No modeling work should start until this is resolved.

### 5.3 Dataset construction (`build_dataset.py`)

- Join profiles + metadata + labels on compound ID + concentration + replicate/well key.
- **Critical leakage control:** each compound appears in up to 16 rows (8 concentrations × 2 replicates). All downstream train/test splitting must be done at the **compound level**, not the row level — i.e., all rows for a given compound must land entirely in train or entirely in test within any fold. This mirrors the exact leakage risk already identified in the ALS project this team evaluated earlier; the same discipline applies here. Implement via `GroupKFold` (scikit-learn) grouped on compound ID, not standard `KFold`.
- Output: one long-format table (endpoint, compound, concentration, feature vector, label) ready for per-endpoint slicing.

---

## 6. Modeling & Statistics — Detailed Specification

### 6.1 Per-endpoint model pair

For each endpoint `E`:
- **Baseline model:** XGBoost classifier, features = `[cell_count, plate_id (encoded), well_row, well_col, batch/day if available]`. This is the "shortcut" model — it should only be able to exploit cell survival and technical/positional confounds, nothing about cell morphology.
- **Full model:** XGBoost classifier, same target, features = full morphology profile (primary: CellProfiler, for interpretability; CP-CNN and DINO as secondary robustness checks if time allows in Days 6–7).

### 6.2 Cross-validation

- `GroupKFold` (k=5), grouped by compound ID, as specified in 5.3.
- Stratify by label where feasible given group constraints.
- Record per-fold AUROC and PRAUC for both models.

### 6.3 Significance testing per endpoint

- Compute `delta = full_model_AUROC − baseline_AUROC` (mean across folds).
- Bootstrap resample held-out predictions (~1,000 iterations) to build a distribution of `delta`; derive a one-sided p-value for `delta > 0`.
- This avoids relying on normality assumptions and works with small per-endpoint sample sizes.

### 6.4 Multiple-testing correction

- Apply Benjamini-Hochberg FDR correction (`statsmodels.stats.multitest.multipletests`) across **all endpoints that pass the power check (6.5)** — do not include indeterminate endpoints in the correction pool, since that would just dilute power further for no benefit.
- This is the project's core statistical contribution over the source paper, which does not appear to apply this correction.

### 6.5 Power check / indeterminate flagging

- Define a minimum threshold (starting heuristic: **n_active ≥ 15 AND n_inactive ≥ 15** within the endpoint's compound set; revisit after seeing the real label distribution in Day 2–3).
- Endpoints below threshold → verdict = `"indeterminate"`, reported separately, explicitly not claimed as either a finding of advantage or no-advantage.
- This directly addresses the paper's own data: it notes some ToxCast categories have as few as ~33 tested compounds with ~7% active — i.e., single-digit positive counts are plausible for some endpoints, and these must not be silently reported as a clean AUROC.

### 6.6 Final verdict categories (per endpoint)

| Verdict | Condition |
|---|---|
| **Morphology advantage** | Passes power check AND `delta > 0` AND FDR-corrected q-value < 0.05 |
| **No advantage over cell count** | Passes power check AND does not meet the above |
| **Indeterminate** | Fails power check |

### 6.7 Calibration check

- On held-out folds, bin the full model's predicted probabilities and compare to observed outcome frequency (reliability diagram).
- Report Brier score as a single summary number.
- Purpose: a model can have a good AUROC and still give overconfident wrong answers — this check is what lets the demo responsibly show a "confidence" number.

### 6.8 Output artifact

`results/audit_table.csv` — one row per endpoint:
`endpoint_id | endpoint_description | assay_target_family | n_compounds | n_active | baseline_AUROC | baseline_PRAUC | full_AUROC | full_PRAUC | delta_AUROC | bootstrap_p | FDR_q | verdict | calibration_brier`

This table is the single artifact that feeds both the enrichment analysis (Section 7) and the demo (Section 8).

---

## 7. Biology Track — Detailed Specification

| Task | Description | Output |
|---|---|---|
| **A1** | Resolve outcome-label source (see 5.2) | Confirmed label source, joined table |
| **A2** | Build compound ↔ endpoint ↔ outcome table | Clean table feeding `build_dataset.py` |
| **A3** | Pull endpoint-level target/assay-family metadata from ToxCast's own assay annotations (EPA publishes `intended_target_family` / assay description fields per endpoint) — **note the refinement from earlier planning: enrichment is done at the endpoint/assay level using EPA's existing curation, not by annotating each of the 1,085 compounds individually.** This is simpler and reuses data EPA has already curated. | Endpoint → target-family mapping table |
| **A4** | Confirm/refine endpoint category definitions (cytotoxicity / cell-based / cell-free, reusing or refining the paper's own buckets) | Final category scheme, applied before modeling begins |
| **A5** | **(after `audit_table.csv` exists)** Run Fisher's exact test: is a given target family over-represented among "morphology advantage" endpoints vs. "no advantage" endpoints? | A short, honestly-reported result — including a null result if target family doesn't predict where morphology wins |
| **A6** | Write limitations section: 2D not chip, single time point, donor variability unmodeled, label-source caveats, any endpoints excluded for data-quality reasons | Report section |
| **A7** | Write motivation/impact framing: ties to FDA NAM draft guidance's "technical characterization" and "fit-for-purpose" principles, and to the JUMP consortium's own prior finding about cell-count shortcuts | Report section |

---

## 8. Demo Specification

**Goal:** a reviewer should be able to pick a compound and/or endpoint and see the audit's verdict, not just read numbers in a report.

**Build:** Streamlit app, single page.
- Input: dropdown/search over compounds (from `metadata.parquet`) and/or endpoints (from `audit_table.csv`).
- Output for a selected compound: its measured values across available readouts, plus per-endpoint verdicts and confidence (from the full model), pulled from precomputed `audit_table.csv` and per-compound prediction cache (not live retraining).
- Optional visual: if time allows (Day 6–7), pull 1–2 representative images for the selected compound using `index.parquet` to look up the specific S3 key and fetch only that file — **do not bulk-download or recursively scan the raw `cpg0037-oasis` bucket (confirmed ~650TB)**.
- Everything in the demo reads from precomputed artifacts — no GPU, no live model training, fast to run for a reviewer.

---

## 9. Reproducibility & Submission Requirements Checklist

Mapped directly to the competition's stated requirements:

- [ ] **Category declaration** at top of Kaggle Writeup: Model & Algorithm
- [ ] **Demo video** (≤5 min, public link or Kaggle attachment, viewable without login)
- [ ] **Public code repository** (GitHub), including:
  - [ ] Core code (`src/cpsa/`)
  - [ ] `README.md` with setup + reproduction instructions
  - [ ] `requirements.txt` / `environment.yml`
  - [ ] Clear entry script: `scripts/03_run_audit.py`
  - [ ] Input/output descriptions
- [ ] **Technical report** (15–20 pages excl. references/appendices): problem definition, data sources/licenses, methods, results, reliability analysis & limitations, reproduction instructions, external tool/data citations
- [ ] **Team composition declared** in report (cross-disciplinary bonus: +0.5 on Interpretability & Reliability dimension)
- [ ] **Optional demo link**: Streamlit app, publicly accessible without login
- [ ] **Final reproducibility dry-run** from a clean environment before submission (Day 9)

---

## 10. Day-by-Day Schedule (Oct 2 → Oct 10)

| Date | Bio track | ML track | Joint |
|---|---|---|---|
| **Oct 2 (today)** | A1 (resolve labels — top priority) | Set up repo structure, B1 (load/inspect profiles) | Registration form if not done; lock scope |
| **Oct 3** | Finish A1/A2 | B2 (merge), B3 scaffold (baseline model) | — |
| **Oct 4** | A3, A4 | B3 full run, start B4 (full model) | — |
| **Oct 5** | Review early results | B4 complete, B5 (scale across endpoints) | — |
| **Oct 6** | — | B6 (FDR correction), B7 (power check) | **Checkpoint: is there a real, defensible finding? If the signal is too thin, narrow scope now** |
| **Oct 7** | A5 (enrichment) | B8 (calibration), B9 (package as reusable script) | Start demo (Section 8) |
| **Oct 8** | A6, A7 | Support demo, finalize `audit_table.csv` | Finish demo, start technical report |
| **Oct 9** | Report review | Report review | Record demo video, repo cleanup, reproducibility dry-run |
| **Oct 10** | — | — | Submit |

---

## 11. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Outcome labels (A1) not found in SI or repo | Medium | High — blocks everything | Fallback to direct ToxCast pull; start this check immediately on Oct 2 |
| Too few endpoints pass the power check (6.5) | Medium | Medium — thinner result set | Still a valid, honestly-reported finding; adjust framing, not abandon project |
| No endpoint shows a significant morphology advantage after FDR correction | Low-medium | Medium — changes report's headline claim | This is itself a legitimate, interesting finding (treat as primary result, not a failure) — decide framing at the Oct 6 checkpoint |
| Compound-level grouping leaves too few compounds per fold for some endpoints | Medium | Medium | Reduce to 3-fold for sparse endpoints, document the deviation |
| Target-family metadata from ToxCast incomplete for some endpoints | Medium | Low | Report coverage % explicitly; exclude unannotated endpoints from A5 only, not from 6.6 |
| Demo scope creep | Medium | Low | Demo reads only precomputed artifacts — no live training, keep it to 1 page |

---

## 12. Sources & References

**Primary paper**
- Ewald, J.D. et al. (2026). "Cell Painting for cytotoxicity and mode-of-action analysis in primary human hepatocytes." *Cell Systems* 17(5), 101566. https://doi.org/10.1016/j.cels.2026.101566
- bioRxiv preprint (Jan 2025): https://www.biorxiv.org/content/10.1101/2025.01.22.634152

**Data**
- Zenodo data record (profiles, metadata, image index): https://zenodo.org/records/17067683 (DOI 10.5281/zenodo.17067683), CC-BY 4.0
- GitHub analysis repository: https://github.com/jessica-ewald/2024_09_09_Axiom_OASIS (also mirrored at https://github.com/broadinstitute/2024_09_09_Axiom_OASIS), BSD-3-Clause
- Raw images (reference only, not for bulk download): `s3://cellpainting-gallery/cpg0037-oasis/` — Cell Painting Gallery, hosted on AWS Registry of Open Data, CC0, no-sign-request access. Confirmed to contain ~650TB under this prefix.
- Cell Painting Gallery documentation: https://github.com/broadinstitute/cellpainting-gallery

**Regulatory / motivation context**
- FDA draft guidance (2026) on New Approach Methodologies — technical characterization and fit-for-purpose validation principles for non-animal toxicology methods (verify current version/link directly on fda.gov before citing in the final report, as guidance documents are periodically updated).
- JUMP Cell Painting consortium prior finding on cell-count-driven shortcut performance in cytotoxicity benchmarks — cite the JUMP Cell Painting dataset paper (Chandrasekaran et al.) and associated consortium publications; confirm exact citation when drafting the report.

**Toxicology label data (for Task A1 fallback)**
- EPA CompTox Chemicals Dashboard: https://comptox.epa.gov/dashboard/
- ToxCast/Tox21 data access routes (`invitrodb`, `ctxR`, or dashboard bulk download) — confirm current access method at project start, as EPA tooling changes over time.

**Competition**
- Kaggle competition page: https://www.kaggle.com/competitions/ai-4-s-open-innovation-artificial-intelligence-for-life-scien
- Official competition website: https://www.aicompetition-pz.com/
- Required registration form: https://docs.google.com/forms/d/e/1FAIpQLSdRAat5jIunRaFNh_NntsVeJUnekEJDrbuokLZ32LFgCwPtiA/viewform

**Note on verification status:** All data/repository links above were directly checked during project scoping (Oct 1–2, 2026). The FDA guidance and JUMP consortium citations should be re-verified and pinned to exact document versions/DOIs before they appear in the final technical report, since this plan cites them from earlier conversational research rather than a fresh check at writing time.
