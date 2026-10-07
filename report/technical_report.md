# Cell Painting Shortcut Audit: does morphology beat a cell-count baseline, once we correct for testing hundreds of endpoints?

**Competition:** AI for Life Science Challenge (5th Pazhou Algorithm Competition) · **Category:** Model & Algorithm
**Team:** <<TEAM COMPOSITION — declare members and disciplines (cross-disciplinary bonus)>>
**Code:** <<GITHUB URL>> · **Demo:** <<STREAMLIT URL>> · **Video:** <<VIDEO URL>>

> All numbers are regenerated from `results/` (see §7). Placeholders in `<<…>>` (team, URLs) are for the team to fill before submission.

---

## 1. Problem definition

Image-based phenotypic profiling (Cell Painting) is being adopted as a scalable, human-relevant new approach methodology (NAM) for toxicology. Models trained on such profiles can score well for the wrong reason: for cytotoxicity-related readouts, a model that only *counts surviving cells* can match a sophisticated morphology model, so an apparent "morphology signal" may be a cell-count shortcut. The source study (Ewald et al., 2026) compares morphology to a cell-count baseline, but, over ~400 external endpoints, does not (i) correct for the number of endpoints tested, (ii) separate endpoints with too few positives to support any conclusion, or (iii) test whether morphology's advantage clusters in a coherent biological family.

**Contribution.** A reusable, dataset-agnostic *audit* that answers, per endpoint: (1) does the full morphological profile beat a cell-count-only model after Benjamini–Hochberg FDR control across all tested endpoints? (2) is there enough data to say — otherwise the endpoint is reported as *indeterminate*; (3) where morphology wins, is the advantage concentrated in an assay/target family, or arbitrary?

**Scope and non-claims.** The data are a 2D primary-human-hepatocyte plate assay, not an organ-on-a-chip. We position the work as a validation-methodology contribution relevant to organ-on-chip toxicology, whose readouts face the same shortcut risk; we do not claim chip results. No raw-image modeling (precomputed profiles only), no new wet-lab data, a single 44 h time point, donor variability not modeled.

## 2. Data sources and licenses

| Item | Source | License |
|---|---|---|
| Well-level profiles (CellProfiler, CP-CNN, DINOv2), metadata, image index | Zenodo 10.5281/zenodo.17067683 | CC-BY 4.0 |
| ToxCast/Tox21 binary hit-calls, endpoint annotations (target family, assay design), SI tables (LDH/MT/cell-count hits and points of departure) | github.com/jessica-ewald/2024_09_09_Axiom_OASIS | BSD-3-Clause |
| Source paper | Ewald et al., *Cell Systems* 17(5):101566 (2026), doi:10.1016/j.cels.2026.101566 | — |

Design of the source experiment: 1,085 compounds (pharmaceuticals, pesticides, industrial chemicals), 8 concentrations × 2 replicates, primary human hepatocytes, 384-well, 44 h exposure; LDH, metabolic-activity (MT) and Cell Painting (6 dyes, 5 channels) readouts with cell count extracted from the images. We use only the published, precomputed profiles; raw images (Cell Painting Gallery, ~650 TB bucket) are not downloaded.

### 2.1 Dataset statistics
| quantity | value |
|---|---|
| wells / plates / DMSO control wells | 21,456 / 65 / 4,849 |
| compounds with an OASIS id (profiled, modeled) | 967 (966 for DINOv2) of 1,086 compound names; the rest are blinded |
| concentrations per compound | 8 (one compound 7), 2 replicates, spanning 13–16 plates from two production batches |
| raw → retained features after variance and correlation (|r| > 0.9) filters | CellProfiler 5,640 → 984 · CP-CNN 672 → 666 · DINOv2 4,608 → 4,424 |
| endpoints | 292 cell-based + 72 cell-free + 38 cytotoxicity (ToxCast/Tox21) + MT, LDH, cell_count (native) = 405 |
| profiled compounds with a label per endpoint, median (min–max) | cell-based 202 (10–652) · cell-free 20 (7–131) · cytotoxicity 229 (13–667) |
| actives per endpoint among profiled compounds, median (10th–90th percentile) | cell-based 15 (5–39) · cell-free 8 (4–23) · cytotoxicity 69 (5–132) |
| endpoints with ≥ 15 actives and ≥ 15 non-hits among modeled compounds | 181 of 404 (cell-based 147 / 292, cytotoxicity 24 / 38, cell-free 8 / 72, native 2) |

Label provenance: the Zenodo record contains profiles and metadata only; outcome labels were located in the authors' public analysis repository (`1_snakemake/inputs/annotations/*_binary.parquet`, `2_downstream_analysis/compiled_results/SI_tables/`). We therefore did not need to re-derive hit-calls from EPA invitrodb. The repository's cytotoxicity file contains 38 endpoint columns (the paper text refers to 48); we use what is shipped.

## 3. Methods

### 3.1 Profile preprocessing (re-implemented from the authors' pipeline)
Per plate, features are centred and scaled by the median and MAD of that plate's DMSO controls; features that are non-finite, or have zero MAD / |MAD/median| ≤ 10⁻³ in any plate, are dropped; a greedy correlation filter (|r| > 0.9) removes redundant features. Per-compound profiles are obtained by averaging treated wells under three aggregation rules taken from the source repository: `all` (all wells), `allpod` (wells above the compound's Cell Painting point of departure, POD), `allpodcc` (wells above the POD and below the cell-count POD, with the source's fallbacks). PODs are taken from the published SI tables.

### 3.2 Models
For every endpoint, three XGBoost classifiers (150 trees, learning rate 0.05, `scale_pos_weight` = neg/pos — identical to the source classifier) are trained on identical folds:

* **full** — the morphology profile (CellProfiler primary; CP-CNN and DINOv2 as robustness checks);
* **scalar_cc** — the source paper's baseline: the mean cell count (one number);
* **strong_cc** — a deliberately *stronger* shortcut model that still sees **only cell-count information**: the scalar count, cell count at each of the 8 concentrations (÷ plate DMSO median), their minimum and mean, and the cell-count POD.

The strong baseline is the primary comparator: a verdict against a one-number baseline can be an artefact of that baseline being weak. The technical-covariate baseline in the original plan (plate, well position, batch) is not applicable at compound level, because each compound's profile averages wells from 13–16 plates.

### 3.3 Cross-validation and effect estimate
Stratified, compound-grouped, shuffled 5-fold CV (folds reduced to min(5, n_pos, n_neg) for sparse endpoints), repeated 3× with different fold seeds; out-of-fold probabilities are averaged over repeats. ΔAUROC = AUROC(full) − AUROC(baseline) on pooled out-of-fold predictions. A paired bootstrap over compounds (1,000 resamples) gives a 95% interval and a standard error SE; the raw one-sided p-value is p = (1 + #{Δ* ≤ 0})/(B + 1). This captures *evaluation-sample* variability only — not the variability of the fitted models — which turns out to matter (§3.4).

### 3.4 Auditing the audit: label-permutation null and calibrated p-values
We re-ran the identical pipeline on hundreds of endpoints whose labels were randomly permuted across compounds (true Δ = 0 by construction; `scripts/08_null_control.py`). Under this null the z-score z = Δ/SE has a standard deviation clearly above 1, so raw bootstrap p-values are anti-conservative. Following Efron's empirical-null idea we estimate sd₀ from the powered permutation runs and use the calibrated one-sided p-value p_cal = P(Z > z/sd₀). The estimate is feature-set specific (§4.4); split-half validation (sd₀ from the first runs, error rates on held-out runs) checks that the calibration holds. Raw bootstrap p-values and verdicts are retained alongside (`bootstrap_p`, `verdict_uncalibrated`).

### 3.5 Power check, multiple-testing correction, verdicts
An endpoint is *powered* if it has ≥ 15 actives and ≥ 15 inactives among profiled compounds (pre-specified heuristic; sensitivity in §4.10 — it was not tuned after seeing results). Benjamini–Hochberg FDR is applied to the calibrated p-values across **powered** endpoints only. Verdicts: `morphology_advantage` (powered, Δ > 0, q < 0.05), `no_advantage` (powered, otherwise), `indeterminate` (under-powered; excluded from the FDR pool and from the enrichment test, never reported as a finding either way). The `cell_count` native endpoint, whose label is defined from the cell-count curve, is a positive control (the strong baseline must reach AUROC ≈ 1) and is outside the pool. BH assumes independence or positive dependence; endpoints share compounds and are correlated, which we flag as a limitation.

### 3.6 Calibration
On out-of-fold probabilities of the full model: Brier score, Brier skill versus the prevalence predictor, expected calibration error (10 bins), calibration slope of a logistic recalibration, and the Brier score after cross-validated Platt recalibration. Because training re-weights classes, raw probabilities are scores rather than frequencies; the demo shows this explicitly.

### 3.7 Biological enrichment
For each grouping (assay/target family, assay design type, cell line, tissue, category), a one-sided Fisher exact test asks whether the group is over-represented among `morphology_advantage` endpoints relative to the other powered endpoints (min group size 5; BH within each grouping). Target families come from EPA's own `intended_target_family` annotation, so no per-compound annotation is required. Null results are reported as such.

### 3.8 Implementation
The package `src/cpsa/` mirrors the pipeline: `data/` (`load_profiles`, `preprocess`, `load_labels`, `build_dataset`, `technical`), `models/` (`train_eval`: grouped CV, XGBoost wrapper, paired bootstrap; `baseline`, `full_model`: feature-set definitions), `stats/` (`power_check`, `multiple_testing`, `null_calibration`, `calibration`), `biology/` (`enrichment`, `feature_families`), `viz/plots`, and `audit.py` (end-to-end orchestration with joblib parallelism over endpoints). Numbered scripts in `scripts/` are thin command-line wrappers; every figure and number in this report is regenerated from `results/` by scripts 07, 12 and 13. Design choices that serve reproducibility: one parquet per (representation, aggregation) cached in `data/processed/`; fixed seeds everywhere; MD5-verified downloads; no hidden state in notebooks (notebooks are exploration only); 30 unit tests including a grouped-CV leakage test (duplicated compounds with noise features must score at chance), a toy BH reference, an empirical-null recovery test and a reproduction of the paper's unshuffled fold protocol. The demo reads only precomputed artifacts.

## 4. Results

Primary configuration (pre-specified in the project plan): **CellProfiler features, `allpod` aggregation**; numbers from `results/cellprofiler_allpod/` (`headline_numbers.json`, regenerated by `scripts/12_headline_numbers.py`). CP-CNN and DINOv2 are robustness checks and were audited with the same settings and their own matched permutation nulls. 404 endpoints were tested (292 cell-based, 72 cell-free, 38 cytotoxicity ToxCast/Tox21 endpoints plus the paper's MT and LDH hits); `cell_count` is the positive control. All three audits cover every endpoint (3 CV repeats, 1,000 bootstrap resamples), use the concentration-aware cell-count features, and were run once, in full, after the label-semantics corrections of §2.

![Pipeline](../results/figures/pipeline.png)

### 4.1 The pipeline reproduces the source paper's AUROC level
Against the paper's published per-endpoint AUROCs (CP-CNN features, `allpod`, 179 powered endpoints), re-running our models under the **paper's own protocol** (unshuffled `StratifiedKFold`, one run, the paper's row order; `scripts/11_paper_faithful_cv.py`) gives a mean AUROC difference of +0.004 (full model) and +0.007 (cell-count baseline), Pearson r = 0.79 / 0.70 across endpoints — limited by small-sample noise: for CellProfiler features and endpoints with ≥ 30 actives r = 0.91 / 0.85. Our default protocol (shuffled folds, 3 repeats averaged) scores the full model ≈ +0.04 higher, i.e. the offset is a cross-validation-protocol effect, not a data-processing one.

![Agreement with the paper (default protocol, CellProfiler)](../results/figures/paper_agreement.png)

### 4.2 Most endpoints cannot support a claim, and naive wins are mostly not certifiable
![Correction funnel](../results/figures/correction_funnel.png)

| step (CellProfiler, `allpod`) | endpoints |
|---|---|
| tested (excl. positive control) | 404 |
| ΔAUROC > 0 against the strong cell-count baseline — the naive "morphology wins" | 292 |
| … of which powered (≥ 15 actives and ≥ 15 non-hits); **223 (55 %) are indeterminate** | 160 |
| … and raw bootstrap p < 0.05, uncorrected | 95 |
| … and BH q < 0.05 on raw bootstrap p-values (anti-conservative, §4.4) | 64 |
| … and **BH q < 0.05 on null-calibrated p-values → `morphology_advantage`** | **15** (of 181 powered) |

For indeterminate endpoints the reported AUROCs are essentially noise: the full-model AUROC has a standard deviation of 0.175 across indeterminate endpoints versus 0.117 across powered ones, and 56 of 223 lie outside [0.4, 0.8] (40 below 0.4). Reporting them as clean AUROCs would manufacture spurious wins and losses (figure below: indeterminate endpoints in yellow scatter from 0.02 to 1.0).

![Full vs cell-count-only AUROC per endpoint](../results/figures/scatter_strong.png)

Yet effect sizes among the powered endpoints are broadly positive — ΔAUROC > 0 for 160 of 181 (88 %), median Δ = +0.094 (median AUROC 0.650 full vs 0.541 strong cell-count baseline) — so the limit is per-endpoint statistical certainty, not an absence of signal: with 15–60 actives per endpoint only large effects can be certified individually. **The certified count is a band, not a point:** it is 67, 38, 23, 15 and 5 endpoints if the null sd is taken as 1.0, 1.2, 1.33, 1.45 and 1.65 (our matched estimate is 1.37), so it must not be read as "how many endpoints benefit".

### 4.3 The certified endpoints overlap strongly across representations
| representation (`allpod`) | null sd₀ (powered null runs) | certified vs strong cc | vs scalar cc | raw-bootstrap counts (strong / scalar) |
|---|---|---|---|---|
| CellProfiler (primary) | 1.37 (379) | **15** | 51 | 64 / 75 |
| CP-CNN | 1.54 (274) | 15 | 29 | 60 / 84 |
| DINOv2 | 1.27 (58; noisy) | 54 | 59 | 92 / 85 |

Every CellProfiler-certified endpoint and every CP-CNN-certified endpoint is also certified with DINOv2; **8 endpoints are certified under all three representations** (`MT`, the PR-bla and GR-bla antagonist assays, two BioMAP assays — a proliferation and an E-selectin readout — and three cytotoxicity endpoints: HEK293, ME-180, ERR-HEK293T); 22 are certified under at least two and 54 under any. ΔAUROC is rank-correlated across representations over the 181 powered endpoints (Spearman 0.70 CellProfiler vs CP-CNN, 0.66 vs DINOv2). The *number* certified depends strongly on the representation (15 / 15 / 54), and DINOv2's null rests on only 58 runs, so we report all three rather than a single headline.

The 15 endpoints certified with CellProfiler features:

| endpoint | category | actives | AUROC strong cc → full | Δ | q CellProfiler | q CP-CNN | q DINOv2 |
|---|---|---|---|---|---|---|---|
| `TOX21_PR_BLA_Antagonist_ratio` | cellbased | 145 | 0.53 → 0.72 | +0.184 | 0.002 | 0.001 | 0.000 |
| `cell_type__HEK293` | cytotox | 186 | 0.71 → 0.82 | +0.110 | 0.002 | 0.007 | 0.001 |
| `tissue__kidney` | cytotox | 130 | 0.77 → 0.88 | +0.105 | 0.002 | 0.054 | 0.000 |
| `cell_type__ME-180` | cytotox | 101 | 0.68 → 0.83 | +0.145 | 0.002 | 0.006 | 0.000 |
| `TOX21_PXR_agonist` | cellbased | 124 | 0.53 → 0.69 | +0.160 | 0.005 | 0.153 | 0.005 |
| `BSK_CASM3C_Proliferation` | cellbased | 34 | 0.33 → 0.62 | +0.290 | 0.014 | 0.007 | 0.005 |
| `cell_type__ERR-HEK293T` | cytotox | 131 | 0.72 → 0.83 | +0.116 | 0.014 | 0.024 | 0.000 |
| `tissue__intestinal` | cytotox | 100 | 0.71 → 0.82 | +0.110 | 0.014 | 0.065 | 0.008 |
| `BSK_SAg_Eselectin` | cellbased | 30 | 0.50 → 0.74 | +0.241 | 0.015 | 0.007 | 0.000 |
| `BSK_SAg_Proliferation` | cellbased | 56 | 0.58 → 0.79 | +0.204 | 0.015 | 0.120 | 0.001 |
| `MT` | native | 377 | 0.87 → 0.92 | +0.042 | 0.019 | 0.039 | 0.001 |
| `BSK_3C_HLADR` | cellbased | 39 | 0.47 → 0.72 | +0.246 | 0.019 | 0.301 | 0.010 |
| `cell_type__HepG2` | cytotox | 150 | 0.77 → 0.86 | +0.088 | 0.021 | 0.065 | 0.002 |
| `tissue__liver` | cytotox | 150 | 0.77 → 0.86 | +0.088 | 0.021 | 0.065 | 0.002 |
| `TOX21_GR_BLA_Antagonist_ratio` | cellbased | 53 | 0.53 → 0.74 | +0.215 | 0.029 | 0.021 | 0.000 |

![ΔAUROC with bootstrap intervals; green = certified after calibration](../results/figures/delta_forest.png)

### 4.4 Auditing the audit: raw bootstrap p-values are anti-conservative
![Null control](../results/figures/null_control.png)

Permuted-label runs use exactly the audit's settings (3 repeats, 1,000 resamples; an earlier null with fewer repeats and resamples had estimated sd 1.45 and produced a count of 9, which is why the count is now larger). In 379 powered CellProfiler runs (mean AUROC 0.49 full, 0.50 strong baseline, as they should) the z-score has sd 1.37 where a valid test would give 1.0, so the raw p-value rejects at nominal 5 % in **10.3 %** of runs. Calibrating with an sd estimated from the first 190 runs (1.45) brings the error rate on the other 189 runs to **2.6 %** (nominal 5 %) and 0.5 % (nominal 1 %) against 7.9 % raw: conservative. For CP-CNN the inflation is larger (sd 1.54 from 274 runs; raw 17.9 %); the split-half calibrated rate is 6.5 % (nominal 5 %; 139 held-out runs, within sampling noise) and 0 % at nominal 1 %. For DINOv2 the null has only 58 powered runs (sd 1.27, raw 11.7 %); no split-half check is possible, and the uncalibrated pipeline made 2 BH calls in 60 permuted runs (0 for CellProfiler and CP-CNN). The null sd depends mildly on endpoint size for CellProfiler (1.44 / 1.35 / 1.21 for 15–29 / 30–59 / ≥ 60 actives); we use one pooled value per representation.

### 4.5 Weak baselines hide shortcuts
Against the paper's scalar cell-count baseline 51 endpoints are certified; 38 of them lose the advantage against the stronger cell-count baseline (2 are certified only against the strong baseline, where the many-feature model over-fits at small n). The strong baseline matters most for the native readouts:

| readout | scalar cc | strong cc | full (CellProfiler) | Δ vs strong | BH q (CP / CP-CNN / DINO) | verdict vs scalar |
|---|---|---|---|---|---|---|
| MT (metabolic activity) | 0.835 | 0.873 | 0.916 | +0.042 | 0.019 / 0.039 / 0.001 | advantage (q < 0.001) |
| LDH (membrane damage) | 0.936 | 0.957 | 0.969 | +0.011 | 0.27 / 0.29 / 0.076 | advantage (q = 0.016) |
| cell_count (positive control) | 0.975 | 1.000 | 0.991 | — | — | control passes |

Morphology beats cell count for MT with all three representations but is **never** certified for LDH at the 5 % level (DINOv2 is borderline, q = 0.076) — in agreement with the paper — while the one-number baseline would have called LDH a win. The strong baseline reaching AUROC 1.000 on the cell-count positive control confirms that it captures everything cell count can give.

### 4.6 Where morphology wins
| category | powered | certified (CellProfiler / CP-CNN / DINOv2) |
|---|---|---|
| cell-based reporter endpoints | 147 / 292 | 7 / 10 / 36 |
| cytotoxicity endpoints (cell-type / tissue summaries, 1 = cytotoxic) | 24 / 38 | 7 / 4 / 17 |
| cell-free (biochemical) endpoints | 8 / 72 | 0 / 0 / 0 |
| native (MT, LDH) | 2 | 1 / 1 / 1 |

89 % of cell-free endpoints cannot be adjudicated and none of the 8 that can shows an advantage — consistent with the paper's finding that Cell Painting does not predict purely biochemical activity.

![Verdicts by endpoint category](../results/figures/verdicts_by_category.png)

![Enrichment](../results/figures/enrichment_family.png)

**Enrichment — descriptive only.** One-sided Fisher tests (BH within each grouping). Unadjusted, cytotoxicity endpoints look over-represented among certified endpoints with CellProfiler (7/24 vs 5 % elsewhere, odds ratio 7.7, q = 0.003) and DINOv2 (17/24 vs 24 %, OR 7.9, q < 0.001) but not with CP-CNN (4/24, q = 0.36). They are also far larger (median 97.5 actives vs 25 for cell-based endpoints), so they are simply easier to certify: restricted to endpoints with at least 50 actives the difference is not significant (CellProfiler cytotoxicity 7/22 vs cell-based 4/20, OR 1.87, p = 0.49; CP-CNN 4/22 vs 4/20, OR 0.89, p = 1.0; DINOv2 17/22 vs 10/20, OR 3.40, p = 0.11, undecided). We therefore report it as a pattern confounded with endpoint size, not as a finding. No other target family, cell line or tissue is enriched in more than one representation (DINOv2 flags cell-adhesion-molecule assays, 5/6, q = 0.043; CP-CNN flags antagonist-mode assays, q = 0.073; neither replicates). The "sub-lethal stress" interpretation rests on the compound-level analysis of §4.8, not on this enrichment.

### 4.7 Not a technical shortcut
Plate number, mean well row/column, batch and replicate counts alone predict ToxCast labels at chance (mean AUROC 0.512; 11.6 % of endpoints above 0.6; `scripts/09_technical_probe.py`); adding them to the strong cell-count model changes nothing (0.563 vs 0.555), and for none of the 15 certified endpoints does "technical + cell count" match the full profile.

### 4.8 Compound level: where morphology adds information
Per compound, the *benefit* is how much better the full model ranks the compound (against the compounds of the opposite class) than the cell-count baseline does, averaged over the powered endpoints in which the compound is active. Compounds are grouped by the paper's own hit pattern across readouts (a mechanism-free grouping):

| compound group | compounds | mean benefit [95 % CI] | benefit for all other compounds |
|---|---|---|---|
| active in Cell Painting only (no cell loss, MT or LDH hit) | 133 | **+0.149** [0.103, 0.188] | +0.045 |
| MT or LDH hit without cell loss | 186 | **+0.128** [0.098, 0.162] | +0.041 |
| cell loss (cell-count hit) | 195 | **+0.010** [0.001, 0.019] | +0.087 |

Where cells die, counting them already ranks the compounds almost as well as morphology; where cells stay alive but change, morphology adds the information. (BH q < 0.001 for each group against the rest; the "inactive" group is not interpretable for active-compound benefit.) This is consistent with morphology reading sub-lethal stress before cells are lost, but it is an association within one dataset and has not yet been controlled for the baseline's own score range.

The `signal_class` labels split "no advantage" into *why*: of 181 powered endpoints, **30 are count-explained** (the cell-count baseline is already informative), **57 show morphology signal that cannot be certified as an advantage**, **79 show no detectable signal**, and 15 are certified. Gain importance by compartment, feature type and channel is nearly identical for certified and other endpoints (granularity 29.7 % of gain vs 24.5 % of features for certified, 30.2 % otherwise), so no single channel or compartment explains where morphology wins.

### 4.9 The cytotoxicity filter and the shortcut
A ToxCast 0 is a "non-hit" (§3.3b), and the cytotoxicity filter that sets some hits to 0 applies only where a consensus cytotoxicity AC50 exists. Re-scoring the saved predictions by whether the filter could apply (`scripts/19_filter_stratified.py`, 48 cell-based endpoints scored in both strata): the filter could apply to 22.6 % of compound–endpoint pairs; the strong cell-count baseline is near chance in both strata (median AUROC 0.534 where it applies, 0.518 where it cannot), whereas the full profile is stronger where it cannot apply (0.649 vs 0.548) so morphology's median advantage is larger there (+0.118 vs +0.035). The strata are defined by compound properties (a matched cytotoxicity AC50 exists mainly for cytotoxic compounds), so this is descriptive, not causal.

### 4.10 Robustness
* **Power threshold** (min actives/non-hits per class 10 / 15 / 20 / 30): 19 / **15** / 22 / 26 certified endpoints out of 242 / 181 / 139 / 75 powered (a stricter threshold shrinks the BH pool and tends to raise the count; 15/15 was fixed in advance and not tuned). **α = 0.10**: 36.
* **Chance-floored baseline** (Δ = full − max(baseline, 0.5), so a below-chance baseline cannot inflate Δ): 13 of the 15 are retained.
* **Null sd** (§4.2): 67 / 38 / 23 / 15 / 5 certified at sd 1.0 / 1.2 / 1.33 / 1.45 / 1.65.
* **Aggregation rule** (CellProfiler, same settings and the same null): certified endpoints vs the strong baseline `all` **18**, `allpod` **15**, `allpodcc` **3** (vs scalar: 56 / 51 / 60). `allpodcc` averages only wells below the cell-count point of departure, which by construction leaves little for any model to separate from the full cell-count curve, and MT is no longer certified there (q = 0.90); `all` certifies MT (q = 0.004). The aggregation dependence of the count is a limitation; `allpod` is the paper's default and our pre-specified primary.
* **Chemical-similarity folds** (Butina clusters of ECFP4 similarity ≥ 0.4 as CV groups, 921 clusters; `--group-by chem_cluster`; own permutation null, 88 powered runs, sd 1.44): median full-model AUROC 0.650 → 0.651 and median ΔAUROC 0.094 → 0.091 over the 181 powered endpoints (Spearman 0.83 between the two sets of Δ); 24 endpoints are certified and **all 15 of the primary set are among them**. Structural analogues in different folds therefore do not explain the result.
* **Representation**: §4.3.

### 4.11 Calibration
Raw probabilities are not trustworthy as confidences: for every one of the 181 powered endpoints the calibration slope is below 0.8 (median 0.33, i.e. strongly over-confident, a consequence of class re-weighting), the median expected calibration error is 0.081, and only 32 % of powered endpoints have a Brier score better than the prevalence predictor (median Brier skill −0.07). Cross-validated Platt recalibration lowers the median Brier score from 0.112 to 0.104. The demo therefore shows recalibrated probabilities together with each endpoint's calibration diagnostics.

![Reliability diagram, all powered endpoints pooled](../results/figures/pooled_reliability.png)

## 5. Reliability analysis and limitations

What we did to make the audit itself trustworthy: label-permutation null with split-half validation (§4.4); a positive control that the strong baseline must solve (§4.5); a chance-floored baseline (§4.10); a technical-confound probe (§4.7); calibration diagnostics (§4.11); reproduction of the source paper's numbers under its own protocol (§4.1); a pre-specified power threshold with a sensitivity grid (§4.10); replication across three representations, aggregation rules and chemical-similarity folds (§4.3, §4.10); 50 unit tests covering preprocessing, grouped CV, bootstrap, BH, calibration, enrichment and the null calibration (`python -m pytest`).

Limitations, stated plainly:

* **Per-endpoint power is low.** With 15–60 actives per endpoint only large effects are individually certifiable; 55 % of endpoints are indeterminate, and the certified count (15 / 15 / 54 by representation; 5 to 67 for CellProfiler across plausible null sds) is a band limited by power, not an estimate of how many endpoints benefit.
* **Empirical-null calibration is approximate.** sd₀ is estimated from permutation runs at the audit's own settings (CellProfiler 379 powered runs, CP-CNN 274, DINOv2 only 58, so the DINOv2 calibration is the weakest and has no split-half check); it varies mildly with endpoint size (CellProfiler 1.44 / 1.35 / 1.21 for small / medium / large endpoints) and we use one pooled value per representation. The CP-CNN split-half error rate (6.5 % at nominal 5 %) shows the calibration is not exactly conservative everywhere. A per-endpoint permutation test would be cleaner but costs ~10⁵ model fits per configuration.
* **Dependence in BH.** Endpoints share compounds and are correlated; BH is valid under positive dependence but this is not verified here.
* **"Morphology" is not orthogonal to cell density.** CellProfiler (neighbour counts, granularity, image-level features) and especially embedding features can encode confluence and clumping; the strong baseline sees only cell-count summaries. A "morphology advantage" therefore means *information beyond the cell-count curve*, which may include other density-related image properties.
* **Aggregation dependence.** The strong-baseline count varies with the aggregation rule (§4.10); `allpod` is the paper's default and our pre-specified primary.
* **Baseline strength has no ceiling.** A still stronger cell-count model (more features, tuning) might certify fewer endpoints; ours is a deliberate best-effort shortcut, not an upper bound.
* **What a 0 means.** In the ToxCast matrices a 0 is *not* "tested and negative": it is no hit, a hit overridden by the cytotoxicity filter, or a tie. The filter (hit set to 0 if the endpoint AC50 exceeds half the matched consensus cytotoxicity AC50) applies only where such a consensus exists (≥ 20 % of matched viability tests fired, ≤ 100 µM; about 35 % of cell-based records), so many cell-based labels are not cytotoxicity-adjusted. Untested pairs stay missing (61.9 % of the cell-based matrix; pinned by a regression test). We say "non-hit" for the 0-class; the `n_inactive` column keeps its name. A stratified, descriptive check is in §4.9.
* **Irregular design.** Concentration series differ by source batch (952 compounds on 0.0456–100 µM, 14 on a 100× lower series, one with 7 points) and replicate counts vary (829 single-well compound-concentration cells); dose-response baseline features therefore include concentration-interpolated cell counts and the technical probe includes well counts.
* **Label quality.** ToxCast hit-calls come from heterogeneous assays and cell systems; the 38 cytotoxicity columns are cell-type/tissue summaries where 1 = *cytotoxic* (the paper's 48 is before a minimum-5-positive / 5-negative filter); actives are rare for many endpoints; structurally related compounds in different folds could inflate all models equally.
* **Scope.** Single donor/pool of primary hepatocytes (donor variability not modeled), one 44 h time point, 2D plate assay — not an organ-on-chip; compound-level modeling means plate/well covariates are probed separately rather than included.

## 6. Motivation and impact

*Regulatory context.* FDA's March 2026 draft guidance "General Considerations for the Use of New Approach Methodologies in Drug Development" (Federal Register 2026-05390, 19 March 2026) names four factors for validating a NAM — context of use, human biological relevance, **technical characterization** and **fit for purpose**. An audit that tells a developer whether a profiling model's predictive power is a real signal or a trivial confound, endpoint by endpoint and with explicit uncertainty, is a concrete instance of technical characterization; indeterminate verdicts and calibration diagnostics speak to fit for purpose.

*Scientific context.* The Cell Painting community has documented that cell count alone can predict many small-molecule bioactivity benchmark labels (Seal et al., *Nat. Commun.* 2026), and the JUMP-CP resource (Chandrasekaran et al., *Nat. Methods* 2024) is the large public dataset where such baselines matter. Ewald et al. (2026) apply that insight to hepatocyte cytotoxicity; we turn it into a reusable, statistically controlled procedure and show, with a permutation control, which of its most natural implementations (plain bootstrap p-values) would have over-called.

*Organ-on-chip relevance.* Chip assays will produce rich image-based readouts on small numbers of compounds — exactly the regime where multiplicity, low power and trivial survival confounds bite. The audit is dataset-agnostic: it needs profiles, a cell-count-like column and binary endpoints.

## 7. Reproduction instructions

See `README.md`. Entry points: `scripts/01_download_zenodo_data.py`, `02_fetch_labels.py`, `03_run_audit.py`, `08_null_control.py`, `15_calibrate_verdicts.py`, `04_run_enrichment.py`, `07_finalize_results.py`, `12_headline_numbers.py`, `13_sensitivity_postprocess.py`; demo `streamlit run demo/app.py`. Everything is CPU-only with fixed seeds and MD5-verified downloads. Measured cost on a 10-core laptop (uncontended): CP-CNN full audit of all 404 endpoints ≈ 14 min, CellProfiler ≈ 29 min; 200 permutation runs ≈ 10–25 min. Unit tests: `python -m pytest`.

## 8. References

1. Ewald, J.D., Titterton, K.L., Bäuerle, A., et al. Cell Painting for cytotoxicity and mode-of-action analysis in primary human hepatocytes. *Cell Systems* 17(5), 101566 (2026). doi:10.1016/j.cels.2026.101566
2. Zenodo record 10.5281/zenodo.17067683 — Axiom OASIS profiles and metadata (CC-BY 4.0).
3. Authors' analysis repository, github.com/jessica-ewald/2024_09_09_Axiom_OASIS (BSD-3-Clause).
4. Seal, S., Dee, W., Shah, A., et al. Counting cells can accurately predict small-molecule bioactivity benchmarks. *Nat. Commun.* 17, 2436 (2026). doi:10.1038/s41467-026-68725-5
5. Chandrasekaran, S.N., Cimini, B.A., Goodale, A., et al. Three million images and morphological profiles of cells treated with matched chemical and genetic perturbations. *Nat. Methods* 21, 1114–1121 (2024). doi:10.1038/s41592-024-02241-6
6. U.S. FDA. General Considerations for the Use of New Approach Methodologies in Drug Development; Draft Guidance for Industry; Availability. *Federal Register*, 19 March 2026, document 2026-05390.
7. Benjamini, Y., Hochberg, Y. Controlling the false discovery rate: a practical and powerful approach to multiple testing. *J. R. Stat. Soc. B* 57, 289–300 (1995).
8. Efron, B. Large-scale simultaneous hypothesis testing: the choice of a null hypothesis. *J. Am. Stat. Assoc.* 99, 96–104 (2004).
9. Chen, T., Guestrin, C. XGBoost: a scalable tree boosting system. *KDD* (2016).
10. Cell Painting Gallery (cpg0037-oasis), AWS Registry of Open Data, CC0 — used only for a dozen demo images.

## Appendix A. Output schema (`results/audit_table.csv`)
One row per endpoint: `endpoint_id`, annotations (`category`, `assay_target_family`, `assay_design_type`, `cell_short_name`, `tissue`, `endpoint_description`), `n_compounds`, `n_active`, `n_inactive`, `powered`, `scalar_cc_AUROC/PRAUC`, `strong_cc_AUROC/PRAUC`, `full_AUROC/PRAUC`, `delta_AUROC` (+ `delta_ci_lo/hi`, `delta_z`), `bootstrap_p` (raw), `calibrated_p`, `fdr_q`, `verdict`, `verdict_uncalibrated`, the `_scalar` counterparts versus the one-number baseline, and `calibration_brier`, `brier_skill`, `ece`, `calib_slope`, `brier_recalibrated`.

## Appendix B. All configurations
`results/sensitivity_summary.csv`. `cellprofiler_allpod`, `cpcnn_allpod` and `dino_allpod` audit every endpoint; the other configurations (CellProfiler `all`, `allpodcc`, and chemical-cluster folds `_chem`) were run on powered endpoints only. Each representation uses its own matched-settings permutation null; CellProfiler aggregation variants use the CellProfiler `allpod` null (assumed shared), and the chemical-cluster run its own.

| configuration | endpoints modeled | powered | certified vs strong cc | certified vs scalar cc | raw-bootstrap (strong / scalar) | null sd₀ used | median AUROC full / strong / scalar (powered) |
|---|---|---|---|---|---|---|---|
| `cellprofiler_all` | 181 | 181 | 18 | 56 | 52 / 81 | 1.374 | 0.638 / 0.543 / 0.509 |
| `cellprofiler_allpod` | 404 | 181 | 15 | 51 | 64 / 75 | 1.374 | 0.650 / 0.541 / 0.540 |
| `cellprofiler_allpod_chem` | 181 | 181 | 24 | 51 | 56 / 71 | 1.442 | 0.651 / 0.531 / 0.540 |
| `cellprofiler_allpodcc` | 181 | 181 | 3 | 60 | 44 / 88 | 1.374 | 0.657 / 0.539 / 0.521 |
| `cpcnn_allpod` | 404 | 181 | 15 | 29 | 60 / 84 | 1.54 | 0.654 / 0.547 / 0.534 |
| `dino_allpod` | 404 | 181 | 54 | 59 | 92 / 85 | 1.273 | 0.680 / 0.544 / 0.557 |

## Appendix C. Power-threshold and α grid (CellProfiler `allpod`)
`results/cellprofiler_allpod/sensitivity_postprocess.csv`.

| min actives & non-hits | α | baseline | p-values | powered | certified |
|---|---|---|---|---|---|
| 10 | 0.05 | strong_cc | null-calibrated | 242 | 19 |
| 10 | 0.05 | scalar_cc | null-calibrated | 242 | 39 |
| 10 | 0.1 | strong_cc | null-calibrated | 242 | 38 |
| 10 | 0.1 | scalar_cc | null-calibrated | 242 | 76 |
| 15 | 0.05 | strong_cc | null-calibrated | 181 | 15 |
| 15 | 0.05 | scalar_cc | null-calibrated | 181 | 51 |
| 15 | 0.05 | strong_cc | raw bootstrap | 181 | 64 |
| 15 | 0.05 | scalar_cc | raw bootstrap | 181 | 75 |
| 15 | 0.1 | strong_cc | null-calibrated | 181 | 36 |
| 15 | 0.1 | scalar_cc | null-calibrated | 181 | 69 |
| 15 | 0.1 | strong_cc | raw bootstrap | 181 | 95 |
| 15 | 0.1 | scalar_cc | raw bootstrap | 181 | 94 |
| 20 | 0.05 | strong_cc | null-calibrated | 139 | 22 |
| 20 | 0.05 | scalar_cc | null-calibrated | 139 | 50 |
| 20 | 0.1 | strong_cc | null-calibrated | 139 | 34 |
| 20 | 0.1 | scalar_cc | null-calibrated | 139 | 64 |
| 30 | 0.05 | strong_cc | null-calibrated | 75 | 26 |
| 30 | 0.05 | scalar_cc | null-calibrated | 75 | 35 |
| 30 | 0.1 | strong_cc | null-calibrated | 75 | 39 |
| 30 | 0.1 | scalar_cc | null-calibrated | 75 | 44 |
