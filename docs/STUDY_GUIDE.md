# Study guide: Beyond the Count (CPSA)

A complete walkthrough of the project for the final-round presentation and defense: the idea, the data, every step of the pipeline with its formula and its code, every number, every design decision, every weak point and how to answer it, and a bank of likely judge questions.

How to use it: read Parts 0 to 2 once for the story, study Parts 3 and 4 until you can draw the pipeline and quote the headline numbers from memory, read Part 6 (limitations) until the weak points feel like strengths because you name them first, then use Part 7 (Q&A) and Part 10 (self-quiz) to rehearse. Numbers come from `results/` and `report/technical_report.md`; where a statement is not verified it is marked **[unverified]**.

Contents

0. The stories to tell (60 seconds, 5 minutes)
1. Background from first principles
2. The data in detail
3. The pipeline, step by step (with formulas and code)
4. Results cheat sheet
5. Why each design choice (the decision log)
6. Limitations and honest answers
7. Likely judge questions with model answers
8. Things never to say
9. Formulas and glossary
10. Code map and how to run things
11. Self-quiz

---

## 0. The stories to tell

### The 60-second version

Cell Painting turns cells into thousands of image features and is becoming a core, animal-free readout for toxicology. But many toxicity labels are tied to cytotoxicity, so a model that only counts how many cells survived can look as good as one that reads morphology. We built an audit, not a better classifier: for every endpoint it asks whether the full morphology profile beats a strong cell-count baseline, with calibrated statistics, multiple-testing control and an explicit "not enough data" verdict. On the public hepatocyte data, 181 of 405 endpoints can be tested and 15 are certified, including metabolic activity in all three image representations. A plain bootstrap would have claimed 64, which is why we audited our own audit with a permutation test. Because organ-on-chip studies will adopt the same AI readouts at much smaller sample sizes, we ship a planner that shows what the audit can conclude at 20 to 200 compounds.

### The 5-minute version (matches the video script)

1. **Problem (0:45).** Cell Painting; the cell-count shortcut; Seal et al.; FDA's "technical characterization"; organ-on-chip inherits the risk.
2. **Method (1:00).** Matched models on identical compound-grouped folds; strong baseline; paired compound bootstrap; permutation-calibrated p-values; power check; BH-FDR; four verdicts.
3. **Results (1:30).** The funnel 292, 160, 95, 64, 15; MT certified in all three representations, LDH never; nested test 23; compound-level story (sub-lethal change).
4. **Reliability (0:45).** Permutation control 10.3% to 2.6%; plate/well/batch probe at chance; simulation with known truth; chemical folds; limitations named first.
5. **Organ-on-chip (0:45).** Simulation at 20 to 200 compounds; planner; replicate-chip leakage; what must change for a chip.
6. **Close (0:15).** Dataset-agnostic reusable audit; live demo; reproducible on a laptop CPU.

The one sentence to keep returning to: **the contribution is the audit, not a classifier.**

---

## 1. Background from first principles

### Cell Painting

- Cells in a multi-well plate are stained with six fluorescent dyes imaged in five channels: **DNA** (Hoechst), **ER** (concanavalin A), **RNA** (SYTO 14), **AGP** (actin, Golgi and plasma membrane: phalloidin plus wheat germ agglutinin) and **mitochondria** (MitoTracker).
- Software (CellProfiler) segments cells and measures thousands of features per cell: size and shape, intensity, texture, granularity, spatial correlations between channels, neighbour counts. Cells are summarised per well, and wells per compound. The result is a "morphological fingerprint" per compound.
- Why toxicologists like it: one cheap, unbiased assay instead of one pre-chosen readout per mechanism.

### The labels

- **ToxCast / Tox21** (EPA and NIH) run hundreds of in vitro assays; a **hit-call** says whether a compound was active in an assay endpoint. Here the labels come from invitrodb v4.1, processed by the dataset authors.
- An **endpoint** is one binary outcome column: a ToxCast assay endpoint, a cytotoxicity summary per cell type or tissue, or one of the paper's own readouts (MT, LDH, cell count).
- **MT** is metabolic activity (an early stress readout); **LDH** is lactate dehydrogenase release (membrane damage, later death); **cell count** is the number of cells left.

### The shortcut

- A compound that kills or stresses cells changes almost every feature at once, including the crudest one: how many cells are left. Many toxicity labels are defined from, or correlated with, cytotoxicity, so **a model that only knows the cell count can predict them**.
- Seal et al. (Nature Communications 2026; preprint text checked): in the Hofmarcher benchmark a cell-count-only baseline (mean AUC 0.68 ± 0.21) matched a fully connected network on full Cell Painting profiles (0.67 ± 0.19); in the Moshkov data 31 of 49 (63%) endpoints well predicted by gene expression were predicted equally well by cell count.
- Consequence: a high AUROC from a "morphology model" does not show the model reads morphology. The right question is **what the profile adds beyond cell count**, endpoint by endpoint.

### Why it matters

- **Regulators:** FDA's March 2026 draft guidance on New Approach Methodologies names four validation factors: context of use, human biological relevance, technical characterization, fit for purpose. A shortcut audit is a concrete piece of technical characterization. (The exact quotations in the team draft were not verified against the guidance text; the four factors are confirmed by secondary sources.)
- **Organ-on-chip:** chips adopt AI image readouts too, with far fewer compounds, and chip structure and flow can add density-like confounds. The audit is dataset-agnostic so a chip team can run it before trusting a readout.

### AUROC in one paragraph

AUROC is the probability that a randomly chosen active compound gets a higher score than a randomly chosen non-hit. 0.5 is chance, 1.0 is perfect. It is rank-based, so it ignores calibration and is insensitive to class imbalance. The effect size in this project is **ΔAUROC = AUROC(full morphology) minus AUROC(baseline)**, computed on pooled out-of-fold predictions.

---

## 2. The data in detail

### OASIS primary human hepatocyte data (Ewald et al., Cell Systems 2026)

- Source: Zenodo 10.5281/zenodo.17067683 (CC-BY 4.0): well-level profiles in three image representations (**CellProfiler** hand-engineered features, **CP-CNN** from a Cell-Painting-specific network, **DINOv2** self-supervised embeddings) plus metadata and an image index.
- Scale: **21,456 wells on 65 plates** (4,849 DMSO controls), one 44-hour time point, primary human hepatocytes in 2D plates. **967 compounds** carry an OASIS id and are profiled; the rest are blinded by the authors (usable only for native readouts). The paper describes 1,085 compounds (1,086 names appear in the metadata).
- Dosing: 8 concentrations per compound; 952 compounds share one series (0.0456 to 100 µM), 14 use a 100-fold lower series, one has 7 points. 829 compound-concentration cells have a single well. This is why the dose-response baseline uses concentration-aware features.
- Each compound occupies wells on about 13 to 16 plates from two production batches, so plate and well are not defined at compound level (they are probed separately).

### Labels and their meaning

- Files used: `toxcast_{cellbased,cellfree,cytotox}_binary.parquet` plus the paper's hit summary (MT, LDH, cell count). **Only the `*_binary` files are valid ground truth.** The `*_info` files contain a raw continuous `hitcall`, which is never used as a label.
- Hit rule (authors): hitcall > 0.9, then set to 0 if the endpoint AC50 exceeds **half the matched consensus cytotoxicity AC50** for that cell type (tissue as fallback). A consensus exists only if at least 20% of matched viability tests fired and the median AC50 is at most 100 µM. Duplicates are collapsed by majority vote; ties count as no hit.
- Therefore **a 0 means "not a (filtered) hit"**, which can be no hit, a hit overridden by the filter, or a tie. It never means "tested negative".
- **Untested pairs are NaN and stay out of that endpoint's compound set.** On the cell-based matrix only 107,103 of 281,196 compound-endpoint cells (963 × 292) are observed, about 62% missing. Zero-filling would silently inflate every endpoint's non-hit count and break the AUROC, power check and FDR downstream.
- The cytotoxicity filter can apply to only about 35% of raw cell-based records (22.6% of compound-endpoint pairs among the scored endpoints), so many labels are **not** cytotoxicity-adjusted, and a cell-count shortcut can reach them directly.
- Endpoint inventory: **292 cell-based, 72 cell-free, 38 cytotoxicity** (each retained with at least 5 positives and 5 negatives; 38 of 48 source cytotoxicity categories survive) plus **MT, LDH, cell_count**: **405 endpoints**. `cell_count` is the positive control (its label is defined from cell count).
- Coverage: ToxCast labels cover 670 profiled compounds for cytotoxicity and cell-based endpoints; cell-free coverage is 242 profiled compounds in our count and 251 in the label files (unresolved). A 89% figure that is sometimes quoted is a union of chemical identifiers, not the intersection with compounds actually tested; we use the intersection.
- Known deviation of the released labels from the published Methods: the 100 µM ceiling is applied to the consensus cytotoxicity AC50 only, not per endpoint. Applying it as described would change the cell-based positive count from 8,400 to about 8,332 and drop one endpoint. We used the released labels as they are (re-deriving ground truth was out of scope). A unit test pins 107,103 observed and 8,400 positives.

### The second source (EU-OPENSCREEN)

- Zenodo 10.5281/zenodo.19347244 (CC-BY 4.0): HepG2 profiles of about 2,500 bioactives at 10 µM, imaged at four sites with four replicates; 327 compounds overlap our profiled compounds (InChIKey skeleton match), 49 endpoints clear 15/15 on that overlap. Used as an exploratory second morphology source (Part 4).

---

## 3. The pipeline, step by step

Package layout: `src/cpsa/` (data, models, stats, biology, viz, plus `audit.py`, `api.py`, `nested.py`, `chip.py`, `simulate.py`); numbered scripts in `scripts/` wrap each stage; 69 tests in `tests/`.

### 3.1 Normalisation and feature selection (`cpsa.data.preprocess`)

1. For each plate, take its DMSO control wells; compute per-feature median and MAD (median absolute deviation, unscaled).
2. Normalise every well: **x' = (x − median_DMSO,plate) / MAD_DMSO,plate**. This removes plate effects and puts features on a robust z-score scale.
3. Drop a feature if it is non-finite anywhere, or if in any plate its DMSO MAD is 0 or |MAD/median| ≤ 10⁻³ (the authors' "variant feature" rule).
4. **Correlation filter:** scan features in order, drop any with |Pearson r| > 0.9 against an already kept feature (greedy; result depends on column order, which is fixed).

### 3.2 Per-compound aggregation (the unit of analysis)

Labels are per compound, so each compound becomes **one row**: the mean of its selected wells' normalised features, plus `Cell_Count` (mean well cell count). Three rules (mirroring the authors' code):

- `all`: every treated well.
- `allpod` (**primary, pre-specified, the paper's default**): wells above the compound's Cell Painting **point of departure (POD)** (the lowest concentration with a detectable profile change). Compounds with no well above the POD fall back to all wells. This keeps low, inactive concentrations from diluting the profile.
- `allpodcc`: wells above the Cell Painting POD and below the cell-count POD (the window where morphology changes before cells are lost), with two fallbacks (first concentration above the POD; then all wells).

### 3.3 The baselines (`cpsa.data.build_dataset`, `cell_count_curve_features`)

- **scalar_cc:** `Cell_Count` alone, the paper's baseline (one number).
- **strong_cc:** `Cell_Count`, `cc_r1..cc_r8` (cell count divided by plate DMSO median at each of the compound's eight concentration ranks), `cc_min`, `cc_auc` (mean ratio), `cc_pod_log10` (cell-count POD) and concentration-aware `cc_at_{0.1,1,10,100}uM` (log-concentration interpolation, NaN outside the tested range because "not measured" is not 0). It sees **only** cell count information, so any advantage over it is information beyond the cell-count curve.
- **full:** all profile features (everything that is not a baseline column).
- No plate, well or batch features are in any baseline (they are probed separately).

### 3.4 Models and cross-validation (`cpsa.models.train_eval`)

- Classifier: **XGBoost**, 150 trees, learning rate 0.05, `tree_method=hist`, `scale_pos_weight = n_negative / n_positive` (the paper's classifier). Hyperparameters are fixed, so no tuning ever sees a test fold.
- CV: **StratifiedGroupKFold**, groups = compound, shuffled, k = min(5, n_pos, n_neg) folds, **3 repeats** with different seeds. Each repeat yields one out-of-fold (OOF) probability per compound; repeats are averaged into one pooled score per compound.
- Because there is one row per compound, "grouped by compound" means no compound is ever in both train and test. A unit test checks that duplicated compounds with noise features score at chance (the leakage test).
- Metrics: rank-based AUROC (ties get average rank), PR-AUC, and the sd of AUROC across repeats.

### 3.5 The effect and its uncertainty: paired compound bootstrap

1. Resample the endpoint's compounds with replacement (n draws) **B = 1000** times, using the **same indices for both models** (paired).
2. In each resample compute Δ* = AUROC(full) − AUROC(baseline) on the fixed OOF predictions; skip resamples with one class only.
3. Report the observed Δ, the 2.5/97.5 percentile interval, and the one-sided p **p = (1 + #{Δ* ≤ 0}) / (B + 1)**.

Key insight to explain: the bootstrap holds the OOF predictions fixed, so it captures **evaluation-sample variability only**, not the variability of the fitted models (training split noise). That is why its p-values are too small (anti-conservative).

### 3.6 Auditing the audit: permutation null and calibrated p-values (`cpsa.stats.null_calibration`)

1. **Permutation control:** pick random powered endpoints, **shuffle their labels across compounds** (true Δ = 0, prevalence kept), and run the identical pipeline (3 repeats, 1,000 resamples). Null AUROCs sit at 0.49 to 0.50, as they should.
2. For each run compute **z = Δ / SE**, where SE is recovered from the bootstrap interval, SE = (CI_hi − CI_lo) / (2 × 1.96).
3. A valid test would give z with sd 1. Measured **sd₀ = 1.37** (379 powered CellProfiler runs): the raw p-value rejects **10.3%** of permuted runs at a nominal 5%.
4. **Efron-style empirical null:** widen the reference distribution: **p_cal = P(Z > z / sd₀)** with Z standard normal. Then BH on p_cal.
5. Validation by split-half: sd₀ estimated on the first 190 runs gave a rate of **2.6%** on the other 189 (nominal 5%; 0.5% at nominal 1%). CP-CNN: sd₀ 1.54, raw 17.9%, split-half calibrated 6.5% (within noise). DINOv2: only 58 powered null runs (sd₀ 1.27), so it has no split-half check, and its calibration is the weakest.

### 3.7 Power check, BH, verdicts (`cpsa.stats.multiple_testing`, `power_check`)

- **Power rule:** an endpoint is *powered* if it has at least **15 actives and 15 non-hits** among profiled compounds (pre-specified; sensitivity in Part 4). Otherwise **`indeterminate`**: no verdict, no q-value, never reported as a win or a loss. 223 of 405 (55%) are indeterminate; their AUROCs are noise (sd 0.175 against 0.117 for powered).
- **Benjamini-Hochberg:** sort the m powered p-values; q_(i) = min over j ≥ i of m · p_(j) / j. The pool is the 181 powered endpoints, excluding the positive control. BH controls the expected fraction of false discoveries under independence or positive dependence.
- **Verdict:** `morphology_advantage` if powered, Δ > 0 and q < 0.05; `no_advantage` otherwise (**read as "no detectable advantage"**: absence of evidence, not equivalence); `indeterminate`; `positive_control`.
- **Signal classes** (descriptive, not tests), splitting `no_advantage` into why: an AUROC is "informative" if it exceeds the 95th percentile of permuted-label AUROCs (full 0.631, strong 0.639). Of 181 powered: **30 count-explained** (strong baseline informative), **57 morphology signal not certifiable**, **79 no detectable signal**, **15 certified**.

### 3.8 Calibration of the model's probabilities (`cpsa.stats.calibration`)

Brier score (and skill against the prevalence predictor), expected calibration error (10 bins), calibration slope (logistic fit of outcome on logit prediction), and cross-validated Platt recalibration. Raw scores are over-confident (class re-weighting): median slope **0.33**, median ECE 0.081, only 32% of powered endpoints beat the prevalence Brier score; Platt recalibration lowers the median Brier from 0.112 to 0.104. The demo shows recalibrated probabilities.

### 3.9 Enrichment (`cpsa.biology.enrichment`)

- One-sided **Fisher exact test**: is a target family, assay design, cell type or tissue over-represented among certified endpoints (against powered non-certified)? Min group size 5, BH within each grouping, families from EPA's `intended_target_family`.
- Endpoints from one assay are not independent, so each group also gets an **assay-level permutation p** (endpoints clustered by the first two tokens of the id, 106 clusters; the same number of clusters drawn at random as the null "in" set; a single-assay group cannot go below about 1 / number of clusters).
- Finding: cytotoxicity endpoints look enriched (CellProfiler 7/24 against 5% elsewhere, Fisher q = 0.003) but they have about four times more actives (median 97.5 against 25), and with at least 50 actives the difference disappears; at assay level p = 0.12 (CellProfiler). Reported as descriptive.

### 3.10 Compound-level benefit (`cpsa.biology.compound_classes`)

- AUROC decomposes as the mean over actives (and over non-hits) of a per-compound ranking accuracy r_i: for an active, the probability its score exceeds a random non-hit's; for a non-hit, the probability its score is below a random active's.
- **Benefit_i = r_i(full) − r_i(baseline)**, averaged over powered endpoints in which compound i is active. Compounds are grouped by the paper's own hit pattern (mechanism-free):

| compound group | compounds | mean benefit [95% CI] |
|---|---|---|
| active in Cell Painting only (no cell loss, MT or LDH hit) | 133 | +0.149 [0.103, 0.188] |
| MT or LDH hit without cell loss | 186 | +0.128 [0.098, 0.162] |
| cell loss (cell-count hit) | 195 | +0.010 [0.001, 0.019] |

Interpretation: where cells die, counting them already ranks compounds almost as well; where cells stay but change, morphology adds information. It is an association within one dataset, consistent with sub-lethal stress being visible before cell loss.

### 3.11 Robustness machinery

- **Technical probe** (`scripts/09`): plate number, mean well row and column, batch, replicate counts alone predict labels at chance (mean AUROC 0.512; adding them to the strong baseline changes nothing, 0.563 against 0.555).
- **Chemical-similarity folds** (`chem_groups.py`): ECFP4 fingerprints, Butina clustering at similarity 0.4 (921 clusters) as CV groups so analogues stay together. Median AUROC 0.650 to 0.651, median Δ 0.094 to 0.091, 24 certified with all 15 primary ones among them.
- **Paper-faithful CV** (`scripts/11`): under the paper's own unshuffled protocol our AUROCs differ from the paper's by +0.004 (full) and +0.007 (baseline) on average, r = 0.79 / 0.70 (0.91 / 0.85 for at least 30 actives). Our default protocol scores about +0.04 higher, a protocol effect, not a data-processing one.
- **Simulation with known truth** (`simulate.py`, `scripts/17`): latent cell-survival factor plus morphology factors; endpoint kinds `null`, `shortcut` (depends on count only), `adds` (count plus morphology), `morph_only`. Calibrated credit: null 3.7% and shortcut 2.2% (raw 8.0% and 5.6%); `adds` 72.5 / 98.3 / 100% and `morph_only` 77 / 100 / 100% for effect 0.5 / 1.0 / 1.5.
- **Sensitivities:** power threshold 10 / 15 / 20 / 30 gives 19 / **15** / 22 / 26 certified; α = 0.10 gives 36; a chance-floored baseline keeps 13 of 15; null sd 1.0 / 1.2 / 1.33 / 1.45 / 1.65 gives 67 / 38 / 23 / 15 / 5 (the count is a **band**); aggregation `all` / `allpod` / `allpodcc` gives 18 / 15 / 3.

### 3.12 Added after review

- **Nested model** (`cpsa.nested`, `scripts/22`): a fourth XGBoost model on morphology plus the cell-count curve, compared with the cell-count-only model. Own permutation null (600 runs, 555 powered; sd₀ 1.38; raw type-I 13%; held-out calibrated 6.2%). **23 endpoints certified, all 15 included**; MT yes (q = 0.015), LDH no (q = 0.19). Median AUROC 0.661 against 0.541.
- **Logistic regression comparator:** median AUROC 0.618 against 0.650 for XGBoost, so the signal is not specific to trees.
- **Minimum detectable effect** (`cpsa.stats.precision`): MDE = (z₀.₉₅ + z₀.₈₀) × sd₀ × SE, the smallest true ΔAUROC detectable with one-sided α 0.05 and 80% power. Median over powered endpoints **0.25**; only 18 endpoints reach 0.10. Makes "no detectable advantage" interpretable.
- **Chip-scale simulation** (`cpsa.chip`, `scripts/24`): audit on simulated data at 20 to 200 compounds (table in Part 4); replicate-chip leakage demo (random labels reach AUROC 0.85 when each chip is treated as independent, 0.51 when grouped by compound); planning curve MDE ≈ e^0.98 × n_eff^−0.54 (R² 0.76), n_eff = 4 / (1/n_active + 1/n_inactive).
- **Second source** (`cpsa.data.eu_os`, `scripts/25`): HepG2 against primary hepatocytes on 327 shared compounds and 49 endpoints, single-number baseline in both arms (HepG2 has one concentration).

---

## 4. Results cheat sheet

### The funnel (CellProfiler, allpod, 404 tested plus the positive control)

| step | endpoints |
|---|---|
| naive ΔAUROC > 0 against the strong baseline | 292 |
| of which powered (15/15) | 160 |
| and raw bootstrap p < 0.05, uncorrected | 95 |
| and BH q < 0.05 on raw p | 64 |
| and BH q < 0.05 on null-calibrated p: **certified** | **15** (of 181 powered) |

Powered endpoints overall: median AUROC full 0.650 against strong baseline 0.541, median Δ +0.094, Δ > 0 for 160 of 181 (88%). The signal is broad; per-endpoint certainty is limited by power.

### Representations and the band

| representation | sd₀ (runs) | certified vs strong | vs scalar |
|---|---|---|---|
| CellProfiler (primary) | 1.37 (379) | 15 | 51 |
| CP-CNN | 1.54 (274) | 15 | 29 |
| DINOv2 | 1.27 (58) | 54 | 59 |

Every CellProfiler- or CP-CNN-certified endpoint is also certified with DINOv2; **8 endpoints are certified under all three**; 22 under at least two. Band for CellProfiler: 5 to 67 depending on the assumed null width.

### The 15 certified endpoints (CellProfiler)

| endpoint | kind | actives | strong cc → full | Δ | q |
|---|---|---|---|---|---|
| TOX21_PR_BLA_Antagonist_ratio | reporter | 145 | 0.53 → 0.72 | +0.184 | 0.002 |
| cell_type__HEK293 | cytotox | 186 | 0.71 → 0.82 | +0.110 | 0.002 |
| tissue__kidney | cytotox | 130 | 0.77 → 0.88 | +0.105 | 0.002 |
| cell_type__ME-180 | cytotox | 101 | 0.68 → 0.83 | +0.145 | 0.002 |
| TOX21_PXR_agonist | reporter | 124 | 0.53 → 0.69 | +0.160 | 0.005 |
| BSK_CASM3C_Proliferation | BioMAP | 34 | 0.33 → 0.62 | +0.290 | 0.014 |
| cell_type__ERR-HEK293T | cytotox | 131 | 0.72 → 0.83 | +0.116 | 0.014 |
| tissue__intestinal | cytotox | 100 | 0.71 → 0.82 | +0.110 | 0.014 |
| BSK_SAg_Eselectin | BioMAP | 30 | 0.50 → 0.74 | +0.241 | 0.015 |
| BSK_SAg_Proliferation | BioMAP | 56 | 0.58 → 0.79 | +0.204 | 0.015 |
| MT | native | 377 | 0.87 → 0.92 | +0.042 | 0.019 |
| BSK_3C_HLADR | BioMAP | 39 | 0.47 → 0.72 | +0.246 | 0.019 |
| cell_type__HepG2 | cytotox | 150 | 0.77 → 0.86 | +0.088 | 0.021 |
| tissue__liver | cytotox | 150 | 0.77 → 0.86 | +0.088 | 0.021 |
| TOX21_GR_BLA_Antagonist_ratio | reporter | 53 | 0.53 → 0.74 | +0.215 | 0.029 |

By category: 7 cell-based reporters (including the BioMAP human primary-cell assays), 7 cytotoxicity summaries, MT. Cell-free: 0 of the 8 testable (89% of cell-free endpoints cannot be adjudicated); consistent with the paper's finding that Cell Painting does not predict purely biochemical activity. The cell-based certified set includes antagonist-mode assays (PR-bla, GR-bla), where a signal decrease can mimic cytotoxicity: **a biologist's judgment on these is still pending** and is the main biological caveat.

### MT and LDH (the paper's own readouts)

| readout | scalar cc | strong cc | full | Δ vs strong | q (CP / CP-CNN / DINO) |
|---|---|---|---|---|---|
| MT | 0.835 | 0.873 | 0.916 | +0.042 | 0.019 / 0.039 / 0.001 |
| LDH | 0.936 | 0.957 | 0.969 | +0.011 | 0.27 / 0.29 / 0.076 |
| cell_count (control) | 0.975 | 1.000 | 0.991 | | |

The weak-baseline lesson: against the one-number baseline, 51 endpoints are certified and 38 of them vanish against the strong baseline; LDH would have been called a win against the one-number baseline (q = 0.016) but is not certified against the strong one. The positive control reaching 1.000 shows the strong baseline captures everything cell count can give.

### Nested, comparator, detectable effect

Nested test: **23 certified** (all 15 included); logistic regression median AUROC 0.618 against 0.650; median MDE 0.25 (18 powered endpoints at 0.10 or better, 36 at 0.15, 54 at 0.20).

### Chip scale (simulation, 720 endpoints per size)

| compounds | cannot be judged (15/15) | false credit raw / calibrated | power (calibrated) | smallest detectable advantage |
|---|---|---|---|---|
| 20 | 100% | 0% / 0% | 1% | 0.72 |
| 40 | 54% | 2% / 0.5% | 11% | 0.50 |
| 60 | 29% | 4% / 0.5% | 6% | 0.37 |
| 100 | 1% | 8% / 1.5% | 33% | 0.27 |
| 200 | 0% | 8% / 2% | 83% | 0.19 |

### Second source: HepG2 against primary hepatocytes (exploratory)

| | HepG2 profiles | primary hepatocytes |
|---|---|---|
| certified (plain bootstrap) | 16 (22) | 1 (7) |
| median AUROC full / baseline | 0.71 / 0.57 | 0.68 / 0.61 |
| MT | certified (0.79 vs 0.58) | certified (0.92 vs 0.81) |
| LDH | not certified (q 0.09) | not certified |

How to read it: MT agrees and LDH agrees; ten of the 16 HepG2 endpoints are also certified in the main audit. The 16 against 1 is **not** "HepG2 shows more morphology": the HepG2 baseline is one 10 µM count from another cell system (a much weaker baseline for labels defined in primary hepatocytes) and 327 compounds give much less power than 967. It reproduces the weak-baseline effect: **a verdict count depends on baseline strength and sample size.**

---

## 5. Why each design choice (the decision log)

| decision | why | what we rejected |
|---|---|---|
| Audit, not a new classifier | The trust problem is unsolved by higher AUROC; reusability is the value | Chasing a better model |
| Strong dose-response baseline | A weak baseline makes "morphology wins" a foregone conclusion (38 of 51 wins vanish) | Only the paper's single count |
| Compound-level rows | Labels are per compound; avoids replicate leakage and well-level label noise | Well-level rows (low concentrations inherit "active") |
| Grouped CV by compound (and chemical clusters as a check) | Prevents leakage, near-duplicate inflation | Plain row-level CV |
| XGBoost with paper hyperparameters, no tuning | Constant capacity, no tuning leakage, matches the source paper | Nested-CV tuning (about 10x cost, no gain for certification) |
| Paired compound bootstrap | Compounds are the exchangeable unit; pairing removes shared sampling noise | Row-level bootstrap |
| Permutation-calibrated p-values | The bootstrap ignores training variability (10.3% raw false positive) | Trusting nominal p |
| 15/15 power rule, fixed in advance | Count heuristic to avoid reporting noise; sensitivity 19/15/22/26 | Tuning the threshold to results |
| BH-FDR over powered endpoints only | Controls the false-discovery rate across hundreds of tests; valid under positive dependence | Bonferroni (too strict), no correction |
| `indeterminate` as a verdict | Under-powered endpoints cannot support any claim | Reporting clean AUROCs |
| Primary = CellProfiler + allpod | Pre-specified, paper default; CP-CNN and DINOv2 as robustness checks | Picking the representation after seeing results |
| Report a band, not a point | The count depends on the null width (5 to 67) | A single "how many benefit" number |
| Missing is not negative | Zero-filling would invalidate every statistic | Treating untested as 0 |
| Nested model added | Answers "adds beyond count", the sharper question | Only alone-vs-alone |
| Verdict wording "no detectable advantage" | It is absence of evidence, not equivalence | "No advantage" |
| Dataset-agnostic API (`cpsa.api`) | One call for any profiling table with a baseline and labels | One-off scripts |

---

## 6. Limitations and honest answers

Name these first; judges respect it.

1. **Per-endpoint power is low.** Only large effects are individually certifiable (median MDE 0.25). The certified count is a band (5 to 67), not "how many endpoints benefit". Answer: the signal is broad (Δ > 0 for 88%), the claim is about certification, and "no detectable advantage" is read with the detectable effect.
2. **The count depends on the aggregation rule** (18 / 15 / 3 for `all` / `allpod` / `allpodcc`). `allpodcc` leaves little beyond the cell-count curve by construction (and MT drops out, q = 0.90). `allpod` was pre-specified as primary. We state it as a limitation.
3. **The empirical null is an estimate.** 379 runs (CellProfiler), 274 (CP-CNN), only 58 for DINOv2 (weakest); calibrated rates are 2.6% (CellProfiler), 6.5% (CP-CNN), 6.2% (nested): near nominal, not exact.
4. **"Morphology" is not orthogonal to density.** Neighbour counts, granularity and embeddings can encode confluence; an "advantage" means information beyond the cell-count curve.
5. **Labels are partly cytotoxicity-entangled by construction** (the filter covers about 35% of records) and a 0 is a non-hit, not a confirmed negative. It bounds how strong any "beyond cell count" claim can be on this dataset.
6. **One hepatocyte pool, one 44 h time point, 2D plates.** Donor variability is unmodelled; this is not organ-on-chip. The chip results are simulations of what the method needs.
7. **BH assumes positive dependence** (plausible for endpoints sharing compounds and assays; not verified).
8. **The baseline can always be stronger.** Ours is best effort, not an upper bound.
9. **The HepG2 second source is exploratory**, not a replication (weaker baseline, fewer compounds).
10. **Biology pending:** a biologist has not yet judged the antagonist-mode reporters (PR-bla, GR-bla) and the BioMAP assays, which can be cytotoxicity-confounded.
11. **Open data discrepancies:** 242 against 251 cell-free compounds; the 100 µM ceiling deviation (8,400 against about 8,332 positives).
12. **Unverified items in the team draft [unverified]:** the "5-donor pool" statement and the two FDA quotations. Do not repeat them as facts.

---

## 7. Likely judge questions with model answers

**Q1. In one sentence, what did you build?**
A reusable statistical audit that tells you, per endpoint, whether a full Cell Painting profile beats a strong cell-count baseline, with calibrated p-values, multiple-testing control and an explicit "not enough data" verdict.

**Q2. Isn't this just a cell-count baseline? Where is the AI?**
The AI is the thing being audited. The contribution is the framework that makes AI readouts trustworthy: matched models on identical folds, a strong baseline, a permutation-calibrated test, power-aware verdicts. A better classifier would not answer the trust question.

**Q3. Why is the cell-count baseline "strong"?**
It sees the whole dose-response curve (eight concentrations, minimum, mean ratio, point of departure, interpolated counts), only cell-count information. The positive control endpoint reaches AUROC 1.000 against it, showing it captures everything count can give. 38 of 51 wins against the one-number baseline vanish against it.

**Q4. Why did you not tune the models?**
Fixed hyperparameters (the paper's) mean no tuning ever touches a test fold and model capacity is constant across comparisons. Tuning inside nested CV would multiply runtime about tenfold and could not change the certification logic. A logistic regression comparator confirms the signal is not model-specific.

**Q5. How do you know your p-values are right?**
We ran the identical pipeline on label-permuted endpoints (true effect 0). The raw bootstrap rejected 10.3% at a nominal 5%, so we calibrated with an empirical null (sd 1.37). On held-out permutation runs the calibrated rate is 2.6%. CP-CNN is 6.5% and the nested test 6.2%, close to nominal.

**Q6. Why is the bootstrap too optimistic?**
It resamples compounds but keeps the out-of-fold predictions fixed, so it ignores the variability of the fitted models (different training splits). The permutation null measures the full variability.

**Q7. Why only 15 of 181? Is your method too conservative?**
Power. With 15 to 60 actives per endpoint only large effects are certifiable: median detectable effect 0.25 AUROC. The signal is broad (88% have Δ > 0, median +0.094). The count is a band (5 to 67) depending on null width, and we say so. The nested test certifies 23.

**Q8. What is an `indeterminate` endpoint and why 55% of them?**
Fewer than 15 actives or 15 non-hits. Their AUROCs are noise (sd 0.175), so we refuse to report them as wins or losses. This is a result: most ToxCast endpoints cannot be judged on about 1,000 compounds.

**Q9. Does "no advantage" mean cell count is enough?**
No. It means no detectable advantage after correction. It is absence of evidence, not equivalence, and is only as informative as the endpoint's detectable effect, which we report for every endpoint.

**Q10. Why BH and not Bonferroni?**
BH controls the false discovery rate, the right quantity when screening hundreds of endpoints and tolerating a small fraction of false discoveries; Bonferroni controls the chance of any false positive and would leave almost nothing. BH is valid under independence or positive dependence, plausible here; we state that it is assumed.

**Q11. Endpoints are correlated. Does that break BH?**
BH remains valid under positive dependence; effective independent tests are fewer than nominal, so it is conservative. For enrichment, where non-independence matters more, we added an assay-level permutation test.

**Q12. Why is 15/15 the power threshold?**
A pre-specified heuristic, fixed before looking at results, not tuned. Sensitivity: 10 / 15 / 20 / 30 gives 19 / 15 / 22 / 26 certified. We also report a precision view: the minimum detectable effect per endpoint.

**Q13. Why is MT certified but LDH not?**
MT is an earlier stress readout (metabolic activity), where morphology adds beyond count with all three representations (Δ +0.042, q 0.019 / 0.039 / 0.001). LDH is membrane damage, close to cell loss, and cell count already predicts it very well (AUROC 0.957 against 0.969 for full): Δ +0.011, q 0.27 / 0.29 / 0.076. Against the one-number baseline LDH would have looked like a win.

**Q14. Which endpoints did morphology win on and what is the biology?**
Seven cell-based reporters (including BioMAP primary-cell assays and the PR-bla and GR-bla antagonist assays, PXR agonist), seven cytotoxicity summaries and MT. None of the 8 testable cell-free endpoints. At compound level the benefit concentrates in compounds that change cells without killing them (+0.15), and is almost nil where cells die (+0.01).

**Q15. Could the antagonist-mode reporter wins just be cytotoxicity?**
Possibly: signal-decrease assays are classically cytotoxicity-confounded. They are certified against a strong cell-count baseline, which reduces that risk but does not remove it (morphology is not orthogonal to density). A biologist's review of these is the main pending biological check.

**Q16. Why is cytotoxicity enriched among certified endpoints?**
It looks enriched (7/24, Fisher q = 0.003), but those endpoints have about four times more actives, so they are easier to certify; with at least 50 actives the difference disappears, and with whole assays as the unit p = 0.12. We report it as descriptive.

**Q17. How do you know it is not a plate or batch artifact?**
A probe using plate, well, batch and replicate counts alone scores AUROC 0.512 (chance); adding them to the cell-count model changes nothing. Compounds also span 13 to 16 plates from two batches, so layout averages out.

**Q18. Could near-duplicate compounds inflate performance?**
We re-ran with chemical-similarity folds (ECFP4 Butina clusters, 921 groups): median AUROC 0.650 to 0.651 and all 15 certified endpoints stay certified.

**Q19. Why three image representations?**
To show the result does not depend on one feature extractor: 8 endpoints are certified under all three; MT in all three. The certified count differs by representation (15 / 15 / 54), which we report instead of picking one.

**Q20. Why does DINOv2 certify 54 and CellProfiler 15?**
Different representations carry different signal and different noise; DINOv2 also has the weakest null (58 runs), so its count is the softest. We treat CellProfiler as primary because it was pre-specified and has the strongest calibration evidence.

**Q21. Does the result depend on the aggregation rule?**
Yes: 18 / 15 / 3 for `all` / `allpod` / `allpodcc`. `allpodcc` leaves little beyond the cell-count curve by construction, and MT drops out there. We pre-specified `allpod` and state the dependence as a limitation.

**Q22. What does the nested model add?**
It answers the sharper question: does morphology add information on top of count? A model on morphology plus the count curve beats the count-only model on 23 endpoints, all 15 included, with its own permutation null (calibrated rate 6.2%).

**Q23. What did the second dataset show?**
On 327 shared compounds, HepG2 profiles certify 16 endpoints and primary hepatocytes 1; MT agrees and LDH agrees. We do not call it a replication: the HepG2 baseline is a single 10 µM count from another cell system and 327 compounds give less power. It demonstrates that verdict counts depend on baseline strength and sample size.

**Q24. How is this relevant to organ-on-chip?**
Chips adopt AI image readouts with tens of compounds and extra density-like confounds. Our simulation at 20 to 200 compounds shows the calibrated audit stays safe (false credit near 0 to 2%) but needs about 100 to 200 compounds for real power; below 60, only AUROC differences of about 0.4 or more are certifiable. Replicate chips need folds grouped by compound (random labels reach AUROC 0.85 otherwise). The demo's planner and checklist turn this into study design.

**Q25. You have no chip data. Is the chip claim real?**
We do not claim chip results. We claim what the method needs at chip scale, validated on simulation with known truth, and a concrete list of what changes: labels (albumin, urea, CYP3A4, DILI), a 3D cell-count proxy, features, grouping, sample size. No public liver-chip dataset combines images with toxicity labels.

**Q26. What about donor variability and time points?**
Not modelled: one hepatocyte pool, one 44 h time point. We state it. The method, not the 15 endpoints, is what we claim generalises.

**Q27. Are the labels trustworthy?**
They are the authors' filtered ToxCast/Tox21 hit-calls. We use only the `*_binary` files (the raw `hitcall` is not a label), treat untested as missing, and read 0 as "non-hit". About 35% of records get the cytotoxicity filter; the rest are not adjusted, which bounds how strong a claim can be.

**Q28. What is the most important thing that nearly went wrong?**
Using the raw `hitcall` in the `*_info` files as a label. It is unfiltered; only the `*_binary` files carry the real labels. The biology side found it by reading the authors' label-construction notebook, before any model was trained on it.

**Q29. How reproducible is it?**
CPU only, fixed seeds, MD5-verified downloads, 69 tests, precomputed results in the repo so the demo runs without a rebuild, numbered scripts and one driver (`scripts/20_full_audit.sh`) for the full rebuild.

**Q30. What would you do next?**
Biologist review of the reporter endpoints; run on a chip dataset when one with labels exists; a density-residualised comparator; more DINOv2 null runs; more endpoint-specific baselines; extend the audit to other readouts than AUROC.

**Q31. Why not use a deep model on the images directly?**
The audit's question is about what the profile adds to cell count, not about maximising accuracy; the three representations already include a CNN and a self-supervised model, computed by the data authors. The audit can be applied to any model's predictions.

**Q32. Is a "certified" endpoint proof that the model is biologically meaningful?**
No. Certified means the profile carries information beyond the cell-count curve that survives correction. It could still include other density-related image properties. Biological meaning needs follow-up (the compound-level analysis is supportive, not causal).

**Q33. What is the single most important number?**
15 of 181 certified (and 64 raw): the gap between 64 and 15 is the cost of not auditing the audit.

**Q34. What would change your conclusion?**
A stronger baseline certifying fewer endpoints; a different null width (band 5 to 67); another aggregation rule (18 / 15 / 3). Those are all reported, which is why we report a band and not a point.

**Q35. Who did what?**
Vrushti: biology and data (label provenance, filter semantics, dataset QA, interpretation). Dhairya: ML and statistics (models, calibration, multiple testing, demo, pipeline).

---

## 8. Things never to say

- "It replicates on HepG2." (It agrees on MT; 16 against 1 reflects baseline and power.)
- "The nested check uses logistic regression." (The nested model is morphology plus cell count; logistic regression is a separate comparator.)
- "The 15 certified endpoints are a lower bound." (They are a point in a band of 5 to 67.)
- "No advantage means cell count is enough." (It is absence of evidence.)
- "Five donors." [unverified] and the exact FDA quotations [unverified].
- "We built a chip model" or any chip-measured result. (It is simulation and planning.)
- "Cytotoxicity is the biological winner." (Enrichment is confounded by endpoint size and descriptive only.)
- "Morphology is independent of cell density." (It is not.)

---

## 9. Formulas and glossary

**Formulas**

- Normalisation: x' = (x − median_DMSO) / MAD_DMSO (per plate, per feature).
- AUROC = P(score of random active > score of random non-hit) (ties count 0.5); equals the Mann-Whitney U statistic divided by n_active × n_non-hit.
- ΔAUROC = AUROC(full) − AUROC(baseline), pooled OOF predictions.
- Bootstrap p = (1 + #{Δ* ≤ 0}) / (B + 1), B = 1000.
- SE from interval = (CI_hi − CI_lo) / (2 × 1.96); z = Δ / SE.
- Calibrated p = P(Z > z / sd₀), sd₀ = std of z over permutation runs.
- BH: q_(i) = min_{j ≥ i} m · p_(j) / j.
- Brier = mean (p − y)²; skill = 1 − Brier / Brier(prevalence).
- ECE = Σ_bins (n_bin / n) × |mean prediction − observed frequency|.
- MDE = (z₀.₉₅ + z₀.₈₀) × sd₀ × SE ≈ (1.645 + 0.842) × sd₀ × SE.
- n_eff = 4 / (1/n_active + 1/n_non-hit); planning curve MDE ≈ e^0.98 × n_eff^−0.54.
- Per-compound benefit: r_i = P(correct ranking against the opposite class); benefit_i = r_i(full) − r_i(baseline).

**Glossary**

- **Endpoint:** one binary outcome column. **Powered:** at least 15 actives and 15 non-hits. **Indeterminate:** not powered.
- **OOF (out-of-fold):** a prediction from a model that never saw that compound.
- **POD (point of departure):** lowest concentration with a detectable change (profile POD, cell-count POD).
- **Positive control:** the cell_count endpoint, whose label is defined from cell count.
- **Empirical null:** test-statistic distribution when the true effect is zero, estimated by running the whole analysis on shuffled labels.
- **FDR / BH:** expected fraction of false discoveries among those declared; Benjamini-Hochberg procedure.
- **Nested model:** morphology plus the cell-count curve, compared against the count-only model.
- **MDE:** minimum detectable effect.
- **Signal classes:** count_explained, morphology_signal_uncertified, no_detectable_signal (descriptive splits of `no_advantage`).
- **DMSO:** solvent control on every plate. **MAD:** median absolute deviation.
- **CP-CNN / DINOv2:** a Cell-Painting-trained CNN representation; a self-supervised vision-transformer embedding.
- **NAM:** new approach methodology (non-animal methods). **OoC:** organ-on-chip.

---

## 10. Code map and how to run things

| area | files |
|---|---|
| Data loading and preprocessing | `src/cpsa/data/load_profiles.py`, `preprocess.py`, `build_dataset.py`, `load_labels.py`, `label_context.py`, `technical.py`, `chem_groups.py`, `eu_os.py` |
| Models and CV | `src/cpsa/models/train_eval.py` (XGBoost, logistic, grouped CV, paired bootstrap), `baseline.py`, `full_model.py` |
| Statistics | `src/cpsa/stats/null_calibration.py`, `multiple_testing.py`, `power_check.py`, `calibration.py`, `signal_class.py`, `precision.py`, `pod_audit.py` |
| Biology | `src/cpsa/biology/enrichment.py`, `compound_classes.py`, `feature_families.py` |
| Orchestration | `src/cpsa/audit.py` (main audit), `api.py` (dataset-agnostic API), `nested.py`, `chip.py`, `simulate.py` |
| Scripts | `01` download, `02` labels, `03` audit, `04` enrichment, `07` finalise, `08` null control, `09` technical probe, `10` feature families, `11` paper-faithful CV, `15` calibrate verdicts, `17` simulation, `18` compound classes, `19` filter-stratified, `20_full_audit.sh`, `22` nested, `23` detectable effect, `24` chip scale, `25` HepG2, `26` cell-line images, `27` build Kaggle Writeup |
| Demo | `demo/app.py` (Streamlit; tabs Endpoints, Compounds, All results, Where morphology wins, Example images, Cell lines, Organ-on-chip, Method) |
| Docs | `docs/PROBLEM_STATEMENT.md`, `report/technical_report.md` (and `.pdf`), `report/kaggle_writeup.md`, `report/demo_video_script.md` |
| Results | `results/audit_table.csv` (one row per endpoint), `results/<config>/` per configuration, `results/chip_scale/`, `results/eu_os/`, `results/figures/` |

Run commands (from the project folder; the path contains a colon, so call the venv's Python directly):

```bash
.venv/bin/streamlit run demo/app.py               # the demo, from precomputed results
.venv/bin/python -m pytest                        # 69 tests
.venv/bin/python scripts/03_run_audit.py --feature-set cellprofiler --agg allpod
.venv/bin/python scripts/12_headline_numbers.py   # regenerate the headline JSON
```

Live demo: https://beyond-the-count1.streamlit.app/ (kept awake by a scheduled GitHub Actions visit).

Good places to open during a defense: the plate-map hero (all 405 endpoints), the MT endpoint (stats panel with nested comparison and detectable effect), the Where morphology wins tab, the Cell lines tab, and the Organ-on-chip planner (move the sliders).

---

## 11. Self-quiz

Answer from memory, then check against Parts 3 to 6.

1. Draw the pipeline from profiles to verdicts and name the four verdicts.
2. Why is the unit of analysis the compound, and what does `allpod` do?
3. What exactly is in the strong baseline, and why is it "strong"?
4. Write the paired bootstrap p-value and explain why it is anti-conservative.
5. How is sd₀ estimated, what is its value for CellProfiler, and what does it do to the p-value?
6. State the funnel numbers (292, 160, 95, 64, 15) and what removes endpoints at each step.
7. Why is the certified count a band, and what is its range?
8. Explain MT against LDH with numbers.
9. What does "no detectable advantage" not mean?
10. What do 38 of 51 and the HepG2 16 against 1 have in common?
11. How does the nested model differ from the main comparison, and what did it find?
12. What is the minimum detectable effect, its median, and why report it?
13. Why does enrichment of cytotoxicity endpoints not count as a finding?
14. What did the technical probe, chemical folds and simulation each rule out?
15. What does the chip-scale simulation say at 20, 60 and 200 compounds?
16. Why must folds be grouped by compound for replicate chips? (Give the 0.85 against 0.51 result.)
17. List five limitations you would name before a judge does.
18. What is the single most dangerous data mistake you avoided and how?
