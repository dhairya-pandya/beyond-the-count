# Beyond the Count: A Reusable Statistical Audit for Cell-Count Shortcuts in Cell Painting Toxicology

## 1. Title, team, and summary

**Track:** AI for Life Science Challenge, [5th Pazhou Algorithm Competition](https://www.aicompetition-pz.com/), **Model & Algorithm** category.

| Member | Role |
| --- | --- |
| Vrushti | Biology and data lead: label construction and provenance, dataset QA and sanity checks, biological interpretation of results |
| Dhairya | ML and statistics lead: model architecture, calibration and multiple-testing framework, [Streamlit demo](https://beyond-the-count1.streamlit.app/), reproducible pipeline |

**Summary:** [Cell Painting](https://doi.org/10.1038/nprot.2016.105) is a high-content microscopy assay that stains cells with six dyes across five channels and extracts thousands of morphological features per cell. It's being adopted across pharma and regulatory toxicology as a scalable, animal-free readout of compound safety. One risk is that many toxicity labels are entangled with cytotoxicity, so a model that only knows how many cells survived a treatment can score well on those labels without reading any real morphology. We built **CPSA**, a [dataset-agnostic audit](https://github.com/dhairya-pandya/beyond-the-count) that tests, endpoint by endpoint, whether a full morphology profile predicts a toxicity label beyond what a cell-count baseline already predicts, with calibrated statistics, multiple-testing correction, and explicit handling of underpowered endpoints. Applied to the public OASIS primary human hepatocyte dataset ([Ewald et al., *Cell Systems*, 2026](https://doi.org/10.1016/j.cels.2026.101566)), the audit certifies **15 of 181 testable endpoints** for a genuine morphology advantage, including the **metabolic-activity readout (MT)** across **three independent image representations**, and it holds up on a second, independent dataset ([EU-OPENSCREEN](https://zenodo.org/records/19347244) HepG2). The method is built to generalise beyond this one dataset. A built-in **organ-on-chip planner** shows what the same audit would conclude at chip scale, which is why we think of this as a reusable quality-control layer for phenotypic toxicology pipelines generally, 2D plate or organ-on-chip, before anyone trusts an AI readout from one.

## 2. Problem definition and application scenario

Early toxicity screening needs to tell, for a given compound and a given biological endpoint, whether that compound is active, ideally before it reaches animal or human studies. Cell Painting is attractive here because one imaging assay yields a rich, unbiased morphological fingerprint instead of a single pre-chosen readout. But toxicity labels themselves are frequently derived from, or correlated with, cytotoxicity: a compound that kills or visibly stresses cells changes almost every measurable feature at once, including the crudest one: how many cells are left on the plate. This creates a specific, underappreciated **failure mode**: a model can appear to "read morphology" and predict toxicity endpoints well, while in fact doing nothing more sophisticated than counting surviving cells. [Seal, Dee, Shah et al. (*Nature Communications*, 2026)](https://doi.org/10.1038/s41467-026-68725-5) demonstrated this same pattern on independent Cell Painting bioactivity benchmarks: a cell-count-only baseline (**mean AUC 0.68 ± 0.21**) matched a full-profile neural network (**mean AUC 0.67 ± 0.19**), and **31 of 49 (63%)** endpoints well-predicted by gene expression were predicted equally well by cell count alone.

Phenotypic profiling assays are being positioned as a scalable, more human-relevant alternative to animal testing, and regulators are starting to ask for it: FDA's 2026 draft guidance, [*General Considerations for the Use of New Approach Methodologies in Drug Development*](https://www.federalregister.gov/d/2026-05390), states that "**technical characterization** of a NAM is essential to establish scientific confidence in the data obtained," and that a method is "**fit-for-purpose**" only if it is shown to characterize risk comparably to established methods. A phenotypic AI model that is secretly a cell counter has not cleared that bar, even if its benchmark numbers look strong.

The immediate target of this audit is 2D plate assays like the OASIS hepatocyte dataset used here, but the same failure mode will recur, and matter more, as organ-on-chip (OoC) systems scale up. OoC platforms are themselves being developed as animal-free, human-relevant toxicology tools, and as they adopt AI-based image readouts, they inherit the identical risk: an OoC morphology model could pass internal validation by reading cell density or viability rather than the specific tissue-level biology the chip was built to capture. We designed this audit to be dataset-agnostic for this reason, so a chip-based toxicology pipeline can run the same check on its own data before trusting an AI readout, rather than discovering the confound after the readout is already informing a go/no-go decision. The built-in **organ-on-chip planner** ([Section 6](#6-experiments-results-and-evaluation)) makes this concrete: it simulates the audit at chip-realistic compound counts (20–200) to show what sample size a chip study needs before this check has any power to catch a shortcut.

## 3. Data sources, licenses, processing methods, and compliance

**[Ewald et al. (OASIS Consortium), *Cell Systems*, 2026](https://doi.org/10.1016/j.cels.2026.101566)**: Cell Painting profiles of **1,085 compounds** in primary human hepatocytes (a pooled 5-donor system), imaged at a single 44-hour time point. Profiles are published on Zenodo ([10.5281/zenodo.17067683](https://zenodo.org/records/17067683), CC-BY 4.0) in three independent image representations: [CellProfiler](https://cellprofiler.org/) hand-engineered features, a Cell-Painting-specific CNN (CP-CNN), and [DINOv2](https://github.com/facebookresearch/dinov2) self-supervised embeddings. **967 of the 1,085 compounds** carry a real `Metadata_OASIS_ID`; the remaining 119 are anonymized and usable only for native readouts that don't require compound identity.

**[ToxCast](https://comptox.epa.gov/dashboard/)/[Tox21](https://tox21.gov/) bioactivity hit-calls**, released by the dataset's own authors in their [analysis repository](https://github.com/jessica-ewald/2024_09_09_Axiom_OASIS) (BSD-3-Clause), built from invitrodb v4.1. The release provides both a **raw, unfiltered `hitcall`** and a **filtered binary label**; **only the filtered binary files are valid as ground truth**. A hit is defined as `hitcall > 0.9`, then zeroed out if the endpoint's AC50 exceeds half the matched consensus cytotoxicity AC50 for that cell type (tissue as fallback), where a consensus exists only if ≥20% of matched viability tests fired and the median AC50 is ≤100 µM; duplicate records are collapsed by majority vote, with ties counted as no hit. Endpoints are retained only with ≥5 positive and ≥5 negative compounds, giving 38 cytotoxicity, 292 cell-based, and 72 cell-free ToxCast/Tox21 endpoints (402), plus the study's own LDH, MTT, and cell-count readouts embedded directly in `metadata.parquet` (**405 endpoints total**). Native LDH/MTT/cell-count labels have zero missing values across all 21,456 wells; ToxCast/Tox21 labels cover 670 of 1,085 tested compounds (61.8%) for cytotoxicity and cell-based endpoints, and 251 for cell-free endpoints.

A correctness-critical rule governs how missing data is handled: **an untested compound–endpoint pair is *missing*, never a negative hit-call.** This is confirmed directly on the released binary label matrix: 963 compounds × 292 cell-based endpoints = 281,196 possible cells, only 107,103 observed (~62% missing). Zero-filling missing pairs would silently inflate every endpoint's "inactive" count and invalidate the per-endpoint AUROC, power check, and FDR correction downstream, exactly the class of measurement error this project exists to catch elsewhere, so `cpsa.data.load_labels` and `cpsa.data.preprocess` treat an absent pair as excluded from that endpoint's compound set throughout, never as `0`.

Before committing to the pipeline, we ran a **standalone cell-count-only check** across 670 compounds with both a usable cell-count feature and a ToxCast label: a cell-count-only classifier reached a **median cross-validated AUC of only 0.55** across 320 testable endpoints (mean 0.53, 25th/75th percentile 0.45/0.62), with just 7 of 320 (2.2%) endpoints at AUC ≥0.75. This was a go/no-go check on the data, not the final result. It confirmed that cell count alone does not trivially explain most labels, which justified building the full calibrated audit rather than concluding the dataset was unusable.

A second, independent dataset, **[EU-OPENSCREEN](https://zenodo.org/records/19347244) HepG2 Cell Painting profiles** (four imaging sites), was used to re-run the audit on 327 compounds shared with the OASIS compound set, as an out-of-dataset check on the MT-readout result.

**All data and code used are public and explicitly licensed for reuse:** Zenodo profiles and the OASIS metadata are CC-BY 4.0 (attribution required, redistribution permitted); the label-construction repository is BSD-3-Clause. No proprietary, restricted, or personally identifying data is used. Compound identities are either public ToxCast/Tox21 chemical identifiers or intentionally anonymized by the original authors, and we did not attempt to de-anonymize the 119 anonymized compounds, since doing so serves no analytical purpose here and the native-readout audit uses every compound productively without it.

## 4. Methods, models, algorithms, and system architecture

**Core design: three matched models, not one.** For every assay endpoint, `CPSA` trains three models on identical compound-grouped folds (5-fold × 3 repeats, grouped by `Metadata_OASIS_ID` so no compound's replicates leak across train/test):

1. **Full morphology**: the complete per-compound feature profile (CellProfiler, CP-CNN, or DINOv2, aggregated under one of three schemes: `all`, `allpod`, `allpodcc`).
2. **Cell count (single number)**: one scalar cell-count summary per compound, the simplest possible shortcut baseline.
3. **Cell count (dose-response curve)**: the full concentration-resolved cell-count curve (minimum, area under the curve, cell-count point-of-departure), a deliberately *strong* baseline so that any detected advantage reflects information beyond cell count, not an artefact of a weak comparison.

All three use [XGBoost](https://xgboost.readthedocs.io/) classifiers so that model capacity is held constant across the comparison; a nested check re-runs the comparison with a regularized [logistic regression](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression) to confirm the result is not an artefact of one model family.

**Statistical pipeline, applied uniformly to every endpoint:**

1. **Paired compound bootstrap of ΔAUROC** between the full-morphology model and the cell-count-curve baseline.
2. **Label-permutation null**, run at the audit's own settings, to calibrate the bootstrap p-values empirically rather than trusting their nominal distribution. This calibration was validated to land at 2.6% false-positive certification against a nominal 5% threshold on held-out runs.
3. **Power check**: an endpoint only receives a verdict if it has at least 15 positive and 15 non-hit compounds; otherwise it is marked `indeterminate` rather than forced into a conclusion the data can't support.
4. **Benjamini–Hochberg FDR correction** across all powered endpoints simultaneously, valid under the positive dependence that endpoints sharing compounds and assay batches display.
5. **Verdict assignment**, one of four per endpoint: `morphology_advantage` (certified after correction), `no_advantage` (enough data, no detectable edge), `indeterminate` (underpowered), `positive_control` (the cell-count endpoint itself, solved by construction).
6. **Probability calibration** of the winning model's predictions (Brier score, expected calibration error, calibration slope) and **enrichment analysis** (Fisher's exact test plus a permutation test over whole assays, so correlated endpoints from one assay are not double-counted) by target family, assay design, and cell type.
7. **Compound-level benefit decomposition**: a ranking-benefit score per compound showing which compounds gain most from morphology over cell count, cross-referenced against assay/target-family enrichment.
8. **Organ-on-chip simulation**: the same audit re-run on simulated data at 20–200 compounds to characterize detectable effect size as a function of study size, feeding the demo's chip-study planner (Section 6).

**Pipeline stages and where they live in the codebase:**

| Stage | Module |
| --- | --- |
| Normalization (per-plate DMSO-MAD), feature filtering, per-compound aggregation | `cpsa.data.preprocess` |
| Label loading and semantics (missing ≠ negative, cytotoxicity filter) | `cpsa.data.load_labels`, `cpsa.data.label_context` |
| Three matched models, grouped cross-validation, bootstrap of ΔAUROC | `cpsa.models` |
| Permutation null and calibrated p-values | `cpsa.audit`, `cpsa.stats.null_calibration` |
| Power check, BH-FDR, verdict assignment | `cpsa.stats.multiple_testing` |
| Probability calibration | `cpsa.stats.calibration` |
| Enrichment by target family, assay design, cell type | `cpsa.biology.enrichment` |
| Compound-level benefit and compound classes | `cpsa.biology.compound_classes` |
| Organ-on-chip simulation with known ground truth | `cpsa.simulate` |
| Public API | `cpsa.api` (`shortcut_audit`, `permutation_null`) |

![System architecture](https://raw.githubusercontent.com/dhairya-pandya/beyond-the-count/main/results/figures/pipeline.png)

**Figure: system architecture.** Inputs (morphology profiles, cell counts, ToxCast/Tox21 labels) feed a preprocessing stage, which branches into the three matched models described above (full morphology; cell count as a single number; cell count as a dose-response curve). All three feed one statistics pipeline (paired compound bootstrap of ΔAUROC, empirical label-permutation null calibration, the power check, and Benjamini–Hochberg FDR correction), which sorts every endpoint into one of four verdicts: `morphology_advantage`, `no_advantage`, `indeterminate`, or `positive_control`.

*Each endpoint runs through three matched models trained on identical folds; one statistics pipeline then sorts the result into one of four verdicts.*

## 5. Implementation details and technical components

**[Repository layout](https://github.com/dhairya-pandya/beyond-the-count):** `src/cpsa/` (the installable package), `scripts/` (numbered, ordered pipeline stages), `tests/` (unit tests), `demo/` (the Streamlit app), `docs/` (problem statement and planning documents), `report/` (this report and supporting write-ups), `results/` (generated outputs, included pre-computed so the demo runs without a full rebuild).

**Rebuilding from raw data:** run the numbered scripts in `scripts/` in order, from `01_download_zenodo_data.py` through `15_calibrate_verdicts.py`, or run `scripts/20_full_audit.sh` for the complete three-image-representation audit in one call.

**Technical stack:**

| Component | Choice | Why |
| --- | --- | --- |
| Classifier | [XGBoost](https://xgboost.readthedocs.io/) (gradient-boosted trees) | Held constant across all three models so the comparison isolates feature content, not model capacity |
| Nested robustness check | Regularized logistic regression | Confirms the result is not an artifact of one model family |
| Cross-validation | Compound-grouped K-fold (5-fold × 3 repeats) | Prevents replicate/concentration rows of the same compound leaking across train/test |
| Significance testing | Paired bootstrap of ΔAUROC + empirical permutation null | Calibrates p-values against the audit's own null, not an assumed distribution |
| Multiple-testing correction | Benjamini–Hochberg FDR | Valid under positive dependence across endpoints sharing compounds/assays |
| Calibration diagnostics | Brier score, expected calibration error (ECE), calibration slope | Checks predicted probabilities are trustworthy, not just well-ranked |
| Demo / interface | [Streamlit](https://beyond-the-count1.streamlit.app/) | Fast, shareable, interactive exploration of per-endpoint results, calibration, enrichment, and the organ-on-chip planner |
| Reproducibility controls | Fixed random seeds, MD5-verified downloads | Every number in this report is exactly reproducible from the raw public data |

**Design choices worth calling out:**

- **A strong, not weak, baseline.** The cell-count comparator sees the entire concentration-resolved dose-response curve (eight concentrations, minimum, AUC, point-of-departure), not a single median count. This was a deliberate choice: a weak baseline would make "morphology beats cell count" a foregone conclusion and not a meaningful test.
- **Three image representations, not one.** Running the identical audit on CellProfiler features, a Cell-Painting CNN, and DINOv2 embeddings lets us ask which certified endpoints are representation-independent (**8 of the 15** certified endpoints hold under all three) versus representation-specific, a biologically more defensible claim than reporting a result from a single feature extractor.
- **Everything runs on a laptop CPU.** No GPU or cloud compute dependency, which matters for a tool meant to be re-run by other toxicology teams on their own infrastructure, not just once for this submission.

## 6. Experiments, results, and evaluation

The headline result: of 405 total endpoints, 181 had enough data (≥15 positives and ≥15 non-hits) to receive a verdict; of those 181, **15 are certified `morphology_advantage`** after calibration and FDR correction, including the metabolic-activity readout (MT), which holds across **all three** image representations (CellProfiler, CP-CNN, DINOv2). **8 of the 15** certified endpoints hold under all three representations simultaneously, the strongest form of the result since it does not depend on which feature extractor was used.

| Check | Result |
| --- | --- |
| Endpoints certified `morphology_advantage` (main audit) | 15 of 181 powered endpoints |
| Endpoints certified under all 3 image representations | **8** |
| Nested test: full morphology vs. cell-count **curve** (not just a number) | **23 endpoints** certified, including all 15 from the main audit |
| Model-family robustness | Regularized logistic regression confirms the same signal for a linear model |
| Calibration accuracy (empirical vs. nominal) | **2.6%** false-positive rate at a nominal 5% threshold, on held-out runs |
| Compound-level benefit | Morphology helps most for compounds that damage cells while leaving them in place (**+0.15 ranking benefit**): a toxicity signature a cell count would miss by construction |
| Cross-dataset check | Independent EU-OPENSCREEN HepG2 screen (327 shared compounds, 4 imaging sites) reproduces the MT-readout result |

**Robustness checks, all consistent with the main result:**

- Alternative feature-aggregation rules (`all`, `allpod`, `allpodcc`).
- Chemical-similarity-grouped folds (not just compound-identity-grouped), to rule out near-duplicate compounds inflating apparent performance.
- A plate/well/batch probe designed specifically to surface hidden technical confounds.
- A simulation with known ground truth (used independently to validate the organ-on-chip planner below).

**The early biology-side sanity check from Section 3, and the final audit** should not be read as the same measurement: the early check asked "can cell count alone explain most labels, across all compounds and endpoints, with no calibration or correction?" and found a weak, mostly-chance signal (**median AUC 0.55**), useful as a go/no-go on the data before investing in the full pipeline. The final audit asks the sharper, decision-relevant question: "after controlling for multiple testing and requiring adequate power, where does the full profile beat a *strong* cell-count baseline?" and answers it endpoint by endpoint with calibrated certainty. The two results are complementary, not duplicative: the first establishes that cell count isn't trivially sufficient; the second establishes exactly where morphology adds value beyond it.

Re-running the full audit on simulated data at compound counts from 20 to 200 characterizes the minimum detectable AUROC advantage as a function of study size, directly answering "how many compounds does a chip study need before this audit has power to catch a shortcut?" This feeds the [Streamlit demo's Organ-on-Chip tab](https://beyond-the-count1.streamlit.app/), which turns the simulation into a planner and a pre-study checklist (ground-truth labels, a 3D-appropriate cell-count baseline, feature choice, compound grouping for cross-validation, and minimum sample size).

**Interactive demo:** [beyond-the-count1.streamlit.app](https://beyond-the-count1.streamlit.app/): per-endpoint verdict table, calibration reliability diagrams, recalibrated compound-level predictions, enrichment plots by target family/assay/cell type, and a plate-map view of all 405 endpoints.

## 7. Reliability analysis and limitations

**Statistical reliability, by design:**

- **FDR correction is conservative by construction.** A true but borderline effect at an endpoint may be labelled `no_advantage` rather than `morphology_advantage` because of the correction. This is an intentional, defensible trade-off, not an oversight, but it means the 15 certified endpoints are a lower bound on where morphology plausibly helps, not an exhaustive list.
- **The power check (≥15 positives/15 non-hits) is itself conservative;** endpoints below that threshold are explicitly `indeterminate` rather than silently absorbed into "no advantage," so the 181-of-405 powered fraction should be read alongside the 15-of-181 certified fraction, not instead of it.

**Data and label limitations:**

- **Single 5-donor hepatocyte pool, single 44-hour time point.** Donor-to-donor variability is entirely unmodeled; kinetic or delayed/recovering toxicity outside that one time point is not captured. Results describe this pool and this time point only.
- **2D plate assay, not an organ-on-chip system.** Every number in Section 6 comes from a 2D hepatocyte monolayer; the organ-on-chip simulation (Section 6) shows what the *method* predicts at chip scale, not a chip-measured result. We make no claim that the specific certified endpoints transfer unchanged to a 3D or perfused system.
- **Label construction is partly cytotoxicity-entangled by design, not fully correctable.** The cytotoxicity-AC50 filter only applies where a consensus cytotoxicity AC50 exists for a compound; it does not exist for roughly 65% of raw cell-based records, so a non-trivial share of labels are not cytotoxicity-adjusted at all. A `0` in a label column can mean "no hit," "a hit overridden by the filter," or "a tie between duplicate records", never a confirmed negative. This is a property of the released ground truth, not of our audit, but it bounds how strong a claim "beyond cell count" can be for any method run on this dataset.
- **Endpoint coverage gaps.** 38 of 48 source cytotoxicity categories survive the minimum-class-support filter (≥5 positive/≥5 negative); the same filter applies to cell-based and cell-free endpoints. Compound coverage for ToxCast/Tox21 endpoints is 61.8% of the tested library (670/1,085), versus the 89% figure sometimes quoted for this dataset, which is a union of ToxCast chemical identifiers rather than an intersection with compounds actually tested in this screen. We report the intersection figure throughout this report as the honest denominator.

**One known deviation from the published methodology:** the released label files apply the paper's stated 100 µM cytotoxicity-AC50 ceiling only to the consensus cytotoxicity AC50 itself, not per-endpoint as the published Methods describes; we used the released labels as-is rather than re-deriving them, since re-deriving ground truth labels was out of scope for an audit tool meant to consume whatever labels a dataset provides. Applying the ceiling exactly as described would change the cell-based positive count from 8,400 to an estimated 8,332 and drop one endpoint, a small, bounded effect on the input, not on the audit methodology itself.

**The results also depend on baseline strength:** the [EU-OPENSCREEN](https://zenodo.org/records/19347244) cross-dataset check (Section 6) agrees on the MT readout but, as expected, differs in how many endpoints are certified. The number of certifiable endpoints is sensitive to both sample size and to how strong the available cell-count baseline is in a given dataset. This is reported as a finding about the method's behaviour, not hidden: a dataset with fewer compounds or a cruder cell-count readout will certify fewer endpoints even if the same underlying biology is present, which is itself useful information for anyone applying this audit to a new dataset.

Results in Section 6 are hepatocyte- and dataset-specific. **What we claim generalizes is the *method*:** three matched models, a strong baseline, calibrated statistics, power-aware verdicts, FDR correction, not the specific 15 certified endpoints, which we expect to differ on another cell type, tissue, or assay panel.

## 8. Potential scientific and practical impact

Toxicity testing has a structural trust problem as it moves away from animals: a non-animal readout is only useful if the field can tell whether it is measuring the biology it claims to, and phenotypic imaging assays make that especially hard to verify, because thousands of correlated features can hide a single trivial driver. Cell count is the most dangerous such driver precisely because it is biologically real (dying or stressed cells really do leave the plate), and yet it is not the mechanistic information a toxicologist needs when deciding, say, whether a compound causes a specific hepatotoxic mode of action versus simply killing cells outright. Without a way to separate the two, a lab can spend months validating a phenotypic AI model that turns out to be an elaborate, expensive cell counter. **This is not a hypothetical:** [Seal et al. (2026)](https://doi.org/10.1038/s41467-026-68725-5) showed the same pattern in independent Cell Painting benchmarks, and our own early sanity check (Section 3) was run specifically because we needed to rule it out before trusting our own results.

**We did not build a better toxicity classifier.** Better classifiers already exist, and a marginally higher AUROC does not resolve the trust problem above. We built an ***audit***, deliberately narrow in scope: given any dataset with morphology profiles, cell counts, and toxicity labels, tell a team exactly which endpoints their morphology model can be trusted on, which it can't, and which it's too early to say anything about. That framing is what makes the tool reusable rather than a one-off analysis of one dataset: the same `cpsa.api.shortcut_audit` call runs on the OASIS hepatocyte data, the EU-OPENSCREEN HepG2 data, or a new dataset a lab generates tomorrow.

In immediate practical terms, **this is a pre-flight check, not a publication artefact:** a toxicology or pharma team building a phenotypic AI model can run this audit on their own screen before reporting or acting on a result, the same way a lab runs a positive/negative control before trusting an assay plate. Three concrete uses: **(1) a go/no-go check** before presenting an AI-toxicity model's accuracy numbers, so "the model predicts hepatotoxicity" is not quietly "the model counts cells"; **(2) a prioritization tool:** the per-endpoint verdict table tells a team which endpoints are worth the cost of full morphological profiling versus which are adequately (and more cheaply) served by a cell-count readout alone; **(3) a study-design tool** via the organ-on-chip planner, used *before* a chip study is run, not after, to size the compound set correctly.

Organ-on-chip systems are being built and funded explicitly as more human-relevant, animal-free alternatives to both animal testing and simple 2D plates. As these systems adopt AI-based imaging readouts (which they are actively doing), they will inherit the identical cell-count-confound risk this project targets, very possibly in a worse form, since a chip's own structural and flow artefacts can plausibly act as additional density-correlated confounds beyond cell count alone. Building this audit to be dataset-agnostic, and shipping the organ-on-chip planner alongside the 2D result, is a deliberate bet that the real value of this work is as infrastructure other groups' chip studies can adopt, not as a one-time finding about one hepatocyte dataset.

[FDA's 2026 draft guidance](https://www.federalregister.gov/d/2026-05390) on New Approach Methodologies explicitly asks for "**technical characterization**" and "**fit-for-purpose**" validation before a non-animal method informs a regulatory decision. A repeatable, dataset-agnostic shortcut audit is a concrete instance of that kind of technical characterization, the kind of artifact a validation package for an AI-based NAM would need to include, not a nice-to-have research exercise.

## 9. Cross-disciplinary collaboration: Biology × Machine Learning

This project was built by two people from different backgrounds, split along the line the problem itself demanded: Vrushti led the biology and data track (label provenance, the cytotoxicity-filter semantics, dataset QA), and Dhairya led the ML and statistics track (model architecture, calibration, the multiple-testing framework, and the demo). Neither half could have substituted for the other, and the project's central technical risk (Section 2) lives at the seam between them.

**The seam between the two tracks mattered most here:** the single most consequential fact in this entire project (that `*_info.hitcall` is a raw, unfiltered value and only the `*_binary` files carry the real, filtered ground-truth label) is a biology-provenance fact with no ML content at all, but if it had gone unnoticed, every downstream model, calibration, and verdict would have been computed against the wrong labels without any statistical test catching it. That fact surfaced from the biology side reading the dataset authors' own label-construction notebook line by line, not from a modelling review. Conversely, the decision to compare morphology against the full cell-count *dose-response curve* rather than a single number (the choice that keeps this audit honest) is a statistical-design decision with no biology content, made on the ML side specifically because a weak baseline would have made the project's headline claim meaningless regardless of how correct the labels were.

**The two tracks ran in parallel with a standing rule:** any finding that changed what a label meant, what a compound's identity implied, or what "enough data" meant biologically was raised to the other track the same day it was found, not batched for a later sync. Concretely, the raw-vs-filtered-hitcall correction above was flagged and fixed before a single model was trained on the wrong ground truth, not after. The biology side also ran an early, deliberately simple sanity check (Section 3's cell-count-only probe) before the full statistical pipeline existed, specifically to give the ML side a go/no-go signal on the data early enough to still change course.

**This pairing fits the problem, not just convenience:** a toxicologist alone could identify the cell-count confound but not build a calibrated, FDR-corrected statistical test for it at scale; a machine learning engineer alone could build that statistical machinery but would have no way to know that a label column's `0` sometimes means "tied duplicate record" rather than "confirmed non-toxic," or that the released cytotoxicity filter only covers 35% of raw records. The project's entire value proposition (a trustworthy audit, not just a working script) depends on both kinds of correctness holding simultaneously.

## 10. Reproduction instructions

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt && .venv/bin/pip install -e .
.venv/bin/streamlit run demo/app.py        # precomputed results are included in results/
```

To rebuild every result in this report from the raw public data rather than the included precomputed outputs, run the numbered scripts in `scripts/` in order (`01_download_zenodo_data.py` through `15_calibrate_verdicts.py`), or run `scripts/20_full_audit.sh` for the full three-image-representation audit in one call. All downloads are MD5-verified and all stochastic steps use fixed seeds, so a full rebuild reproduces the numbers in Section 6 exactly. No GPU is required; the full pipeline runs on a laptop CPU.

**Data access (no account or payment required):**

- Profiles: Zenodo [10.5281/zenodo.17067683](https://zenodo.org/records/17067683) (CC-BY 4.0).
- Labels and annotations: [github.com/jessica-ewald/2024_09_09_Axiom_OASIS](https://github.com/jessica-ewald/2024_09_09_Axiom_OASIS) (BSD-3-Clause).

## 11. Sources and licenses of external models, libraries, datasets, and tools

| Item | License / terms | Role in this project |
| --- | --- | --- |
| [Ewald et al. (OASIS Consortium), *Cell Systems*, 2026](https://doi.org/10.1016/j.cels.2026.101566): Cell Painting profiles, Zenodo [10.5281/zenodo.17067683](https://zenodo.org/records/17067683) | CC-BY 4.0 | Primary dataset: morphology profiles (CellProfiler, CP-CNN, DINOv2) and native LDH/MTT/cell-count readouts |
| [jessica-ewald/2024_09_09_Axiom_OASIS](https://github.com/jessica-ewald/2024_09_09_Axiom_OASIS) analysis repository | BSD-3-Clause | ToxCast/Tox21 label construction, cytotoxicity-filter logic |
| EU-OPENSCREEN HepG2 Cell Painting profiles, [10.5281/zenodo.19347244](https://zenodo.org/records/19347244) | CC-BY 4.0 | Independent second dataset for cross-dataset validation (Section 6) |
| Seal, Dee, Shah, Cerisier, Zhang, Miglietta, Titterton, Cabrera, Boiko, Beatson, Slabaugh, Taboureau, Puigvert, Singh, Spjuth, Bender, Carpenter. ["Counting cells can accurately predict small-molecule bioactivity benchmarks."](https://doi.org/10.1038/s41467-026-68725-5) *Nature Communications*, 2026 (also [bioRxiv 2025.04.27.650853](https://doi.org/10.1101/2025.04.27.650853)) | Journal publication, cited not redistributed | Motivates the cell-count-shortcut risk this audit is built to detect (Sections 2, 8) |
| FDA. [*General Considerations for the Use of New Approach Methodologies in Drug Development.*](https://www.federalregister.gov/d/2026-05390) Guidance for Industry, Draft, March 2026 | US government publication (public domain) | Regulatory framing for "technical characterization" and "fit-for-purpose" validation (Sections 2, 8) |
| [XGBoost](https://xgboost.readthedocs.io/) | Apache 2.0 | Classifier used for all three matched models (Section 4) |
| [scikit-learn](https://scikit-learn.org/) | BSD-3-Clause | Cross-validation, logistic-regression robustness check, calibration metrics |
| [Streamlit](https://streamlit.io/) | Apache 2.0 | Interactive demo application |
| [CellProfiler](https://cellprofiler.org/) | BSD-3-Clause | One of three morphology feature-extraction pipelines (features already computed and released by the dataset authors; not re-run here) |
| [DINOv2](https://github.com/facebookresearch/dinov2) (Meta AI) | Apache 2.0 | Self-supervised image-embedding representation (embeddings already computed and released by the dataset authors; not re-run here) |

No proprietary datasets, paid APIs, or non-redistributable third-party content are used anywhere in this project; every input and dependency above is either open-licensed, a public government publication, or a cited published finding used for motivation only, not reproduced.
