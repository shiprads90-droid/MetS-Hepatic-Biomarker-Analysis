# Reproducibility code: hepatic biomarkers and metabolic syndrome

This package reproduces every numerical analysis and scientific figure reported in the revised manuscript, **“Early Identification of Metabolic Syndrome among Young Adults Using Hepatic Biomarkers: Insights from Conventional to Machine Learning Models.”**

## Data privacy

Participant-level data are not included. Place the de-identified source workbook in a local directory and pass its path with `--input`. The expected worksheet contains 18 columns in this order:

`ID, Sex, Age, Height, Weight, BMI, WC, SBP, DBP, FBG, TG, HDL, stored MetS status, ALT, AST, GGT, stored FLI, stored HSI`.

Sex must be coded `F` or `M`. Units are cm, kg, kg/m², mmHg, mg/dL, and U/L as applicable.

## Installation

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Run the complete analysis

```bash
python analysis.py --input "path/to/deidentified_data.xlsx" --sheet Data --output results --seed 20260907
python make_figures.py --results results --output figures
```

The analysis uses five-fold stratified outer cross-validation and four-fold stratified inner cross-validation. Preprocessing and hyperparameter tuning occur inside the resampling loop. A fixed seed is used for reproducibility.

No class weighting is used in the reported final models; preliminary weighted fits worsened probability calibration. Class imbalance is handled through stratified inner and outer folds, and performance is reported using sensitivity, specificity, PPV, NPV, AUC, Brier score, and confusion matrices rather than accuracy alone.

## Outcome and index definitions

- Conventional MetS: any three of elevated waist circumference, triglycerides, blood pressure, fasting glucose, and reduced HDL-C.
- MetS*: at least two of elevated blood pressure, fasting glucose, and reduced HDL-C. This sensitivity outcome excludes waist circumference and triglycerides to prevent direct overlap with FLI.
- FLI: published Bedogni equation using log triglycerides, BMI, log GGT, and waist circumference.
- HSI: `8 × ALT/AST + BMI + 2 if female`. The diabetes increment is set to zero because diabetes status is absent and diagnosed cardiometabolic disease was an exclusion criterion. This choice must remain disclosed in the manuscript.

## Output-to-manuscript map

| Output | Manuscript result reproduced |
|---|---|
| `audit.csv` | source N, exclusions, missingness, duplicate IDs, stored/recalculated disagreements, FLI/HSI formula audit |
| `processed.csv` | component criteria, recalculated indices, MetS and MetS* outcomes |
| `sex_descriptive.csv` | Tables 1–2: sex-specific MetS/non-MetS comparisons, Welch tests, SMDs, FDR q values |
| `correlation.csv` | Figure 1 correlation matrix |
| `interaction.csv` | Figure 2 formal age-adjusted marker-by-sex interaction tests |
| `adjusted_associations.csv` | Table 5: adjusted odds ratios per one-SD increase in each marker |
| `roc.csv` | Figure 3 and Supplementary Tables 4A–4C: AUCs, bootstrap CIs, thresholds, sensitivity, specificity, PPV, NPV, accuracy |
| `delong.csv` | formal pairwise AUC comparisons with ALT |
| `models.csv` | Table 6: nested cross-validated AUC, Brier score, calibration, threshold metrics |
| `folds.csv` | fold-specific tuned hyperparameters and inner-CV performance |
| `confusion_matrices.csv` | TN, FP, FN, TP, and accuracy for every final out-of-fold model |
| `incremental.csv` | Table 7: ΔAUC, continuous NRI, and IDI with bootstrap CIs |
| `permutation.csv` | Figure 4 cross-validated permutation importance |
| `oof_predictions.csv` | out-of-fold probabilities used for calibration and decision analysis |
| `dca.csv` | Figure 5 decision-curve net benefit |
| `all_results.xlsx` | consolidated workbook containing all numerical tables |

## Important analytic cautions

FLI evaluated against conventional MetS is partly circular because waist circumference and triglycerides contribute to both. The MetS* analysis is the principal leakage-controlled sensitivity analysis. MetS* is an analytic construct, not a validated clinical diagnosis. All model performance is internally validated and requires prospective external validation.

## Zenodo deposition

Upload this directory as a versioned release to Zenodo. After Zenodo issues a DOI, replace `[INSERT DOI URL]` in `CODE_AVAILABILITY_TEMPLATE.txt` and paste that statement into the manuscript under the heading **Code Availability**.
