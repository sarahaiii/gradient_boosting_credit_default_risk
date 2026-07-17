# Credit Default Risk - Gradient Boosting Project

This project follows the Home Credit Default Risk assignment spec: predict loan default with an AUC-focused supervised machine learning pipeline.

## What It Produces

- Missing value summary
- Target distribution chart
- Selected-feature correlation heatmap
- Baseline model AUC
- Cross-validation AUC
- Out-of-sample AUC
- ROC curve
- Precision-recall curve
- Confusion matrix
- Feature importance chart
- SHAP summary plot when `shap` is installed, otherwise a permutation-importance proxy
- Model comparison table
- Kaggle submission file

## Data Setup

Download the Home Credit Default Risk files from Kaggle and place them here:

```text
data/raw/application_train.csv
data/raw/application_test.csv
```

The pipeline starts with `application_train.csv`, as requested in the spec. If `application_test.csv` is present, it creates a Kaggle-ready submission.

## Run

Notebook-first:

```bash
jupyter notebook notebooks/Gradient_Boosting_Model_Home_Credit_Default_Risk.ipynb
```

Script version:
```bash
python3 src/credit_default_pipeline.py
```

If the Kaggle CSVs are not present, the script runs in synthetic demo mode so the workflow can still be tested end to end.

## Outputs

Key files are written to:

```text
outputs/
reports/figures/
models/
```

The main written report is:

```text
outputs/credit_default_report.md
```
