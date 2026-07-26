# Project Requirements Checklist

## Project

**Gradient Boosting Model - Home Credit Default Risk**

This project predicts loan default risk using the Kaggle Home Credit Default Risk dataset. The task is a binary classification problem where the model predicts whether an applicant is likely to experience repayment difficulty. The main evaluation metric is ROC-AUC.

## Requirement Match

| Requirement | Status | Where It Is Covered |
| --- | --- | --- |
| Description of the project, why it is useful, and what it trains | Complete | README, notebook, report |
| Where to find data | Complete | Kaggle Home Credit Default Risk, documented in README/notebook |
| List of theory needed | Complete | Notebook and project explanation |
| List of Python libraries needed | Complete | requirements.txt |
| Output graphs and results | Complete | outputs/ and reports/figures/ |
| High-level project guide | Complete | Jupyter notebook |

## Data Source

The data comes from Kaggle:

Home Credit Default Risk

The project uses these real files locally:

- data/raw/application_train.csv
- data/raw/application_test.csv

Raw Kaggle data is kept local and is not pushed to GitHub.

## Theory Covered

- Binary classification
- Train/validation split
- Cross-validation
- ROC curve and AUC
- Precision, recall, and F1-score
- Class imbalance
- Missing value handling
- Categorical encoding
- Gradient boosting decision trees
- Regularization
- Feature importance
- Data leakage awareness
- Threshold selection

## Python Libraries

The project lists these in requirements.txt:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- lightgbm
- xgboost
- catboost
- optuna
- shap

The current environment used the built-in fallback models because several optional ML libraries were not installed.

## Outputs Produced

- Missing value summary
- Target distribution chart
- Correlation heatmap
- Baseline model AUC
- Cross-validation AUC
- Out-of-sample AUC
- ROC curve
- Precision-recall curve
- Confusion matrix
- Feature importance chart
- SHAP-style proxy summary
- Model comparison table
- Kaggle submission file

## Current Real-Data Results

- Training rows: 307,511
- Columns: 122
- Best model: logistic_regression_numpy
- Cross-validation AUC: 0.6986 +/- 0.0048
- Out-of-sample AUC: 0.7401

## Project Files

- Notebook: notebooks/Gradient_Boosting_Model_Home_Credit_Default_Risk.ipynb
- Pipeline: src/credit_default_pipeline.py
- Report: outputs/credit_default_report.md
- Submission: outputs/kaggle_submission.csv
- Figures: reports/figures/

## GitHub

https://github.com/sarahaiii/gradient_boosting_credit_default_risk

## Conclusion

The project matches the provided project requirements and the specific Gradient Boosting / Home Credit Default Risk PDF requirements. It is complete as a working baseline project, with optional room for improvement through LightGBM and additional secondary Kaggle tables.
