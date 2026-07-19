# Gradient Boosting Model - Home Credit Default Risk

## Project Description

This project builds a supervised machine learning model to predict loan default risk using the Home Credit Default Risk dataset from Kaggle. The goal is to estimate the probability that a loan applicant will have repayment difficulty.

The target variable is `TARGET`, where `0` means the client did not default and `1` means the client had payment difficulty. Since this is a binary classification problem with an imbalanced target, the main evaluation metric is ROC-AUC.

This project is useful because credit default prediction is a common real-world finance and risk-management problem. It trains students to work with messy tabular data, missing values, categorical variables, class imbalance, model validation, AUC evaluation, and feature importance. It is also a strong CV project because it combines finance, machine learning, and practical data engineering.

## Where to Find the Data

The data comes from Kaggle:

Home Credit Default Risk

https://www.kaggle.com/competitions/home-credit-default-risk/data

The main files used in this project are:

- `application_train.csv`
- `application_test.csv`

The project currently uses the real Kaggle files locally in:

- `data/raw/application_train.csv`
- `data/raw/application_test.csv`

The raw Kaggle data is kept local and is not pushed to GitHub.

## Theory Needed

Students should understand the following concepts before or while completing the project:

- Binary classification
- Train/validation/test split
- Cross-validation
- ROC curve
- AUC
- Precision
- Recall
- F1-score
- Confusion matrix
- Class imbalance
- Missing value handling
- Categorical encoding
- Feature engineering
- Gradient boosting decision trees
- Logistic regression baseline models
- Regularization
- Feature importance
- SHAP values or model explanation methods
- Data leakage
- Threshold selection
- Kaggle submission formatting

## Python Libraries Needed

The project lists the recommended libraries in `requirements.txt`:

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

The current version also includes built-in fallback models so the project can run even if some optional machine learning libraries are not installed.

## Expected Output Graphs and Results

The project produces the following outputs:

- Missing value summary
- Target distribution chart
- Correlation heatmap for selected variables
- Baseline model AUC
- Cross-validation AUC
- Out-of-sample AUC
- ROC curve
- Precision-recall curve
- Confusion matrix
- Model comparison table
- Hyperparameter tuning results
- Feature importance table
- Feature importance chart
- SHAP-style proxy summary
- Kaggle submission file

The main output files are:

- `outputs/credit_default_report.md`
- `outputs/model_comparison.csv`
- `outputs/missing_value_summary.csv`
- `outputs/feature_importance.csv`
- `outputs/kaggle_submission.csv`
- `reports/figures/roc_curve.svg`
- `reports/figures/precision_recall_curve.svg`
- `reports/figures/confusion_matrix.svg`
- `reports/figures/correlation_heatmap.svg`
- `reports/figures/feature_importance.svg`

## Current Real-Data Results

The project has been run on the real Home Credit Kaggle data.

Current results:

- Training rows: 307,511
- Columns: 122
- Best model: hist_gradient_boosting_sklearn
- Cross-validation AUC: 0.7476 +/- 0.0041
- Out-of-sample AUC: 0.7594
- Selected threshold: 0.1388
- Precision: 0.2341
- Recall: 0.4484
- F1-score: 0.3076

The PDF does not specify a required minimum AUC. It requires AUC to be used as the main evaluation metric and reported clearly.

## High-Level Project Guide

1. Import the data from Kaggle.

2. Inspect the target variable and check whether the default classes are imbalanced.

3. Create a missing value summary to identify variables with many missing observations.

4. Clean known anomalies, such as unusual placeholder values in employment-day fields.

5. Engineer useful financial ratio features, such as credit-to-income and annuity-to-income ratios.

6. Separate numeric and categorical variables.

7. Fill missing numeric values with medians and missing categorical values with common categories.

8. Encode categorical variables so they can be used by machine learning models.

9. Split the data into training and validation sets.

10. Train a baseline model, such as logistic regression.

11. Train and compare a gradient boosting model or fallback boosted model.

12. Use cross-validation to check whether model performance is stable across different folds.

13. Evaluate the model with AUC, ROC curve, precision-recall curve, confusion matrix, precision, recall, and F1-score.

14. Review feature importance to understand which variables are most predictive.

15. Generate predictions for `application_test.csv`.

16. Create a Kaggle submission file with `SK_ID_CURR` and predicted `TARGET` probabilities.

17. Summarize the results in the report and notebook.

## Things to Look At

Students should think about the following questions while working on or presenting the project:

- Which features are most predictive of default risk?
- How severe is the target imbalance?
- Does the model perform consistently across validation folds?
- How does logistic regression compare with gradient boosting?
- Which threshold gives a reasonable tradeoff between precision and recall?
- Are any variables likely to leak future information?
- Which customers are most often misclassified?
- Would adding secondary tables improve AUC?
- How could LightGBM, XGBoost, or CatBoost improve performance?

## Main Project Files

- Jupyter notebook: `notebooks/Gradient_Boosting_Model_Home_Credit_Default_Risk.ipynb`
- Python pipeline: `src/credit_default_pipeline.py`
- Report: `outputs/credit_default_report.md`
- Kaggle submission: `outputs/kaggle_submission.csv`
- Figures: `reports/figures/`

## GitHub Repository

https://github.com/sarahaiii/gradient_boosting_credit_default_risk

## Summary

This project satisfies the requested project format. It includes the project description, data source, theory, Python libraries, output graphs and results, and a high-level guide. It also includes a working notebook, a Python pipeline, real-data model results, and a Kaggle submission file.
