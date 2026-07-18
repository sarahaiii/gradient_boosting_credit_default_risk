# Credit Default Risk Report

Mode: Home Credit Kaggle data

## Model Results

| model | validation_auc | precision | recall | f1 |
| --- | --- | --- | --- | --- |
| logistic_regression_numpy | 0.7401 | 0.2175 | 0.4167 | 0.2858 |
| gradient_boosted_stumps_numpy | 0.6930 | 0.1784 | 0.4296 | 0.2521 |

Best model: `logistic_regression_numpy`
Cross-validation AUC: 0.6986 +/- 0.0048
Out-of-sample AUC: 0.7401
Selected threshold: 0.6430
Precision: 0.2175
Recall: 0.4167
F1-score: 0.2858

## Optional Library Availability

| library | available |
| --- | --- |
| sklearn | False |
| lightgbm | False |
| xgboost | False |
| catboost | False |
| optuna | False |
| shap | False |
| matplotlib | False |
| seaborn | False |

## Notes

- Drop the Kaggle CSVs into `data/raw` to run on real data.
- The fallback gradient boosted stump model exists so the project runs even without ML packages installed.
- Install `requirements.txt` to enable the full sklearn/LightGBM/XGBoost/CatBoost/SHAP workflow.