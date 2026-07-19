# Credit Default Risk Report

Mode: Home Credit Kaggle data

## Model Results

| model | validation_auc | precision | recall | f1 |
| --- | --- | --- | --- | --- |
| hist_gradient_boosting_sklearn | 0.7594 | 0.2341 | 0.4484 | 0.3076 |
| catboost | 0.7536 | 0.2398 | 0.4123 | 0.3032 |
| logistic_regression_sklearn | 0.7456 | 0.2207 | 0.4444 | 0.2949 |
| logistic_regression_numpy | 0.7438 | 0.2183 | 0.4396 | 0.2917 |
| random_forest_sklearn | 0.7380 | 0.2087 | 0.4408 | 0.2833 |
| gradient_boosted_stumps_numpy | 0.7146 | 0.1901 | 0.4388 | 0.2653 |

Best model: `hist_gradient_boosting_sklearn`
Cross-validation AUC: 0.7476 +/- 0.0041
Out-of-sample AUC: 0.7594
Selected threshold: 0.1388
Precision: 0.2341
Recall: 0.4484
F1-score: 0.3076

## Optional Library Availability

| library | available |
| --- | --- |
| sklearn | True |
| lightgbm | False |
| xgboost | False |
| catboost | True |
| optuna | False |
| shap | False |
| matplotlib | True |
| seaborn | False |

## Notes

- The real Home Credit Kaggle files are loaded from `data/raw` when present.
- Class weights are used to handle the imbalanced default target.
- Feature engineering includes credit/income ratios, annuity ratios, employment-age ratios, external-source aggregates, and missing-value counts.
- Hyperparameter tuning results are saved to `outputs/hyperparameter_tuning_results.csv`.
- scikit-learn and CatBoost models are used when installed; the NumPy stump model remains as a fallback.
- On macOS, LightGBM and XGBoost may require the native `libomp` runtime before they can import.