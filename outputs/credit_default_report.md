# Credit Default Risk Report

Mode: Home Credit Kaggle data

## Model Results

| model | validation_auc | precision | recall | f1 |
| --- | --- | --- | --- | --- |
| hist_gradient_boosting_sklearn | 0.7756 | 0.2688 | 0.4093 | 0.3245 |
| catboost | 0.7693 | 0.2527 | 0.4097 | 0.3126 |
| logistic_regression_sklearn | 0.7671 | 0.2609 | 0.3973 | 0.3150 |
| logistic_regression_numpy | 0.7641 | 0.2532 | 0.3855 | 0.3056 |
| random_forest_sklearn | 0.7497 | 0.2177 | 0.4598 | 0.2955 |
| gradient_boosted_stumps_numpy | 0.7184 | 0.2075 | 0.3771 | 0.2677 |

Best model: `hist_gradient_boosting_sklearn`
Cross-validation AUC: 0.7694 +/- 0.0027
Out-of-sample AUC: 0.7756
Selected threshold: 0.1640
Precision: 0.2688
Recall: 0.4093
F1-score: 0.3245

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
- Feature engineering includes credit/income ratios, annuity ratios, employment-age ratios, external-source aggregates, missing-value counts, bureau history aggregates, previous-application aggregates, and installment payment behavior aggregates.
- Hyperparameter tuning results are saved to `outputs/hyperparameter_tuning_results.csv`.
- scikit-learn and CatBoost models are used when installed; the NumPy stump model remains as a fallback.
- On macOS, LightGBM and XGBoost may require the native `libomp` runtime before they can import.