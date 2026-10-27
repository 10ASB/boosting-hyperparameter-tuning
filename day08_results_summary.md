# Day 8 — Results Summary

## Dataset
- Rows: 1,800
- Columns: 17
- Target: `is_delayed`
- Target counts: `0 = 1,249`, `1 = 551`
- Missing values noted in supplied dataset: `customer_rating = 613`, `product_weight_kg = 295`

## XGBoost randomized search
The supplied `xgb_randomized_search_results.csv` contains 30 parameter configurations and five-fold test scores per configuration.

Best recorded row:
- Rank: 1
- Mean cross-validation score: **0.5996065594**
- Standard deviation: **0.0307** (read from the corresponding rank-1 row when reporting; verify from the CSV)
- `colsample_bytree`: approximately 0.9195
- `learning_rate`: approximately 0.02388
- `max_depth`: 7
- `min_child_weight`: 6
- `n_estimators`: 504
- `subsample`: approximately 0.8055

## Optuna
The highest value in the supplied `optuna_trials.csv` is **0.60948305496**, trial **1**. Its parameter values are recorded in that CSV.

## Interpretation caveat
The CSV files do not define the scoring metric/objective in a human-readable field, and they do not provide a final held-out test score. Confirm the original training code before describing these values as accuracy, F1, or another metric. Do not compare the two best values as if they were directly equivalent unless the objective and validation protocol match.
