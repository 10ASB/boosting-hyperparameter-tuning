# Day 8 Lab Guide — Boosting & Hyperparameter Tuning

## Actual work artifacts
This lab is based on the supplied delivery-delay dataset and tuning logs:
- `internB_delivery_delays.csv`
- `xgb_randomized_search_results.csv`
- `optuna_trials.csv`

The target column in the dataset is `is_delayed`. The dataset contains 1,800 rows and 17 columns, with 1,249 rows labelled 0 and 551 labelled 1.

## Task 1 — Inspect the dataset
1. Load the CSV and inspect shape, column types, missing values, and target balance.
2. Identify identifier columns and fields that might cause leakage.
3. Investigate suspicious or extreme values, particularly in `distance_km`.
4. Document a preprocessing strategy for missing numeric and categorical values.

## Task 2 — Review randomized XGBoost search
1. Load `xgb_randomized_search_results.csv`.
2. Sort by `rank_test_score` and inspect the best configuration.
3. Compare `mean_test_score`, `std_test_score`, and the five fold scores.
4. Explain why cross-validation scores vary between folds.

In the supplied file, rank 1 has a mean cross-validation score of approximately 0.5996. The CSV does not identify the scoring metric by itself; use the original training code to name the metric accurately.

## Task 3 — Review Optuna trials
1. Load `optuna_trials.csv`.
2. Compare objective values across trials and inspect each trial's parameters.
3. Identify the highest recorded value (approximately 0.6095, trial 1).
4. Explain how an optimization objective and search space determine what Optuna learns.

The column `value` alone does not reveal which metric was optimized.

## Task 4 — Compare and report
Create a concise table of the best randomized-search and Optuna results. Do not claim one method is better unless both trials optimize the same metric on comparable validation splits.

## Submission checklist
- [ ] Dataset dimensions, target distribution, and missing values documented.
- [ ] Data quality issues inspected and preprocessing decisions justified.
- [ ] Best randomized-search row and fold variation reported.
- [ ] Best Optuna trial and parameter values reported.
- [ ] Metric/objective definition verified from the original training code.
- [ ] No unsupported claims about test performance.
