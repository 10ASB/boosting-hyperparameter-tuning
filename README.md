# Boosting & Hyperparameter Tuning — Delivery Delay Prediction

This repository contains the Day 8 practical artifacts for tuning an XGBoost model on a delivery-delay dataset, including randomized-search results and Optuna trial logs.

## Actual Day 8 files
- `internB_delivery_delays.csv` — delivery-delay dataset (1,800 rows, 17 columns).
- `xgb_randomized_search_results.csv` — 30 randomized-search configurations with five-fold cross-validation scores.
- `optuna_trials.csv` — Optuna trial log with sampled XGBoost hyperparameters and objective values.
- `day08_results_summary.md` — summary of the supplied result files and best observed trials.
- `day08_boosting_tuning.ipynb` — notebook to inspect the data and supplied tuning results, then reproduce the analysis when the training setup is available.
- `day08_lab_guide.md` — practical tasks and reporting checklist.

## Dataset overview
The supplied dataset contains order, product, delivery, courier, payment, rating, and delay fields. The target column is `is_delayed`; the provided data has 1,249 non-delayed and 551 delayed orders. The file contains missing values, including `customer_rating` and `product_weight_kg`, so preprocessing should be handled deliberately.

## Tuning results in the supplied files
- The best row in `xgb_randomized_search_results.csv` has a mean cross-validation score of approximately **0.5996** (rank 1). See the summary file for the recorded parameter values.
- The highest objective value in `optuna_trials.csv` is approximately **0.6095** (trial 1).

These values are the scores recorded in the supplied artifacts. The exact metric/objective definition and final held-out test score are not identifiable from these CSVs alone; check the original training code before labeling them as accuracy, F1, or another metric.

## Environment
The result files show an XGBoost tuning workflow. To reproduce training, install the packages used by the original notebook/script, for example:
```bash
pip install pandas numpy scikit-learn xgboost optuna jupyter
```

## Reproducibility and data safety
- Do not infer the target metric solely from a column name such as `value` or `mean_test_score`.
- Do not use the held-out test set for hyperparameter selection.
- Split data before fitting preprocessing steps to avoid leakage.
- Treat `order_id` as an identifier, not a predictive feature.
- Inspect unusual values (for example, extreme `distance_km`) before training.
