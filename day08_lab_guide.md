# Day 8 Lab Guide — Boosting & Hyperparameter Tuning

## Tasks

### Task 1 — Bagging vs Boosting
Write a short comparison:
- How bagging trains models and combines predictions.
- How boosting trains learners sequentially, with later learners focusing on errors from earlier learners.
- Typical strengths and risks of each approach.

### Task 2 — Train boosting models
Using the provided notebook, train and compare:
- Decision Tree baseline
- AdaBoost
- Gradient Boosting
- HistGradientBoosting

Record accuracy, precision, recall, F1-score, ROC-AUC, and fit time. Explain why accuracy alone may not tell the full story.

### Task 3 — Hyperparameter tuning
Use RandomizedSearchCV with cross-validation to tune GradientBoostingClassifier. Report:
- Best parameters
- Best mean cross-validation score
- Final test metrics

Explain why the test set must not be used to select hyperparameters.

### Task 4 — Early stopping and extensions
- Describe the purpose of early stopping and how it can help control overfitting.
- Try XGBoost or LightGBM if installed.
- Optionally try Optuna for optimization and record the objective and trial results.

## Submission checklist
- [ ] Notebook runs from top to bottom.
- [ ] Data split is stratified and reproducible.
- [ ] Hyperparameter search uses training data only.
- [ ] A comparison table is included.
- [ ] Results are explained in your own words.
- [ ] Optional library experiments are clearly marked as optional.

## Suggested reflection questions
1. What is the key difference between bagging and boosting?
2. Which model had the best test F1-score in your run?
3. Did tuning improve cross-validation performance? Did that translate to test performance?
4. What signs could indicate overfitting?
5. Why should the held-out test set be evaluated only after model selection?
