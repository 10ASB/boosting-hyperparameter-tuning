# Boosting & Hyperparameter Tuning

A practical lab on ensemble boosting methods, systematic hyperparameter search, and fair model comparison.

## Learning objectives
- Explain the difference between bagging and boosting.
- Train AdaBoost, Gradient Boosting, and HistGradientBoosting classifiers.
- Tune model hyperparameters with cross-validation.
- Compare models using accuracy, precision, recall, F1-score, ROC-AUC, and training time.
- Understand early stopping and why evaluation data must not leak into training.
- Explore XGBoost and LightGBM as optional libraries.

## Contents
- `day08_boosting_tuning.ipynb` — end-to-end notebook.
- `day08_boosting_tuning.py` — script version of the core experiment.
- `day08_lab_guide.md` — tasks, concepts, and submission checklist.

## Dataset
The notebook uses scikit-learn's built-in Breast Cancer Wisconsin dataset, so no external CSV download is needed. It is a binary classification teaching dataset. The notebook uses a stratified train/test split and performs hyperparameter search only on the training data.

## Quick start
Install dependencies:
```bash
pip install numpy pandas matplotlib scikit-learn
```
Then run the notebook cells from top to bottom. XGBoost, LightGBM, and Optuna are optional; install them separately only if you want to try those extensions.

## Core workflow
1. Load and split the data.
2. Establish a baseline with a decision tree.
3. Train AdaBoost, Gradient Boosting, and HistGradientBoosting.
4. Tune selected parameters with `RandomizedSearchCV`.
5. Evaluate the chosen estimator once on the held-out test set.
6. Compare scores and discuss trade-offs.

## Important notes
- Cross-validation is performed on the training split; the test split is reserved for final evaluation.
- Scores are dataset- and split-dependent. Do not copy expected scores without running the experiment.
- This is a learning lab, not a claim that one model is always best.
