"""Boosting and hyperparameter-tuning lab.

Install: pip install numpy pandas scikit-learn
"""
from time import perf_counter

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="target")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

models = {
    "Decision Tree": DecisionTreeClassifier(max_depth=4, random_state=42),
    "AdaBoost": AdaBoostClassifier(n_estimators=100, learning_rate=0.5, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.05, max_depth=2, random_state=42
    ),
    "HistGradientBoosting": HistGradientBoostingClassifier(
        max_iter=100, learning_rate=0.1, max_leaf_nodes=15, random_state=42
    ),
}

rows = []
for name, model in models.items():
    start = perf_counter()
    model.fit(X_train, y_train)
    fit_seconds = perf_counter() - start
    pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        score = model.predict_proba(X_test)[:, 1]
    else:
        score = model.decision_function(X_test)
    rows.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1": f1_score(y_test, pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, score),
        "Fit seconds": fit_seconds,
    })

print(pd.DataFrame(rows).sort_values("F1", ascending=False).to_string(index=False))

search = RandomizedSearchCV(
    GradientBoostingClassifier(random_state=42),
    param_distributions={
        "n_estimators": [50, 100, 150, 200],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "max_depth": [1, 2, 3],
        "subsample": [0.7, 0.85, 1.0],
    },
    n_iter=15,
    scoring="f1",
    cv=5,
    random_state=42,
    n_jobs=-1,
)
search.fit(X_train, y_train)
print("\nBest parameters:", search.best_params_)
print("Best mean CV F1:", round(search.best_score_, 4))

best = search.best_estimator_
pred = best.predict(X_test)
score = best.predict_proba(X_test)[:, 1]
print("Tuned test accuracy:", round(accuracy_score(y_test, pred), 4))
print("Tuned test precision:", round(precision_score(y_test, pred, zero_division=0), 4))
print("Tuned test recall:", round(recall_score(y_test, pred, zero_division=0), 4))
print("Tuned test F1:", round(f1_score(y_test, pred, zero_division=0), 4))
print("Tuned test ROC-AUC:", round(roc_auc_score(y_test, score), 4))

# Optional: try XGBoost / LightGBM after installing their packages.
# Optional: use Optuna to define and optimize a trial objective.
