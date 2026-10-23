"""Choosing a decision threshold from error costs: default 0.5 vs cost theory vs TunedThresholdClassifierCV.

Data: OpenML "credit-g" v1 (UCI Statlog German Credit, 1,000 applicants, 30% bad), which ships with a cost matrix:
approving a bad applicant costs 5, rejecting a good one costs 1. Positive class = "bad".
Run: python threshold.py
"""
import sys

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, make_scorer
from sklearn.model_selection import FixedThresholdClassifier, StratifiedShuffleSplit, TunedThresholdClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

COST_FN, COST_FP = 5, 1          # bad approved as good, good rejected as bad
N_SPLITS = 20


def cost_per_applicant(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return (COST_FN * fn + COST_FP * fp) / len(y_true)


def load():
    df = fetch_openml("credit-g", version=1, as_frame=True).frame
    y = (df.pop("class") == "bad").astype(int)
    # This is a cost-matrix demo, not a lending model: leave out sex/marital status and foreign-worker status.
    X = df.drop(columns=["personal_status", "foreign_worker"])
    return X, y


def model():
    pre = make_column_transformer(
        (OneHotEncoder(handle_unknown="ignore"), make_column_selector(dtype_include="category")),
        (StandardScaler(), make_column_selector(dtype_include="number")))
    return make_pipeline(pre, LogisticRegression(max_iter=1000))


def main():
    print(f"python {sys.version.split()[0]}, scikit-learn {sklearn.__version__}, pandas {pd.__version__}")
    X, y = load()
    print(f"rows {len(y)}, bad {y.sum()} ({y.mean():.0%}), features {X.shape[1]}")
    theory = COST_FP / (COST_FP + COST_FN)
    print(f"cost-optimal threshold for calibrated probabilities: {COST_FP}/({COST_FP}+{COST_FN}) = {theory:.3f}")

    cost_scorer = make_scorer(cost_per_applicant, greater_is_better=False)
    grid = np.linspace(0.01, 0.99, 99)
    rows, chosen_cv, chosen_test, curves = [], [], [], []
    splits = StratifiedShuffleSplit(n_splits=N_SPLITS, test_size=0.3, random_state=0)
    for train, test in splits.split(X, y):
        Xtr, Xte, ytr, yte = X.iloc[train], X.iloc[test], y.iloc[train], y.iloc[test]
        base = model().fit(Xtr, ytr)
        p = base.predict_proba(Xte)[:, 1]

        tuned = TunedThresholdClassifierCV(model(), scoring=cost_scorer, cv=5).fit(Xtr, ytr)
        fixed = FixedThresholdClassifier(model(), threshold=theory).fit(Xtr, ytr)
        test_costs = [cost_per_applicant(yte, (p >= t).astype(int)) for t in grid]   # peeks at the test labels
        chosen_cv.append(tuned.best_threshold_)
        chosen_test.append(grid[int(np.argmin(test_costs))])
        curves.append(test_costs)

        preds = {"default 0.5": base.predict(Xte),
                 f"cost theory {theory:.3f}": fixed.predict(Xte),
                 "TunedThresholdClassifierCV": tuned.predict(Xte),
                 "all rejected": np.ones_like(yte)}
        for name, pred in preds.items():
            tn, fp, fn, tp = confusion_matrix(yte, pred, labels=[0, 1]).ravel()
            rows.append({"strategy": name, "cost": cost_per_applicant(yte, pred),
                         "bad caught": tp / (tp + fn), "good rejected": fp / (fp + tn)})
        rows.append({"strategy": "tuned on the test set (leak)", "cost": min(test_costs),
                     "bad caught": np.nan, "good rejected": np.nan})

    res = pd.DataFrame(rows).groupby("strategy", sort=False)
    print(f"\nTest cost per applicant over {N_SPLITS} stratified 70/30 splits (lower is better)")
    print(f"  {'strategy':30s} {'mean':>6} {'min':>6} {'max':>6} {'bad caught':>11} {'good rejected':>14}")
    for name, g in res:
        rates = "" if g["bad caught"].isna().all() else f"{g['bad caught'].mean():11.0%} {g['good rejected'].mean():14.0%}"
        print(f"  {name:30s} {g.cost.mean():6.3f} {g.cost.min():6.3f} {g.cost.max():6.3f} {rates}")

    print("\nThresholds chosen per split")
    for name, t in [("TunedThresholdClassifierCV", chosen_cv), ("tuned on the test set", chosen_test)]:
        t = np.array(t)
        print(f"  {name:28s} mean {t.mean():.3f}, range {t.min():.3f} to {t.max():.3f}")

    # Average test cost at every threshold over the same splits, for the chart.
    curve = np.mean(curves, axis=0)
    pd.DataFrame({"threshold": grid, "cost": curve}).to_csv("threshold_curve.csv", index=False)
    best = int(np.argmin(curve))
    print(f"\nAverage cost curve: at 0.50 {curve[49]:.3f}, at 0.17 {curve[16]:.3f}, "
          f"lowest {curve[best]:.3f} at {grid[best]:.2f}")


if __name__ == "__main__":
    main()
