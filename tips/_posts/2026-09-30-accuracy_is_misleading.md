---
layout: post
title: "Why is 99% accuracy useless? Precision, recall and F1 explained"
description: >
  On data with 1% fraud, a model that never predicts fraud scores 99% accuracy. Tested with scikit-learn:
  what precision, recall and F1 show instead, and how the decision threshold trades one for the other.
image: /assets/img/tips/accuracy_is_misleading/cover.png
sitemap:
  changefreq: daily
  priority: 1.0
---

# Why is 99% accuracy useless? Precision, recall and F1 explained

**TL;DR** — When one class is rare, accuracy mostly measures how common the other class is.
On a test set with 1% fraud, a model that always says "not fraud" scored 0.990 accuracy and caught zero fraud.
Look at **recall** (how much of the rare class you catch) and **precision** (how many of your alarms are real) instead.

_Tested on 2026-09-28 with scikit-learn 1.9.1 on synthetic data (10,000 rows, 1% positive)._

## Key points

- **Accuracy** = correct predictions / all predictions. On imbalanced data, "always predict the majority" scores high.
- **Recall** = caught positives / all real positives. "Of the fraud that happened, how much did we catch?"
- **Precision** = real positives / everything we flagged. "Of our alarms, how many were real?"
- **F1** is the harmonic mean of precision and recall. It is high only when both are.
- **The threshold** (0.5 by default) trades precision for recall. Lower it to catch more, at the cost of more false alarms.

## Can a useless model really score 99%?

Yes. Here are three models on the same data, where 1% of transactions are fraud:

```python
from sklearn.datasets import make_classification
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

# 10,000 transactions, 1% fraud
X, y = make_classification(n_samples=10_000, n_features=20, n_informative=5,
                           weights=[0.99], flip_y=0, random_state=0)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
print(f"fraud share in test set: {y_te.mean():.1%} ({y_te.sum()} of {len(y_te)})")

models = {
    "always 'not fraud'": DummyClassifier(strategy="most_frequent"),
    "logistic regression": LogisticRegression(max_iter=1000),
    "logistic, balanced": LogisticRegression(max_iter=1000, class_weight="balanced"),
}
print(f"{'model':<21}{'accuracy':>9}{'precision':>10}{'recall':>8}{'F1':>6}")
for name, model in models.items():
    pred = model.fit(X_tr, y_tr).predict(X_te)
    print(f"{name:<21}{accuracy_score(y_te, pred):>9.3f}"
          f"{precision_score(y_te, pred, zero_division=0):>10.3f}"
          f"{recall_score(y_te, pred):>8.3f}{f1_score(y_te, pred):>6.3f}")
```

```
fraud share in test set: 1.0% (30 of 3000)
model                 accuracy precision  recall    F1
always 'not fraud'       0.990     0.000   0.000 0.000
logistic regression      0.989     0.000   0.000 0.000
logistic, balanced       0.834     0.050   0.867 0.095
```

![Grouped bar chart of accuracy, precision, recall and F1 for three models: always 'not fraud' and plain logistic regression both reach 0.99 accuracy with 0.00 on every other metric, while balanced logistic regression has 0.83 accuracy, 0.05 precision, 0.87 recall and 0.09 F1](/assets/img/tips/accuracy_is_misleading/metrics.png)

- The **dummy model** never predicts fraud: 0.990 accuracy, 0 recall.
- **Plain logistic regression** also ended up predicting no fraud at all: 0.989 accuracy, 0 recall. It "learned" that saying no is almost always right.
- **`class_weight="balanced"`** makes each fraud case count about 99 times more in training. Accuracy *drops* to 0.834, but the model now catches 26 of the 30 frauds (recall 0.867).

The model with the worst accuracy is the only useful one.

## What do precision and recall actually measure?

Every prediction falls into one of four boxes:

| | Predicted fraud | Predicted not fraud |
|---|---|---|
| **Actually fraud** | True positive (TP) | False negative (FN): missed |
| **Actually not fraud** | False positive (FP): false alarm | True negative (TN) |

- **Precision** = TP / (TP + FP): the share of alarms that are real.
- **Recall** = TP / (TP + FN): the share of real cases you caught.
- **Accuracy** = (TP + TN) / everything. With 99% negatives, TN dominates it.

The balanced model's precision of 0.050 means about 1 in 20 alarms is real fraud. Whether that is acceptable depends on the cost of a missed fraud versus the cost of checking a false alarm.

## How does the threshold change the result?

A classifier outputs a probability; the threshold turns it into yes or no. Lowering it flags more cases:

```python
# continues from the code above (X_tr, X_te, y_tr, y_te)
from sklearn.metrics import average_precision_score, precision_score, recall_score

proba = LogisticRegression(max_iter=1000).fit(X_tr, y_tr).predict_proba(X_te)[:, 1]
print(f"average precision (area under the PR curve): {average_precision_score(y_te, proba):.3f}")
print(f"{'threshold':>9}{'flagged':>9}{'precision':>10}{'recall':>8}")
for t in [0.5, 0.3, 0.1, 0.05, 0.02]:
    pred = proba >= t
    print(f"{t:>9}{pred.sum():>9}{precision_score(y_te, pred, zero_division=0):>10.3f}{recall_score(y_te, pred):>8.3f}")
```

```
average precision (area under the PR curve): 0.079
threshold  flagged precision  recall
      0.5        2     0.000   0.000
      0.3       10     0.100   0.033
      0.1       61     0.098   0.200
     0.05      124     0.081   0.333
     0.02      304     0.056   0.567
```

Going from 0.5 to 0.02, recall rises from 0 to 0.567 while precision stays low. The same plain model that looked useless at 0.5 catches more than half the fraud at 0.02.
This synthetic problem is hard (average precision 0.079), so none of the settings is great, but the trade-off is typical.

## Which metric should you use?

| Situation | Focus on |
|---|---|
| Missing a positive is expensive (fraud, disease screening) | Recall, then check precision is tolerable |
| False alarms are expensive (spam filter deleting real mail) | Precision |
| You need one number to compare models | F1, or average precision across thresholds |
| Classes are roughly balanced | Accuracy is fine, alongside the others |

Always report the share of the rare class next to any accuracy number. "99% accurate" means nothing without "and 1% of cases are positive".

## Sources

- [Metrics and scoring: quantifying the quality of predictions](https://scikit-learn.org/stable/modules/model_evaluation.html) — scikit-learn user guide, checked 2026-09-28
- [`average_precision_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html)
