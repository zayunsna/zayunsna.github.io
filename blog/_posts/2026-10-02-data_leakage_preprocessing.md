---
layout: post
title: "Why fit preprocessing on training data only? A data leakage demo"
description: >
  Fitting a scaler or a feature selector on all your data before cross-validation leaks test information.
  On pure noise, leaky feature selection scored 0.80 instead of 0.50. Tested with scikit-learn 1.9.1.
image: /assets/img/blog/data_leakage_preprocessing/cover.png
sitemap:
  changefreq: daily
  priority: 1.0
---

# Why fit preprocessing on training data only? A data leakage demo

**TL;DR** — Any step that learns from data (a scaler, an imputer, a feature selector) must be fitted on the
training data only. Otherwise your test score includes information the model won't have in production.
The simplest fix is to put every step in a scikit-learn `Pipeline` and cross-validate the whole pipeline.

_Tested on 2026-09-28 with scikit-learn 1.9.1, NumPy 2.5.3, Python 3.12._

## Key points

- **Data leakage** means the model uses information during training that it won't have at prediction time. It makes scores look better than they are.
- **The rule:** never call `fit` or `fit_transform` on data that includes your test rows.
- **How bad it gets depends on the step.** Scaling leaked almost nothing in our test. Feature selection turned pure noise into 80% accuracy.
- **`Pipeline` prevents it.** Cross-validation then refits every step on each training fold.

## What does "fit on training data only" mean?

A preprocessing step learns numbers from the data. `StandardScaler` learns each column's mean and standard deviation.
`SelectKBest` learns which columns are most related to the target.
If the test rows are included when those numbers are learned, the test is no longer an honest preview of new data.

| Step | What it learns in `fit` | Leakage risk if fitted on all data |
|---|---|---|
| `StandardScaler`, `MinMaxScaler` | Mean, std, min, max | Usually small |
| `SimpleImputer` | Mean or median per column | Usually small |
| `SelectKBest`, other feature selection | Which features relate to **the target** | **Large** |
| Target encoding | Average target per category | **Large** |

The steps that look at the target (`y`) are the dangerous ones.

## Does leaking a scaler really change the score?

In this test, no. Here is `StandardScaler` fitted on all data vs. inside a pipeline, on the breast cancer dataset:

```python
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)

# Wrong: scaler sees all rows, including the ones used for testing
X_scaled = StandardScaler().fit_transform(X)
leaky = cross_val_score(LogisticRegression(max_iter=1000), X_scaled, y, cv=5).mean()

# Right: scaler is refitted on the training folds only
pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
clean = cross_val_score(pipe, X, y, cv=5).mean()

print(f"scaler fitted on all data: {leaky:.4f}")
print(f"scaler inside pipeline:    {clean:.4f}")
```

```
scaler fitted on all data: 0.9807
scaler inside pipeline:    0.9807
```

Identical to four decimals. The mean and std of 569 rows barely change when you drop a fifth of them.
It is still the wrong habit, though. The same code pattern with a target-aware step is a different story.

## What happens when feature selection leaks?

It invents a signal that isn't there. The data below is pure noise: 5,000 random features and random labels.
No model can honestly beat 50% accuracy on it.

```python
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline

rng = np.random.default_rng(0)
X = rng.normal(size=(200, 5_000))   # 5,000 features of pure noise
y = rng.integers(0, 2, size=200)    # random labels: the true accuracy is 50%

# Wrong: pick the 20 "best" features using all rows, then cross-validate
X_selected = SelectKBest(f_classif, k=20).fit_transform(X, y)
leaky = cross_val_score(LogisticRegression(), X_selected, y, cv=5).mean()

# Right: feature selection happens inside each training fold
pipe = make_pipeline(SelectKBest(f_classif, k=20), LogisticRegression())
clean = cross_val_score(pipe, X, y, cv=5).mean()

print(f"selection on all data:     {leaky:.3f}")
print(f"selection inside pipeline: {clean:.3f}")
```

```
selection on all data:     0.795
selection inside pipeline: 0.510
```

With 5,000 noise columns, some will match the random labels by chance, including on the test rows.
When selection sees all rows, it picks exactly those columns, so the test folds look predictable.
Repeating the experiment on 20 random datasets gives the same picture:

```
leaky: mean 0.798, min 0.745, max 0.845
clean: mean 0.496, min 0.430, max 0.605
```

![Dot plot of 5-fold accuracy on 20 pure-noise datasets: feature selection on all data scores between 0.745 and 0.845 with a mean of 0.798, while feature selection inside a Pipeline scores between 0.430 and 0.605 with a mean of 0.496, around the 0.5 chance line](/assets/img/blog/data_leakage_preprocessing/leakage_repeated.png)

Every one of the 20 leaky runs scored at least 0.745 on data with no signal at all.

## How do you prevent leakage in practice?

- **Split first.** Separate the test set before any preprocessing.
- **Use a `Pipeline`.** Put every step that has a `fit` method inside it, then pass the pipeline to `cross_val_score` or `GridSearchCV`.
- **Use `transform`, not `fit_transform`, on new data.** Validation, test, and production data reuse the numbers learned from training.
- **Be suspicious of great results.** If a score looks too good for the problem, check what the preprocessing saw.

## Sources

- [Common pitfalls and recommended practices: Data leakage](https://scikit-learn.org/stable/common_pitfalls.html) — scikit-learn docs, checked 2026-09-28

Related: [Min-max scaling vs standardization: when should you use which?](/blog/2026-09-29-minmax_vs_standard_scaler/)
