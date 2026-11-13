---
hub: leakage
layout: post
title: "Why fit preprocessing on training data only? A data leakage demo"
description: >
  Fitting a scaler or a feature selector on all your data before cross-validation leaks test information.
  On pure noise, leaky feature selection scored 0.80 instead of 0.50. Tested with scikit-learn 1.9.1.
image: /assets/img/blog/data_leakage_preprocessing/cover.png
lastmod: 2026-11-13
last_modified_at: 2026-11-13
sitemap:
  changefreq: daily
  priority: 1.0
---

# Why fit preprocessing on training data only? A data leakage demo

Any step that learns from data (a scaler, an imputer, a feature selector) has to be fitted on the training rows
only. If the test rows are included, the cross-validation score uses information the model won't have in
production. Putting every step in a scikit-learn `Pipeline` and cross-validating the whole pipeline handles this
for you, because each fold refits every step on its own training rows.

scikit-learn's [Common pitfalls page](https://scikit-learn.org/stable/common_pitfalls.html) (checked 2026-11-13,
version 1.9.1) shows the effect with one setting: 200 rows, 10,000 noise features, 0.76 accuracy where the truth
is 0.5. How much a leak costs depends on the step and on the shape of the data. In the runs below, a leaky scaler
changed nothing, leaky feature selection scored 0.80 on 200 rows of pure noise, and on 5,000 rows the same mistake
still added 0.07.

The examples ran on 2026-09-28 with scikit-learn 1.9.1, NumPy 2.5.3 and Python 3.12; the data-shape table was
added on 2026-11-13 with the same versions.

## What does "fit on training data only" mean?

A preprocessing step learns numbers from the data. `StandardScaler` learns each column's mean and standard deviation.
`SelectKBest` learns which columns are most related to the target.
If the test rows are included when those numbers are learned, the test is no longer an honest preview of new data.
Data leakage is the general name for this: the model trains on information it won't have at prediction time, and
the score looks better than it is.

| Step | What it learns in `fit` | Leakage risk if fitted on all data |
|---|---|---|
| `StandardScaler`, `MinMaxScaler` | Mean, std, min, max | Usually small |
| `SimpleImputer` | Mean or median per column | Usually small |
| `SelectKBest`, other feature selection | Which features relate to **the target** | **Large** |
| Target encoding | Average target per category | **Large** |

The steps that look at the target (`y`) are the dangerous ones.

## StandardScaler on all rows scores 0.9807 both ways

Here is `StandardScaler` fitted on all data vs. inside a pipeline, on the breast cancer dataset:

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

## Leaky feature selection finds signal in pure noise

The data below is pure noise: 5,000 random features and random labels.
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

## More candidate features and fewer rows make the leak bigger

The same experiment, 20 random datasets per setting (`k=20`, `LogisticRegression`, 5-fold CV), with the number of
noise features and rows changed one at a time. The 200-row, 5,000-feature row is the run above.
The [script](/assets/code/data_leakage_preprocessing/leak_vs_shape.py) and its
[raw output](/assets/code/data_leakage_preprocessing/leak_vs_shape_output.txt) are in the post's code folder.

| Rows | Noise features | Leaky mean (min) | Pipeline mean | Gap |
|---|---|---|---|---|
| 200 | 20 | 0.522 (0.415) | 0.522 | +0.000 |
| 200 | 100 | 0.640 (0.595) | 0.498 | +0.142 |
| 200 | 1,000 | 0.746 (0.695) | 0.501 | +0.245 |
| 200 | 5,000 | 0.798 (0.745) | 0.496 | +0.302 |
| 200 | 20,000 | 0.833 (0.780) | 0.522 | +0.312 |
| 100 | 5,000 | 0.877 (0.820) | 0.500 | +0.377 |
| 1,000 | 5,000 | 0.656 (0.627) | 0.507 | +0.150 |
| 5,000 | 5,000 | 0.570 (0.562) | 0.500 | +0.070 |

With 20 features and `k=20`, the selector keeps every column, so it has nothing to choose and the two scores match.
From there the gap grows with each extra pool of candidates: more noise columns means more of them line up with the
labels by chance. More rows shrink it, because chance correlations get weaker, but even at 5,000 rows every leaky
run stayed above 0.56 on data with no signal. Wide, short tables (many one-hot or text columns, few labelled rows)
are where this mistake inflates a score the most.

These numbers come from pure noise with one model and `k=20`. On real data the true score is unknown, so the same
leak shows up only as a cross-validation score that later fails to hold on new data.

## Habits that keep test rows out of fit

- **Split first.** Separate the test set before any preprocessing.
- **Use a `Pipeline`.** Put every step that has a `fit` method inside it, then pass the pipeline to `cross_val_score` or `GridSearchCV`.
- **Call only `transform` on new data.** Validation, test, and production data reuse the numbers learned from training.
- **Be suspicious of great results.** If a score looks too good for the problem, check what the preprocessing saw.

Once the pipeline is in place, choosing the scaler itself is the next question; see
[min-max scaling vs standardization](/blog/2026-09-29-minmax_vs_standard_scaler/).
