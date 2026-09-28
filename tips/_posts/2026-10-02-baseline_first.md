---
layout: post
title: "Why you should always build a baseline model first"
description: >
  Before training a real model, measure the simplest possible one. On a price-like series, "tomorrow = today"
  beat a random forest by more than 3×. Tested with scikit-learn, with the reason the forest failed.
image: /assets/img/tips/baseline_first/cover.png
sitemap:
  changefreq: daily
  priority: 1.0
---

# Why you should always build a baseline model first

**TL;DR** — A baseline is the simplest prediction you could make without machine learning, such as "tomorrow equals today"
or "always predict the average". Measure it first. If your model can't beat it clearly, the model isn't learning anything useful.
In our test, the naive baseline scored an error of 0.79; a random forest scored 2.55.

_Tested on 2026-09-28 with scikit-learn 1.9.1 on a synthetic random walk (1,200 days)._

## Key points

- **A baseline** answers "how good is good?" Without it, an error of 2.55 means nothing.
- **Pick a baseline that fits the problem:** the last value for time series, the most common class for classification, the mean for regression.
- **A bad baseline flatters your model.** "Training mean" had an error of 34.01 here, which makes almost anything look great.
- **Tree models can't predict outside the range they saw in training.** That's why the forest lost.
- **A model that only matches the baseline** has usually just rediscovered it.

## Can a simple rule beat a machine learning model?

Here is a price-like series where each day is yesterday plus a random change (a random walk).
We predict the next day from the last five days, training on the first 1,000 days and testing on the rest:

```python
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# A price-like series: each day = yesterday + random change (a random walk)
rng = np.random.default_rng(0)
price = 100 + np.cumsum(rng.normal(0, 1, 1_200))

# Features: the last 5 days. Target: the next day.
lags = 5
X = np.column_stack([price[i:len(price) - lags + i] for i in range(lags)])
y = price[lags:]
split = 1_000                                   # train on the past, test on the future
X_tr, X_te, y_tr, y_te = X[:split], X[split:], y[:split], y[split:]

results = {
    "baseline: tomorrow = today": mean_absolute_error(y_te, X_te[:, -1]),
    "baseline: training mean":    mean_absolute_error(y_te, np.full_like(y_te, y_tr.mean())),
    "linear regression":          mean_absolute_error(y_te, LinearRegression().fit(X_tr, y_tr).predict(X_te)),
    "random forest":              mean_absolute_error(y_te, RandomForestRegressor(random_state=0).fit(X_tr, y_tr).predict(X_te)),
}
print(f"train range: {y_tr.min():.1f} to {y_tr.max():.1f} | test range: {y_te.min():.1f} to {y_te.max():.1f}")
for name, mae in results.items():
    print(f"{name:<28} MAE {mae:6.2f}")
```

```
train range: 50.1 to 111.5 | test range: 39.8 to 65.4
baseline: tomorrow = today   MAE   0.79
baseline: training mean      MAE  34.01
linear regression            MAE   0.80
random forest                MAE   2.55
```

MAE (mean absolute error) is the average size of the miss, in price units. Lower is better.

| Model | MAE | Verdict |
|---|---|---|
| Baseline: tomorrow = today | **0.79** | Best |
| Linear regression | 0.80 | Ties the baseline |
| Random forest | 2.55 | More than 3× worse than the baseline |
| Baseline: training mean | 34.01 | Wrong baseline for time series |

## Why did the random forest lose?

A random forest predicts by averaging training examples, so it can never predict a value outside the range it saw.
The test period went down to 39.8, but the lowest training value was 50.1:

```python
# continues from the code above
lin = LinearRegression().fit(X_tr, y_tr)
print("linear weights on the last 5 days:", lin.coef_.round(2), "| intercept:", round(lin.intercept_, 2))
rf_pred = RandomForestRegressor(random_state=0).fit(X_tr, y_tr).predict(X_te)
print(f"lowest random forest prediction: {rf_pred.min():.1f} (lowest training target: {y_tr.min():.1f}, lowest test value: {y_te.min():.1f})")
```

```
linear weights on the last 5 days: [-0.01  0.    0.02 -0.01  1.01] | intercept: -0.11
lowest random forest prediction: 51.2 (lowest training target: 50.1, lowest test value: 39.8)
```

![Line chart of a 1,200-day random walk. In the test period the actual price drops to about 40; the naive baseline follows it closely, while the random forest stays flat near 52 because it cannot predict below the lowest training value of 50.1](/assets/img/tips/baseline_first/random_walk.png)

- The forest's lowest prediction was 51.2, so it stayed flat while the real price fell to 39.8.
- Linear regression put a weight of 1.01 on today's value and about 0 on the other days. It learned "tomorrow = today", the baseline itself.

For a true random walk, "tomorrow = today" is already the best possible forecast, so no model can beat it by much. The baseline tells you that immediately; the forest's error alone would not.

## Which baseline should you use?

| Problem | Baseline | In scikit-learn |
|---|---|---|
| Classification | Always predict the most common class | `DummyClassifier(strategy="most_frequent")` |
| Imbalanced classification | Predict classes in their training proportions | `DummyClassifier(strategy="stratified")` |
| Regression | Always predict the training mean or median | `DummyRegressor(strategy="mean")` |
| Time series | Last value, or the value one season ago | `X_te[:, -1]`, as above |

Report the baseline next to every model score. "MAE 2.55" is a number; "MAE 2.55 against a naive 0.79" is a decision.

## Sources

- [Dummy estimators](https://scikit-learn.org/stable/modules/model_evaluation.html#dummy-estimators) — scikit-learn user guide, checked 2026-09-28
- [`DummyRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyRegressor.html)

Related: [Why is 99% accuracy useless? Precision, recall and F1 explained](/tips/2026-09-30-accuracy_is_misleading/)
