---
layout: post
title: "Train, validation and test sets: what overfitting looks like in numbers"
description: >
  Why you need three data splits, measured on a real dataset: a fully grown decision tree scored 1.000 on training data
  and 0.912 on validation, and picking the best model on one small validation split overstated the test score.
image: /assets/img/tips/overfitting_train_val_test/cover.png
sitemap:
  changefreq: daily
  priority: 1.0
---

# Train, validation and test sets: what overfitting looks like in numbers

**TL;DR** — Train on the **training set**, choose settings with the **validation set** (or cross-validation), and use the
**test set** once, at the very end. **Overfitting** is when a model learns its training data too well to generalize:
in our test, a decision tree reached 1.000 training accuracy while validation accuracy fell to 0.912.

_Tested on 2026-09-28 with scikit-learn 1.9.1 on the built-in breast cancer dataset (569 rows)._

## Key points

- **Training set:** the data the model learns from. A training score says little about new data.
- **Validation set:** used to compare models and choose settings (hyperparameters), such as tree depth.
- **Test set:** touched once, after every choice is made. It estimates performance on truly new data.
- **Overfitting shows as a gap:** training accuracy rises while validation accuracy stalls or drops.
- **Choosing the best of many on one small validation set overstates the result.** Cross-validation is more reliable.

## What does overfitting look like?

We split the data 60/20/20 and grew decision trees of increasing depth:

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

X, y = load_breast_cancer(return_X_y=True)
# 60% train, 20% validation, 20% test
X_tmp, X_test, y_tmp, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)
X_train, X_val, y_train, y_val = train_test_split(X_tmp, y_tmp, test_size=0.25, stratify=y_tmp, random_state=0)
print(f"train {len(y_train)}, validation {len(y_val)}, test {len(y_test)}")

print(f"{'max_depth':>9}{'train acc':>11}{'val acc':>9}{'gap':>7}")
scores = {}
for depth in [1, 2, 3, 4, 5, 7, 10, None]:
    tree = DecisionTreeClassifier(max_depth=depth, random_state=0).fit(X_train, y_train)
    tr, va = tree.score(X_train, y_train), tree.score(X_val, y_val)
    scores[depth] = va
    print(f"{str(depth):>9}{tr:>11.3f}{va:>9.3f}{tr - va:>7.3f}")

best = max(scores, key=scores.get)
final = DecisionTreeClassifier(max_depth=best, random_state=0).fit(X_train, y_train)
print(f"\nbest depth on validation: {best} (val acc {scores[best]:.3f})")
print(f"test accuracy, measured once: {final.score(X_test, y_test):.3f}")
```

```
train 341, validation 114, test 114
max_depth  train acc  val acc    gap
        1      0.927    0.904  0.023
        2      0.962    0.939  0.023
        3      0.982    0.930  0.053
        4      0.994    0.947  0.047
        5      0.997    0.965  0.032
        7      1.000    0.912  0.088
       10      1.000    0.912  0.088
     None      1.000    0.912  0.088

best depth on validation: 5 (val acc 0.965)
test accuracy, measured once: 0.912
```

![Line chart of accuracy versus decision tree depth 1 to 12: training accuracy climbs to 1.000 from depth 6 on; single-split validation accuracy peaks at 0.965 at depth 5 and drops to 0.912; 5-fold cross-validation accuracy peaks at 0.949 at depth 4 and levels off around 0.94](/assets/img/tips/overfitting_train_val_test/validation_curve.png)

- **Shallow trees (depth 1–2) underfit:** both scores are lower.
- **From depth 6, training accuracy is 1.000.** The tree has memorized every training row, and validation accuracy drops to 0.912.
- **The test score (0.912) came in well below the validation score (0.965) of the chosen model.**

## Why was the test score lower than the validation score?

Because we picked the depth with the highest validation score out of eight. With only 114 validation rows, one row is 0.9 percentage points,
so some of that top score was luck. Choosing the maximum of several noisy scores favors the lucky one.

Five-fold cross-validation uses the training and validation rows five times over, which averages out the luck:

```python
# continues from the code above (X_tmp, y_tmp, X_test, y_test)
from sklearn.model_selection import cross_val_score

print(f"{'max_depth':>9}{'5-fold CV acc':>15}{'std':>7}")
cv = {}
for depth in [1, 2, 3, 4, 5, 7, 10, None]:
    s = cross_val_score(DecisionTreeClassifier(max_depth=depth, random_state=0), X_tmp, y_tmp, cv=5)
    cv[depth] = s.mean()
    print(f"{str(depth):>9}{s.mean():>15.3f}{s.std():>7.3f}")
best = max(cv, key=cv.get)
final = DecisionTreeClassifier(max_depth=best, random_state=0).fit(X_tmp, y_tmp)
print(f"\nbest depth by CV: {best} (CV acc {cv[best]:.3f}) -> test accuracy: {final.score(X_test, y_test):.3f}")
```

```
max_depth  5-fold CV acc    std
        1          0.899  0.015
        2          0.936  0.011
        3          0.941  0.020
        4          0.949  0.011
        5          0.938  0.019
        7          0.938  0.019
       10          0.938  0.019
     None          0.938  0.019

best depth by CV: 4 (CV acc 0.949) -> test accuracy: 0.939
```

| How the depth was chosen | Chosen depth | Score used to choose | Test accuracy | Gap |
|---|---|---|---|---|
| One validation split (114 rows) | 5 | 0.965 | 0.912 | 0.053 |
| 5-fold cross-validation | 4 | 0.949 | 0.939 | 0.010 |

Cross-validation picked a different depth, and its estimate was much closer to the test result.

## Rules of thumb

- **Split before you look.** Set the test set aside first, and don't use it to make any decision.
- **Prefer cross-validation for choosing settings,** especially on small data.
- **Watch the gap.** A large training–validation gap means overfitting; two low scores mean underfitting.
- **Fit preprocessing inside the training folds too,** or information from the validation data leaks in.
- **If you peek at the test set and change the model, it is no longer a test set.** Report it as validation.

## Sources

- [Cross-validation: evaluating estimator performance](https://scikit-learn.org/stable/modules/cross_validation.html) — scikit-learn user guide, checked 2026-09-28

Related: [Why fit preprocessing on training data only? A data leakage demo](/blog/2026-10-02-data_leakage_preprocessing/) · [Why you should always build a baseline model first](/tips/2026-10-02-baseline_first/)
