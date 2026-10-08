---
layout: post
title: "LightGBM LambdaRank in Python: group, NDCG, and the mistakes that don't raise an error"
description: >
  A runnable LGBMRanker example on LightGBM's own ranking data: how the group parameter works, the two mistakes LightGBM
  catches, the one it doesn't, and why scikit-learn's ndcg_score gave 0.715 where LightGBM reported 0.682. Tested with LightGBM 4.7.
image: /assets/img/blog/lightgbm_lambdarank/cover.png
sitemap:
  changefreq: daily
  priority: 1.0
---

# LightGBM LambdaRank in Python: group, NDCG, and the mistakes that don't raise an error

To train a ranking model with LightGBM, you pass `LGBMRanker` the features, a relevance label per row, and a
`group` array that says how many consecutive rows belong to each query. Most mistakes come from that last part.
In the run below, LightGBM stopped two mistakes with a clear error, trained on shuffled rows **without any warning**,
and reported an NDCG@5 of 0.682 while scikit-learn's `ndcg_score` on the same predictions said 0.715.

_Tested on 2026-10-08 with Python 3.12.13, LightGBM 4.7.0, scikit-learn 1.9.1 and NumPy 2.5.3 on macOS (Apple Silicon)._

## Setup and data

```bash
pip install lightgbm scikit-learn numpy
```

On macOS, LightGBM's wheel needs the OpenMP runtime. Without it, `import lightgbm` failed for me with
`Library not loaded: @rpath/libomp.dylib`. `brew install libomp` fixed it.

The data is the small learning-to-rank example that ships with LightGBM itself
([examples/lambdarank](https://github.com/lightgbm-org/LightGBM/tree/main/examples/lambdarank), MIT license):
documents with 300 features and a relevance label from 0 to 4, plus a `.query` file with the number of documents in
each query.

```bash
for f in rank.train rank.train.query rank.test rank.test.query; do
  curl -sLO https://raw.githubusercontent.com/lightgbm-org/LightGBM/main/examples/lambdarank/$f
done
```

The code below is one script, split into blocks. Run the blocks in order.

## How do you train LGBMRanker?

```python
# --- 1. Load LightGBM's own learning-to-rank example data ---
import numpy as np
import lightgbm as lgb
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import ndcg_score

X_train, y_train = load_svmlight_file("rank.train")
X_test, y_test = load_svmlight_file("rank.test", n_features=X_train.shape[1])
group_train = np.loadtxt("rank.train.query", dtype=int)   # number of rows per query, in row order
group_test = np.loadtxt("rank.test.query", dtype=int)
print(X_train.shape, len(group_train), "queries |", X_test.shape, len(group_test), "queries")

# --- 2. Train a ranker and evaluate NDCG@5 on the test queries ---
params = dict(objective="lambdarank", n_estimators=200, learning_rate=0.05,
              num_leaves=31, min_child_samples=10, random_state=0, verbose=-1)
ranker = lgb.LGBMRanker(**params)
ranker.fit(X_train, y_train, group=group_train,
           eval_X=(X_test,), eval_y=(y_test,), eval_group=[group_test], eval_at=[5])
print("LightGBM ndcg@5:", round(ranker.evals_result_["valid_0"]["ndcg@5"][-1], 4))
```

```
(3005, 300) 201 queries | (768, 300) 50 queries
LightGBM ndcg@5: 0.6816
```

`group_train` is not a query id per row. It is a list of sizes: the first query is the first `group_train[0]` rows,
the next query is the next `group_train[1]` rows, and so on. Its sum must equal the number of rows.

Two details changed in LightGBM 4.7.0. `eval_set` is
[deprecated](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRanker.html) in favour of `eval_X`
and `eval_y`, and the types matter: the docs describe `eval_X` and `eval_y` as a value "or tuple thereof", and
`eval_group` as a list of arrays. When I passed `eval_X=[X_test]` (a list), LightGBM treated it as raw data and
failed with `TypeError: Data list can only be of ndarray or Sequence`. With `eval_group=(group_test,)` it failed with
`TypeError: eval_group should be dict or list`. Tuples for `eval_X`/`eval_y`, a list for `eval_group` worked.

## How do you build group from a query-id column?

Real data usually has a query-id column instead of a `.query` file. Sort by it, then count rows per query:

```python
# --- 3. Build group from a query-id column, and check the rows are contiguous ---
qid = np.repeat(np.arange(len(group_train)), group_train)        # what a qid column looks like
order = np.argsort(qid, kind="stable")                          # sort rows by query
_, group_from_qid = np.unique(qid[order], return_counts=True)   # sizes in sorted order
print("same as the .query file:", np.array_equal(group_from_qid, group_train))


def rows_are_grouped(qid):
    """True if each query's rows sit next to each other."""
    starts = np.flatnonzero(np.r_[True, qid[1:] != qid[:-1]])
    return len(starts) == len(np.unique(qid))


perm = np.random.default_rng(0).permutation(len(qid))
print("sorted rows grouped:  ", rows_are_grouped(qid))
print("shuffled rows grouped:", rows_are_grouped(qid[perm]))
lgb.LGBMRanker(**params).fit(X_train[perm], y_train[perm], group=group_train)
print("training on shuffled rows: finished, no error and no warning")
```

```
same as the .query file: True
sorted rows grouped:   True
shuffled rows grouped: False
training on shuffled rows: finished, no error and no warning
```

This is the dangerous one. If your rows get shuffled (a random train/test split, a `sample(frac=1)`, a join that
reorders), the group sizes still add up, so LightGBM happily trains on "queries" that mix documents from different
searches. Nothing tells you.

You might hope the metric would show it. On this data it didn't: in a separate run with five different shuffles
(permutation seeds 0–4), test NDCG@5 ranged from 0.665 to 0.701, around the correctly grouped 0.682. With only
50 test queries, the noise is bigger than the damage. So don't rely on the score to catch it; put a check like
`assert rows_are_grouped(qid)` right before `fit`.

## Which mistakes does LightGBM catch?

```python
# --- 4. Two mistakes LightGBM does catch ---
for name, kwargs in [("group sizes don't add up", dict(group=group_train[:-1], y=y_train)),
                     ("labels go up to 40", dict(group=group_train, y=y_train * 10))]:
    try:
        lgb.LGBMRanker(**params).fit(X_train, kwargs["y"], group=kwargs["group"])
    except lgb.basic.LightGBMError as e:
        print(f"{name}: {e}")

fixed = lgb.LGBMRanker(**params, label_gain=list(range(41)))   # one gain per label value 0..40
fixed.fit(X_train, y_train * 10, group=group_train)
print("labels up to 40 with label_gain=list(range(41)): trained")
```

```
group sizes don't add up: Sum of query counts (3005) differs from the length of #data (2995)
labels go up to 40: Label 40 is not less than the number of label mappings (31)
labels up to 40 with label_gain=list(range(41)): trained
```

The second error surprises people whose labels are scores like 0–100. By default `label_gain` has 31 entries
(`0, 1, 3, 7, …, 2^30 − 1`), and every label must be smaller than the number of entries, so labels can only go up
to 30 ([LightGBM parameters](https://lightgbm.readthedocs.io/en/latest/Parameters.html#label_gain)). Either map your
scores to a few relevance levels, or pass a `label_gain` with one entry per label value. Note that `list(range(41))`
also changes the gains from exponential to linear, which changes what the model optimizes.

## Why doesn't scikit-learn's ndcg_score match LightGBM?

```python
# --- 5. LightGBM's NDCG vs scikit-learn's ndcg_score ---
def mean_ndcg(scores, labels, groups, k=5, exponential=True):
    out, start = [], 0
    for size in groups:
        y, s = labels[start:start + size], scores[start:start + size]
        start += size
        out.append(ndcg_score([2 ** y - 1 if exponential else y], [s], k=k))
    return np.mean(out)


pred = ranker.predict(X_test)
print("sklearn ndcg_score, gains 2^y - 1:", round(mean_ndcg(pred, y_test, group_test), 4))
print("sklearn ndcg_score, raw labels:   ", round(mean_ndcg(pred, y_test, group_test, exponential=False), 4))
```

```
sklearn ndcg_score, gains 2^y - 1: 0.6816
sklearn ndcg_score, raw labels:    0.715
```

Same predictions, two different NDCG@5 values. LightGBM's default gain for a document with label `y` is `2^y − 1`,
so a label-4 document is worth 15 and a label-1 document 1. `sklearn.metrics.ndcg_score` uses the relevance values
you give it as the gains directly, so label 4 is worth 4. Pass `2 ** y - 1` to `ndcg_score` and the two agree
exactly (0.6816).

| Metric | Gain for label 4 vs label 1 | NDCG@5 here |
|---|---|---|
| LightGBM `ndcg@5` (default `label_gain`) | 15 vs 1 | 0.6816 |
| `ndcg_score` with `2 ** y - 1` | 15 vs 1 | 0.6816 |
| `ndcg_score` with raw labels | 4 vs 1 | 0.715 |

If you compare models or report numbers, say which gain you used. Mixing them makes a model look better or worse
than it is.

## Does a ranker beat a plain regressor here?

A fair question before using `LGBMRanker` at all: what if you just regress the label and sort by the prediction?

```python
# --- 6. Ranker vs a plain regressor, with a bootstrap over test queries ---
def per_query(scores):
    out, start = [], 0
    for size in group_test:
        out.append(ndcg_score([2 ** y_test[start:start + size] - 1], [scores[start:start + size]], k=5))
        start += size
    return np.array(out)


reg_params = {k: v for k, v in params.items() if k != "objective"}
regressor = lgb.LGBMRegressor(**reg_params).fit(X_train, y_train)
diff = per_query(pred) - per_query(regressor.predict(X_test))
rng = np.random.default_rng(1)
boot = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(5000)]
print(f"ranker NDCG@5 {per_query(pred).mean():.4f} | regressor {per_query(regressor.predict(X_test)).mean():.4f}")
print(f"difference {diff.mean():+.4f}, 95% bootstrap interval [{np.percentile(boot, 2.5):+.4f}, {np.percentile(boot, 97.5):+.4f}]")
print("lightgbm", lgb.__version__)
```

```
ranker NDCG@5 0.6816 | regressor 0.6627
difference +0.0189, 95% bootstrap interval [-0.0340, +0.0695]
lightgbm 4.7.0
```

The ranker came out 0.019 ahead, but resampling the 50 test queries gives an interval that includes zero. On this
small example, I can't claim the ranker is better. That isn't an argument against LambdaRank; it's a reminder that
50 queries is too few to tell two decent models apart. With your own data, compare on many more queries, and keep
whole queries together when you split train and test (for example
[GroupShuffleSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupShuffleSplit.html)
on the query id), for the same reason as in [data leakage in preprocessing](/blog/2026-10-02-data_leakage_preprocessing/).

## Checklist

| Check | Why |
|---|---|
| Rows sorted by query, `assert rows_are_grouped(qid)` | Shuffled rows train silently |
| `group.sum() == len(X)` | Otherwise LightGBM stops with "Sum of query counts … differs" |
| Labels are integers 0–30, or you set `label_gain` | Otherwise "Label … is not less than the number of label mappings (31)" |
| `eval_X`/`eval_y` as tuples, `eval_group` as a list (LightGBM ≥ 4.7) | `eval_set` is deprecated; wrong container types raise `TypeError` |
| Same NDCG gain in every report | Raw-label and `2^y − 1` gains gave 0.715 vs 0.682 here |
| Split train/test by query | Rows of one query on both sides leak information |

The idea behind LambdaRank in search ranking, and why Airbnb moved to it, is in my notes on
[Applying Deep Learning to Airbnb Search](/ds/2025-05-02-airbnb_model/).
