---
layout: post
title: "Target encoding leakage: what scikit-learn's TargetEncoder cross fitting protects you from"
description: >
  Mean-encoding categories before cross-validation reported an MAE of 4,054 USD; the same model scored 4,636 on a
  holdout. Inside a pipeline, plain mean encoding was fine until an ID-like column appeared, then TargetEncoder's
  cross fitting kept the error flat. Tested with scikit-learn 1.9.1 on real salary data.
image: /assets/img/blog/target_encoding_leakage/cover.png
sitemap:
  changefreq: daily
  priority: 1.0
---

# Target encoding leakage: what scikit-learn's TargetEncoder cross fitting protects you from

**Target encoding** replaces each category with the average target value of its rows. A job title becomes the
average salary of people with that title. It turns a column with hundreds of categories into one useful number,
and it is also an easy way to leak the answer into your features.

I tested three ways of doing it on 9,228 real salaries with a gradient boosting model. Lower MAE is better:

| How the categories were encoded | What cross-validation reported | Error on a holdout nobody touched |
|---|---:|---:|
| Mean encoding computed on all rows, then cross-validation | **4,054** | 4,636 |
| Mean encoding inside a `Pipeline` | 4,875 | 4,636 |
| `TargetEncoder` inside a `Pipeline` | 5,202 | 4,971 |

The first row is the classic leak: cross-validation reported an error 13% lower than the model really made.
The second row was the surprise: plain mean encoding in a pipeline gave an honest score and beat `TargetEncoder`.
`TargetEncoder` earned its place only once an ID-like column was added, shown further down.

_Tested on 2026-10-27 with scikit-learn 1.9.1, pandas 3.0.6, NumPy 2.5.4 and Python 3.12.13 on Apple Silicon (macOS).
Full script: [`target_encoding.py`](/assets/code/target_encoding_leakage/target_encoding.py),
output: [`target_encoding_out.txt`](/assets/code/target_encoding_leakage/target_encoding_out.txt). Two runs printed
identical values._

## The data

[OpenML 42125 `employee_salaries`](https://www.openml.org/d/42125): annual salaries of Montgomery County, Maryland
employees in 2016, released as public domain (CC0). The target is `current_annual_salary` (mean 73,390 USD).
I used five category columns and the hire year:

| Column | Unique values |
|---|---:|
| `division` | 694 |
| `employee_position_title` | 385 |
| `underfilled_job_title` | 84 |
| `department` | 37 |
| `assignment_category` | 2 |

I left out the name, gender and pay columns. `2016_gross_pay_received` is almost the salary itself, which would be
a different kind of leak.

Plain mean encoding is a few lines of pandas: group the training rows by category, take the mean target, and use
the overall mean for categories never seen in training. The script wraps that in a small `MeanEncoder` class so it
can sit in a pipeline. The model is `HistGradientBoostingRegressor` with default settings.

## Leak 1: encoding before the split

This is the version found in many notebooks:

```python
means = df.groupby("division")["salary"].mean()     # uses every row, test rows included
df["division_enc"] = df["division"].map(means)
cross_val_score(model, df[features], df["salary"], cv=5)
```

Every test fold's salaries are already inside the encoded column. A category with two employees, one of them in the
test fold, hands the model half of the answer. Cross-validation reported an MAE of **4,054 USD**. When the same model
was trained on all 7,382 development rows and scored on a separate holdout of 1,846 rows, its error was
**4,636 USD**.

The final model was fine: the pipeline version of it gets exactly the same holdout error, 4,636.
**The problem is the number you report.** If you choose models or features by that cross-validation score, you
choose them with leaked information. With three other random seeds, the reported score was 421 to 719 USD better
than the holdout every time.

The fix is the same as for scalers and feature selectors (see
[why preprocessing must be fitted on training data only](/blog/2026-10-02-data_leakage_preprocessing/)): put the
encoder in a `Pipeline`, so every fold computes its own means from its own training rows.

## Leak 2: the training rows encode themselves

A pipeline protects the test fold. It does not protect the training rows from themselves. When the encoder computes
"average salary per category" from the training rows and then encodes those same rows, every row's own salary is
part of its code. For a category with a single training row, the code **is** that row's salary.

To see what that does, I added a column with no information at all: a random ID, each value shared by about 1.5
rows. It cannot predict salary. Here is how its encoded value relates to salary on the training rows:

![Two scatter plots of the same training rows. Left, plain mean encoding of a random ID column: the encoded value correlates with salary at r = 0.77, with a sharp diagonal line where an ID has a single row and its code equals that row's salary. Right, TargetEncoder fit_transform of the same column: no relationship, r = 0.01, with a vertical band at the overall mean salary](/assets/img/blog/target_encoding_leakage/cover.png)

| How the training rows were encoded | Correlation with salary, training rows | Test rows |
|---|---:|---:|
| Plain mean encoding | **0.772** | -0.006 |
| `TargetEncoder.fit_transform(X, y)` | 0.005 | -0.006 |
| `TargetEncoder.fit(X, y).transform(X)` | **0.761** | -0.006 |

With plain encoding, a useless column looks like one of the strongest features in the training data, and the
model learns to trust it. On test rows the correlation is zero, because their own salaries are not in their codes.

`TargetEncoder.fit_transform` avoids this with **cross fitting**: it splits the training rows into 5 folds and
encodes each fold with means computed from the other 4, so no row sees its own target. It also shrinks rare
categories toward the overall mean (`smooth="auto"`), which is the vertical band in the right panel.
At prediction time, `transform` uses means from all training rows.

## How much does it matter?

It depends on how rare the categories are. Test MAE from 5-fold cross-validation, every encoder inside a pipeline:

![Grouped bar charts of test MAE. HistGradientBoosting: with no random ID, plain mean encoding 4,751 and TargetEncoder 5,065; with a random ID of about 3 rows each, 5,093 and 5,111; with about 1.5 rows each, 5,510 and 5,097. Ridge regression: 7,108 and 7,094; 7,417 and 7,095; 7,943 and 7,096](/assets/img/blog/target_encoding_leakage/rare_categories.png)

| Extra column | Model | Plain mean encoding | `TargetEncoder` |
|---|---|---:|---:|
| none | HistGradientBoosting | **4,751** | 5,065 |
| none | Ridge | 7,108 | 7,094 |
| random ID, ~3 rows each (2,918 IDs) | HistGradientBoosting | 5,093 | 5,111 |
| random ID, ~3 rows each | Ridge | 7,417 | **7,095** |
| random ID, ~1.5 rows each (4,802 IDs) | HistGradientBoosting | 5,510 | **5,097** |
| random ID, ~1.5 rows each | Ridge | 7,943 | **7,096** |

- **`TargetEncoder` stayed flat.** The useless column moved its error by less than 50 USD with either model.
- **Plain encoding got worse as the IDs got rarer:** 4,751 to 5,510 with gradient boosting, 7,108 to 7,943 with
  ridge. Its training error kept falling at the same time (3,397 to 3,240), a sign that the model was fitting the
  leaked codes. With three more random seeds, `TargetEncoder` won the ~1.5 rows-per-ID case by 501 to 595 USD.
- **Without the ID column, plain encoding was better for gradient boosting** (4,751 vs 5,065). The real columns
  carry a lot of signal here, and the cross-fitted codes are noisier. I did not test the cause; smoothing, which
  pulls rare job titles toward the average, is a likely part of it.

So cross fitting is insurance, not a free gain. You pay for it because real data has ID-like columns, long tails of
rare categories, and new columns you have not checked one by one.

## Trap: `fit(X, y).transform(X)` skips the cross fitting

The scikit-learn docs say it plainly: `fit(X, y).transform(X)` does not equal `fit_transform(X, y)`. The third row
of the correlation table shows what that means. If you call the encoder by hand like this:

```python
enc = TargetEncoder().fit(X_train, y_train)
X_train_enc = enc.transform(X_train)     # no cross fitting: rows see their own target
X_test_enc = enc.transform(X_test)
```

you are back to plain mean encoding on the training rows, with some smoothing. On one fold with the random ID
column, gradient boosting trained this way scored 5,661 USD against 5,094 for the pipeline. Call `fit_transform`
on training data, or put the encoder in a `Pipeline` or `ColumnTransformer`, which call `fit_transform` for you.

## What to use

- **Never compute target statistics before the split.** Not for encoding, not for "average target per group"
  features. The score you get is not the score you will have.
- **Use `TargetEncoder` in a pipeline for high-cardinality columns** such as zip codes, product IDs or job titles.
  It is the safe default when you cannot check every column by hand.
- **For low-cardinality columns, one-hot encoding is simpler.** Target encoding helps when one-hot would create
  hundreds of columns.
- **If your folds must respect groups or time, give the encoder the same splitter.** Since scikit-learn 1.9, `cv`
  accepts a splitter such as `GroupKFold(5)`. Passing `groups` to `fit_transform` also requires
  `sklearn.set_config(enable_metadata_routing=True)`; without it, 1.9.1 raises a `ValueError`. A shuffled `KFold`
  inside the encoder can leak the same way it does in the outer loop
  (see [KFold vs TimeSeriesSplit](/blog/2026-10-23-timeseries_cv_leakage/)).
- **`shuffle` and `random_state` on `TargetEncoder` are deprecated in 1.9** and will be removed in 1.11. Pass
  `cv=KFold(5, shuffle=True, random_state=0)` instead, as the script does.

## Limits of this test

One dataset, one regression target, two models with default settings. The random ID column was added to make rare
categories controllable; it is not part of the original data. The seed check
([`seeds_check.py`](/assets/code/target_encoding_leakage/seeds_check.py),
[output](/assets/code/target_encoding_leakage/seeds_check_out.txt)) repeats the leak comparison and the
~1.5 rows-per-ID case with three more seeds. Charts come from
[`te_charts.py`](/assets/code/target_encoding_leakage/te_charts.py).

Sources: [`TargetEncoder` API](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.TargetEncoder.html)
and [user guide](https://scikit-learn.org/stable/modules/preprocessing.html#target-encoder), checked 2026-10-27
against the docstring of the installed 1.9.1.
