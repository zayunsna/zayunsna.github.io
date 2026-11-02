---
layout: post
title: "pandas 3.1 select(): when to use it instead of assign()"
description: >
  DataFrame.select computes columns while choosing the output schema. Compare it with assign on the same order
  table, and reproduce the stale-Series mistake that gives different totals inside one call.
image: /assets/img/blog/pandas_select_assign/cover.png
sitemap:
  changefreq: daily
  priority: 1.0
---

# pandas 3.1 select(): when to use it instead of assign()

Use `select()` when you want to calculate columns **and say exactly which columns leave the step**. Use
`assign()` when you want to add or replace columns while keeping the rest of the table.

On the same three-row order table, both methods calculated identical gross amounts and tax. `select()` returned
three named output columns. `assign()` returned six, including an internal note that the output did not need.
The distinction is the schema, not a claim that one method is faster.

[pandas 3.1.0, released October 22, 2026](https://pandas.pydata.org/pandas-docs/stable/whatsnew/v3.1.0.html),
adds `DataFrame.select`. The examples here were executed on **2026-10-30**, Python 3.12.13 and pandas 3.1.0 on
macOS. API and release details as of 2026-10-30. They use constructed orders, not production data. Full executable
[script](/assets/code/pandas_select_assign/select_assign.py),
[output](/assets/code/pandas_select_assign/reference_output.txt) and
[run instructions](/assets/code/pandas_select_assign/README.txt) are included.

## Choose the result, not just the calculations

Install `pandas==3.1.0` in a Python 3.12 virtual environment, then run these Python blocks in order.

```python
import pandas as pd

orders = pd.DataFrame(
    {'order': ['A', 'B', 'C'], 'qty': [2, 3, 1],
     'price': [10., 20., 30.], 'internal_note': ['x', 'y', 'z']},
    index=[10, 20, 40],
)
selected = orders.select(
    'order',
    gross=pd.col('qty') * pd.col('price'),
    tax=pd.col('gross') * 0.1,
)
print(selected.to_string())
```

```text
   order  gross  tax
10     A   20.0  2.0
20     B   60.0  6.0
40     C   30.0  3.0
```

The positional string keeps `order`. The keyword arguments name computed columns. `tax` can use `gross`,
which was computed earlier in the call. Input columns such as `qty` and `price` are available to expressions
without being included in the final output. The non-default row index survives.

Compare that with [`assign()`](https://pandas.pydata.org/pandas-docs/version/3.1/reference/api/pandas.DataFrame.assign.html):

```python
assigned = orders.assign(
    gross=pd.col('qty') * pd.col('price'),
    tax=pd.col('gross') * 0.1,
)
print(selected.columns.tolist())
print(assigned.columns.tolist())
pd.testing.assert_frame_equal(selected, assigned[['order', 'gross', 'tax']])
print(orders.columns.tolist())
```

```text
['order', 'gross', 'tax']
['order', 'qty', 'price', 'internal_note', 'gross', 'tax']
['order', 'qty', 'price', 'internal_note']
```

The assertion verifies the same values, index, column order and dtypes after projecting the assigned result.
Neither call adds columns to `orders` itself. `select()` saves the separate projection when that projection is
the point of the step; it does not make `assign()` obsolete.

At an export boundary, the explicit list can also keep unrelated columns from drifting into the result when
the input schema grows. It is still your responsibility to inspect the chosen columns; a method name is not a
data-access control.

## A precomputed Series is not a deferred expression

The more consequential mistake is mixing an expression that sees an updated column with a Series calculated
from the old table:

```python
deferred = orders.assign(
    qty=pd.col('qty') + 1,
    gross=pd.col('qty') * pd.col('price'),
)
eager = orders.assign(
    qty=orders['qty'] + 1,
    gross=orders['qty'] * orders['price'],
)
print(deferred['qty'].tolist())
print(deferred['gross'].tolist())
print(eager['gross'].tolist())
```

```text
[3, 4, 2]
[30.0, 80.0, 60.0]
[20.0, 60.0, 30.0]
```

In the second call, Python calculates both Series from the original `orders` before `assign()` receives them.
The new quantity does not retroactively change that already-computed gross amount.

`pd.col()` defers the calculation until pandas evaluates it against the table at that step. A callable such as
`gross=lambda d: d['qty'] * d['price']` can express the same dependency. The
[`select` reference](https://pandas.pydata.org/pandas-docs/version/3.1/reference/api/pandas.DataFrame.select.html)
and release notes document sequential dependencies; the different numbers above come from the executed script.
This is not a rule that all direct Series arguments are wrong—only that they refer to the object used to
construct them, not automatically to intermediate updates.

## Row filtering is a separate decision

`select()` chooses columns, not rows. pandas 3.1 also adds expression-based row filtering through `filter`:

```python
filtered = (
    orders.filter(pd.col('qty') > 1)
    .select('order', gross=pd.col('qty') * pd.col('price'))
)
print(filtered.index.tolist())
```

```text
[10, 20]
```

That is different from older label-oriented uses such as `filter(items=[...])`. The new behavior is documented
in the [3.1 release notes](https://pandas.pydata.org/pandas-docs/stable/whatsnew/v3.1.0.html#filtering-with-a-boolean-mask-in-dataframe-filter-and-series-filter).
For compatibility with code that cannot require 3.1, the downloadable script verifies the equivalent
`loc` → `assign` → column-list result as well. It was tested here on 3.1, not rerun on every older pandas version.

I would keep `assign()` for enrichment steps and reach for `select()` at a deliberate output boundary. Before
rewriting an existing chain, test both the output schema and an intermediate-column dependency. A readable new
API is useful; silently calculating yesterday's expression is not.

For the broader 3.0 migration issues, see [the pandas 3 upgrade notes](/ds/2026-09-28-pandas_3_upgrade/).
The cover visualizes the column lists from this experiment; its
[renderer](/assets/code/tue_wed_figures/render_figures.py) reads the saved result JSON.
