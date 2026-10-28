---
layout: post
title: "Polars 2.0 upgrade: the lazy join that now returns rows in a different order"
description: >
  Polars 2.0 runs lazy queries on the streaming engine by default, and joins no longer keep the left row order.
  A "first purchase per customer" query that was right on 1.44 gave up to 99% wrong rows per run on 2.1.
  The fixes, plus three other changes that alter results without an error.
image: /assets/img/blog/polars_2_upgrade/cover.png
sitemap:
  changefreq: daily
  priority: 1.0
---

# Polars 2.0 upgrade: the lazy join that now returns rows in a different order

Polars 2.0 came out on [2026-10-06](https://github.com/pola-rs/polars/releases/tag/py-2.0.0), and 2.1 followed on
[2026-10-25](https://github.com/pola-rs/polars/releases/tag/py-2.1.0). The biggest change is in the
[upgrade guide](https://docs.pola.rs/releases/upgrade/2/): `LazyFrame.collect()` now uses the **streaming engine**
by default. The streaming engine does not promise row order for joins, group-bys or unpivots.

Most breaking changes in 2.0 raise an error, so you find them on the first run. The row order change doesn't.
I ran the same code on Polars 1.44.2 and 2.1.0 to see which changes are silent.

_Tested on 2026-10-29 with Polars 1.44.2 and 2.1.0, Python 3.12.13, Apple M5 (10 cores), macOS. All data is
generated with NumPy seed 0._

## A query that was right on 1.44 and wrong on 2.1

The data is 2 million transactions sorted by time, for 50,000 customers. The query adds the customer table and
takes each customer's first transaction:

```python
import numpy as np
import polars as pl

rng = np.random.default_rng(0)
n = 2_000_000
tx = pl.DataFrame({
    "ts": np.arange(n),  # rows are already in time order
    "customer": rng.integers(0, 50_000, n),
})
customers = pl.DataFrame({"customer": np.arange(50_000), "region": rng.integers(0, 5, 50_000)})

first = (tx.lazy()
         .join(customers.lazy(), on="customer", how="left")
         .group_by("customer")
         .agg(pl.col("ts").first().alias("first_ts"))
         .collect())

truth = tx.group_by("customer").agg(pl.col("ts").min().alias("true_ts"))
check = first.join(truth, on="customer")
print(f"{(check['first_ts'] != check['true_ts']).mean():.1%} of customers got the wrong first purchase")
```

On Polars 1.44.2 this printed 0.0% every time. On 2.1.0 the join returned the rows in a different order, so
`first()` picked a later transaction. Two runs of this snippet on 2.1.0 printed 100.0% and 98.5%.

The number changes from run to run, so I ran each setup 10 times with
[first_purchase.py](/assets/code/polars_2_upgrade/first_purchase.py):

| Setup | Wrong "first purchase" (10 runs) | Median time |
|---|---|---|
| Polars 1.44.2, 10 threads | 0% every run | 10 ms |
| Polars 2.1.0, 10 threads | 0.2% to 99.1%, median 48.4% | 9 ms |
| Polars 2.1.0, 2 threads | 0% to 75.9%, median 37.3% | 14 ms |
| Polars 2.1.0, 1 thread | 0% every run | 19 ms |
| 2.1.0, `join(..., maintain_order="left")` | 0% every run | 16 ms |
| 2.1.0, `pl.col("ts").min()` instead of `first()` | 0% every run | 8 ms |
| 2.1.0, `collect(engine="in-memory")` | 0% every run | 10 ms |

The bug depends on the number of threads. With one thread, the 2.1 result was correct in all 10 runs. A test
that runs on a small CI machine can pass, and the same code can give wrong numbers on a larger server.

The 1.x code was already relying on undefined behavior. The 1.44.2 docstring for `join` says the default
`maintain_order="none"` means "the ordering might differ across Polars versions or even between different runs".
Polars 1.44 just happened to keep the left order in this case.

The `group_by` was not the problem. Without the join, `group_by().agg(pl.col("ts").first())` was correct in
every run on 2.1.0 (1, 2 and 10 threads), because Polars keeps the row order inside each group. The join changed the order before
the group_by saw it.

## How to fix it

Pick the fix by what the code means:

```python
# Best: say what you mean. "First by time" is the minimum time.
.agg(pl.col("ts").min())

# Or keep the order that later steps depend on.
.join(customers.lazy(), on="customer", how="left", maintain_order="left")

# Or sort right before the step that needs order.
.sort("ts").group_by("customer", maintain_order=True).agg(pl.col("ts").first())
```

`min()` was also the fastest option here (8 ms). `maintain_order="left"` made this query slower
(16 ms against 9 ms).

To get the old engine back while you audit a codebase, the upgrade guide lists three switches. I checked the
environment variable: with `POLARS_ENGINE_AFFINITY=in-memory`, all five queries in the script gave 0% wrong rows.

```python
pl.Config.set_engine_affinity("in-memory")  # whole process
# or: export POLARS_ENGINE_AFFINITY=in-memory
# or: lf.collect(engine="in-memory")      # one query
```

Treat this as a temporary step. The 1.x docs already said not to rely on join order, and the switch only hides
code that depends on it.

Where to look for this bug: `first()`, `last()`, `head()`, `tail()`, `cum_sum()`, `shift()`, `diff()` and
`rank(method="ordinal")` after a lazy join, and any code that writes a lazy result to a file and expects the
old row order.

## Three more changes that don't raise an error

I ran each one on both versions ([probe.py](/assets/code/polars_2_upgrade/probe.py)):

| Code | Polars 1.44.2 | Polars 2.1.0 |
|---|---|---|
| `Int64` column + `UInt64` column, value 9,007,199,254,740,993 + 1 | `Float64`, 9007199254740992.0 | `Int128`, 9007199254740994 |
| `df.select(cs.numeric() & pl.col("x"))` with numeric columns `x=[1,2]`, `y=[3,4]` | column `x` only | columns `x` and `y`, values `x & x` and `y & x` (bitwise AND): y becomes [1, 0] |
| `explode("tags")` where one row has `[]` | 3 rows, the empty list becomes `null` | 2 rows, the row with `[]` is gone |

The integer change fixes a real precision loss: 1.44 rounded the sum to the nearest float. It still changes the
dtype, and code that expects `Float64` may break later.

The `explode()` change can make an ID disappear from your data. If you need the old behavior, the guide says to
pass `empty_as_null=True`.

## Changes that do raise an error

These are easy to find, because the first run fails:

| Code | Polars 2.1.0 |
|---|---|
| `pl.concat([...], how="horizontal")` with different heights | `ShapeError`. Use `how="horizontal_extend"` to pad with nulls |
| int column `.is_in([1.0, 2.5])` | `InvalidOperationError`. Lossy casts are no longer automatic |
| `df.melt(id_vars="id")` | `AttributeRemovedError`. Use `unpivot(index="id")` |

The upgrade guide lists many more, including `read_csv` options, removed APIs and casts that now raise.

## What I would do before upgrading

1. Search the code for `first()`, `last()`, `head()` and cumulative functions that run after a lazy `join`,
   `group_by` or `unpivot`. Replace them with explicit sorts or `min()`/`max()`.
2. Run your tests with `POLARS_MAX_THREADS` set to the core count of production. A one-thread run hid the bug here.
3. Compare a few outputs between 1.x and 2.x on real data, row by row, not only the row count.

For how Polars compares with pandas and DuckDB on speed, see
[pandas vs Polars vs DuckDB in 2026](/blog/2026-10-03-pandas_polars_duckdb_2026/) (tested on Polars 1.44.2).

The scripts and their raw output: [first_purchase.py](/assets/code/polars_2_upgrade/first_purchase.py),
[first_purchase_out.txt](/assets/code/polars_2_upgrade/first_purchase_out.txt),
[probe.py](/assets/code/polars_2_upgrade/probe.py),
[probe_out_1.44.2.txt](/assets/code/polars_2_upgrade/probe_out_1.44.2.txt),
[probe_out_2.1.0.txt](/assets/code/polars_2_upgrade/probe_out_2.1.0.txt).
