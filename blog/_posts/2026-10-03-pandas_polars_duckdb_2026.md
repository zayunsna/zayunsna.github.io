---
layout: post
title: "pandas vs Polars vs DuckDB in 2026: a real benchmark on 10 million rows"
description: >
  pandas 3.0.6, Polars 1.44.2 and DuckDB 1.5.5 on the same 10M-row dataset: read, group by, filter and join,
  plus a single-thread run that shows how much of the gap is just parallelism.
image: /assets/img/blog/pandas_polars_duckdb_2026/cover.png
sitemap:
  changefreq: daily
  priority: 1.0
---

# pandas vs Polars vs DuckDB in 2026: a real benchmark on 10 million rows

**TL;DR** — On 10 million rows, Polars and DuckDB were 2× to 60× faster than pandas at group by, filter and join.
Much of that gap comes from using all CPU cores: limited to one thread, Polars was only about 1.4× to 1.6× faster than pandas.
DuckDB's join stood out, staying fast even on a single thread.

_Tested on 2026-09-28 with pandas 3.0.6 (released 2026-09-17), Polars 1.44.2 (2026-09-09), DuckDB 1.5.5 (2026-07-22),
pyarrow 25.0.1 and Python 3.12, on an Apple M5 with 10 cores and 16 GB of memory._

## Key points

- **Polars was fastest at reading Parquet and simple aggregations.** DuckDB was fastest at the join.
- **pandas was slowest on every task except reading,** where DuckDB was slightly slower.
- **Parallelism explains most of the gap.** At one thread, Polars' group by went from 11 ms to 59 ms, against pandas' 96 ms.
- **All three returned the same numbers.** We checked the join result against each other before comparing times.
- **Speed at this size is rarely the deciding factor.** Every task here finished in under 0.3 seconds, even in pandas.

## Results

Each task ran 5 times after a warm-up; the table shows the median.

| Task (10M rows) | pandas 3.0.6 | Polars 1.44.2 | DuckDB 1.5.5 |
|---|---|---|---|
| Read Parquet (37 MB) | 51.2 ms | **13.3 ms** | 60.5 ms |
| Group by 50 categories (sum, mean, count) | 95.7 ms | **10.8 ms** | 14.1 ms |
| Filter (`amount > 100`) + group by | 17.4 ms | **4.2 ms** | 8.7 ms |
| Join 1,000 stores + group by region | 286.4 ms | 51.7 ms | **4.8 ms** |

![Four bar charts of median time in milliseconds on 10 million rows. Read parquet: pandas 51, Polars 13, DuckDB 61. Group by: pandas 96, Polars 11, DuckDB 14. Filter plus group by: pandas 17, Polars 4.2, DuckDB 8.7. Join plus group by: pandas 286, Polars 52, DuckDB 4.8](/assets/img/blog/pandas_polars_duckdb_2026/benchmark.png)

"Read" means something slightly different for DuckDB: it copies the file into DuckDB's own in-memory table,
while pandas and Polars build a DataFrame.

## What does the code look like in each library?

The same "join, then sum by region" task in all three:

```python
# pandas
sales_pd.merge(stores_pd, on="store_id").groupby("region")["amount"].sum()

# Polars (eager API)
sales_pl.join(stores_pl, on="store_id").group_by("region").agg(pl.col("amount").sum())

# DuckDB (SQL on tables in its in-memory database)
con.execute("SELECT region, sum(amount) FROM sales JOIN stores USING (store_id) GROUP BY region").arrow()
```

All three returned identical sums per region. For example, the first two regions came to `40041037.61` and `39977611.71`.

The data is synthetic: 10 million sales rows with a store ID (1,000 stores), a category (50 values) and an amount,
plus a 1,000-row store table with 10 regions. The full scripts are at the end of this post.

## How much of the speedup is just multithreading?

A lot of it. In our run, pandas' group by and merge used a single core (CPU time ≈ wall time),
while Polars and DuckDB used all 10. pandas' Parquet reading is the exception: pyarrow reads with several threads.
Limiting Polars and DuckDB to one thread gives a fairer engine-to-engine comparison:

```
Polars 1 thread  group by           59.4 ms
Polars 1 thread  join + group by   206.0 ms
DuckDB 1 thread  group by           79.8 ms
DuckDB 1 thread  join + group by    21.0 ms
```

| One thread | pandas | Polars | DuckDB |
|---|---|---|---|
| Group by | 95.7 ms | 59.4 ms | 79.8 ms |
| Join + group by | 286.4 ms | 206.0 ms | 21.0 ms |

On one thread, Polars' group by is about 1.6× faster than pandas, not 9×. DuckDB's join is the exception:
it stays more than 10× faster than pandas even on one core.
(Polars was limited with `POLARS_MAX_THREADS=1` and DuckDB with `SET threads = 1`.)

## Which one should you use?

| If you… | Consider |
|---|---|
| Have existing pandas code, small or medium data, many library integrations | pandas 3.0 |
| Want the fastest DataFrame API and are starting fresh | Polars |
| Think in SQL, join many tables, or query Parquet files directly | DuckDB |
| Need both | DuckDB can query Polars and pandas DataFrames directly |

This is one machine, one synthetic dataset and four tasks. Your data types, sizes and queries may give a different ranking,
so time your own workload before switching.

## Full benchmark code

<details markdown="1">
<summary>Data generation and benchmark scripts</summary>

```python
# make_data.py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

rng = np.random.default_rng(0)
n = 10_000_000
sales = pa.table({
    "store_id": rng.integers(0, 1_000, n),
    "category": pa.array(rng.choice([f"cat_{i:02d}" for i in range(50)], n)),
    "amount": rng.gamma(2.0, 20.0, n).round(2),
})
stores = pa.table({"store_id": np.arange(1_000), "region": [f"region_{i % 10}" for i in range(1_000)]})
pq.write_table(sales, "sales.parquet")
pq.write_table(stores, "stores.parquet")
```

```python
# bench.py
import statistics, time
import duckdb
import pandas as pd
import polars as pl

def timed(fn, runs=5):
    fn()                                            # warm-up
    times = []
    for _ in range(runs):
        t0 = time.perf_counter(); out = fn(); times.append(time.perf_counter() - t0)
    return statistics.median(times) * 1000, out

def pd_read(): return pd.read_parquet("sales.parquet")
sales_pd, stores_pd = pd_read(), pd.read_parquet("stores.parquet")
pandas = {
    "read parquet": pd_read,
    "group by": lambda: sales_pd.groupby("category")["amount"].agg(["sum", "mean", "count"]),
    "filter + group by": lambda: sales_pd[sales_pd["amount"] > 100].groupby("category")["amount"].sum(),
    "join + group by": lambda: sales_pd.merge(stores_pd, on="store_id").groupby("region")["amount"].sum(),
}

def pl_read(): return pl.read_parquet("sales.parquet")
sales_pl, stores_pl = pl_read(), pl.read_parquet("stores.parquet")
polars = {
    "read parquet": pl_read,
    "group by": lambda: sales_pl.group_by("category").agg(pl.col("amount").sum().alias("sum"),
                                                          pl.col("amount").mean().alias("mean"),
                                                          pl.len().alias("count")),
    "filter + group by": lambda: sales_pl.filter(pl.col("amount") > 100).group_by("category").agg(pl.col("amount").sum()),
    "join + group by": lambda: sales_pl.join(stores_pl, on="store_id").group_by("region").agg(pl.col("amount").sum()),
}

con = duckdb.connect()
con.execute("CREATE TABLE stores AS SELECT * FROM 'stores.parquet'")
con.execute("CREATE TABLE sales AS SELECT * FROM 'sales.parquet'")
def q(sql): return lambda: con.execute(sql).arrow()
duck = {
    "read parquet": lambda: con.execute("CREATE OR REPLACE TABLE s2 AS SELECT * FROM 'sales.parquet'"),
    "group by": q("SELECT category, sum(amount), avg(amount), count(*) FROM sales GROUP BY category"),
    "filter + group by": q("SELECT category, sum(amount) FROM sales WHERE amount > 100 GROUP BY category"),
    "join + group by": q("SELECT region, sum(amount) FROM sales JOIN stores USING (store_id) GROUP BY region"),
}

for lib, tasks in [("pandas", pandas), ("Polars", polars), ("DuckDB", duck)]:
    for task, fn in tasks.items():
        ms, _ = timed(fn)
        print(f"{lib:<7} {task:<18} {ms:8.1f} ms")
```

</details>

## Sources

- [pandas 3.0.6](https://pypi.org/project/pandas/3.0.6/), [Polars 1.44.2](https://pypi.org/project/polars/1.44.2/), [DuckDB 1.5.5](https://pypi.org/project/duckdb/1.5.5/) — PyPI, release dates checked 2026-09-28
- [Polars lazy API user guide](https://docs.pola.rs/user-guide/lazy/) — the lazy API can optimize further than the eager API used here
- [DuckDB: integration with Polars](https://duckdb.org/docs/stable/guides/python/polars)

Related: [What breaks when you upgrade to pandas 3.0?](/ds/2026-09-28-pandas_3_upgrade/)
