"""A pipeline that relies on row order: first purchase per customer after a lazy join.

Run: python first_purchase.py
Set POLARS_MAX_THREADS or POLARS_ENGINE_AFFINITY before running to compare.
"""
import json
import os
import platform
import statistics
import sys
import time

import numpy as np
import polars as pl

RUNS = 10
print(f"polars {pl.__version__}, python {sys.version.split()[0]}, {platform.machine()}, "
      f"threads {pl.thread_pool_size()}, "
      f"POLARS_ENGINE_AFFINITY={os.environ.get('POLARS_ENGINE_AFFINITY', '(unset)')}")

rng = np.random.default_rng(0)
N, CUSTOMERS = 2_000_000, 50_000
# Transactions are already sorted by time: row i happened before row i + 1.
tx = pl.DataFrame({
    "ts": np.arange(N),
    "customer": rng.integers(0, CUSTOMERS, N),
    "amount": rng.integers(1, 500, N),
})
customers = pl.DataFrame({"customer": np.arange(CUSTOMERS), "region": rng.integers(0, 5, CUSTOMERS)})
truth = tx.group_by("customer").agg(pl.col("ts").min().alias("true_first_ts"))


def first_purchase(agg=pl.col("ts").first(), **join_kw):
    return (tx.lazy()
            .join(customers.lazy(), on="customer", how="left", **join_kw)
            .group_by("customer")
            .agg(agg.alias("first_ts")))


def wrong_share(df):
    j = df.join(truth, on="customer")
    return (j["first_ts"] != j["true_first_ts"]).mean()


CASES = [
    ("join -> first()", first_purchase(), {}),
    ("same, engine='in-memory'", first_purchase(), {"engine": "in-memory"}),
    ("join(maintain_order='left')", first_purchase(maintain_order="left"), {}),
    ("min('ts') instead of first()", first_purchase(pl.col("ts").min()), {}),
    ("no join, group_by -> first()",
     tx.lazy().group_by("customer").agg(pl.col("ts").first().alias("first_ts")), {}),
]
results = {}
print(f"\n{'query':30s} {'wrong first purchase over ' + str(RUNS) + ' runs':>36s}  median time")
for label, q, kw in CASES:
    shares, times = [], []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        out = q.collect(**kw)
        times.append(time.perf_counter() - t0)
        shares.append(wrong_share(out))
    print(f"{label:30s} min {min(shares):6.1%}  median {statistics.median(shares):6.1%}  "
          f"max {max(shares):6.1%}  {statistics.median(times) * 1000:6.0f} ms")
    results[label] = shares

tag = f"{pl.__version__}_t{pl.thread_pool_size()}_{os.environ.get('POLARS_ENGINE_AFFINITY', 'auto')}"
json.dump(results, open(f"first_purchase_{tag}.json", "w"), indent=1)
