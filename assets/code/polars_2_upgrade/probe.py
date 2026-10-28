"""Run the same code on Polars 1.x and 2.x and print what changed.

Run: python probe.py  (once per Polars version)
"""
import io
import platform
import sys

import numpy as np
import polars as pl
import polars.selectors as cs

print(f"polars {pl.__version__}, python {sys.version.split()[0]}, {platform.machine()}")


def show(title, fn):
    print(f"\n[{title}]")
    try:
        fn()
    except Exception as e:
        print(f"  {type(e).__name__}: {str(e).splitlines()[0][:150]}")


# 1. Row order of a lazy left join.
rng = np.random.default_rng(0)
n = 2_000_000
orders = pl.DataFrame({"order_id": np.arange(n), "customer": rng.integers(0, 50_000, n)})
customers = pl.DataFrame({"customer": np.arange(50_000), "tier": rng.integers(0, 3, 50_000)})


def join_order():
    out = orders.lazy().join(customers.lazy(), on="customer", how="left").collect()
    ids = out["order_id"]
    print(f"  rows {out.height:,}, same order as input: {ids.equals(orders['order_id'])}, "
          f"first ids {ids.head(5).to_list()}")
    kept = orders.lazy().join(customers.lazy(), on="customer", how="left", maintain_order="left").collect()
    print(f"  maintain_order='left': same order as input: {kept['order_id'].equals(orders['order_id'])}")


show("1. lazy left join, row order", join_order)


# 2. Signed int + UInt64.
def int_uint():
    df = pl.DataFrame({"a": [9_007_199_254_740_993], "b": pl.Series([1], dtype=pl.UInt64)},
                      schema_overrides={"a": pl.Int64})
    out = df.select(total=pl.col("a") + pl.col("b"))
    print(f"  dtype {out.schema['total']}, value {out['total'][0]}  (exact answer 9007199254740994)")


show("2. Int64 + UInt64", int_uint)


# 3. Selector combined with pl.col.
def selector_and():
    df = pl.DataFrame({"x": [1, 2], "y": [3, 4], "flag": [True, False]})
    out = df.select(cs.numeric() & pl.col("x"))
    print("  columns", out.columns, "values", out.to_dicts())


show("3. cs.numeric() & pl.col('x')", selector_and)


# 4. Horizontal concat of different heights.
def concat_h():
    out = pl.concat([pl.DataFrame({"a": [1, 2, 3]}), pl.DataFrame({"b": [10, 20]})], how="horizontal")
    print("  ", out.to_dicts())


show("4. concat(how='horizontal') with heights 3 and 2", concat_h)


# 5. explode on an empty list.
def explode_empty():
    out = pl.DataFrame({"id": [1, 2], "tags": [["a", "b"], []]}).explode("tags")
    print("  rows", out.height, out.to_dicts())


show("5. explode() with an empty list", explode_empty)


# 6. is_in with int column and float values.
def is_in_float():
    out = pl.DataFrame({"x": [1, 2, 3]}).filter(pl.col("x").is_in([1.0, 2.5]))
    print("  ", out["x"].to_list())


show("6. int column .is_in([1.0, 2.5])", is_in_float)


# 7. melt.
show("7. df.melt()", lambda: print("  ", pl.DataFrame({"id": [1], "a": [2]}).melt(id_vars="id").to_dicts()))
