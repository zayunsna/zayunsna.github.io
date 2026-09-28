---
layout: post
title: "What breaks when you upgrade to pandas 3.0?"
description: >
  Chained assignment stops updating your DataFrame, strings get a new dtype, and datetimes switch to
  microseconds. Tested side by side on pandas 2.3.3 and 3.0.6.
image: /assets/img/ds/pandas_3_upgrade/cover.png
sitemap:
  changefreq: daily
  priority: 1.0
---

# What breaks when you upgrade to pandas 3.0?

**TL;DR** — The most dangerous change in pandas 3.0 is that chained assignment like `df["col"][mask] = 0`
no longer updates `df`. You get a warning, not an error, so a script can keep running with wrong data.
Also check any code that relies on `object` dtype for strings or converts datetimes to integers.

_Tested on 2026-09-28 with pandas 2.3.3 and 3.0.6 (released 2026-09-17), Python 3.12, pyarrow 25.0.1.
Where outputs are labeled `2.3.3 ->` and `3.0.6 ->`, the same code was run on each version._

## Key points

- **Copy-on-Write is always on.** Anything you get from indexing behaves like a copy, so editing it never changes the original DataFrame.
- **Chained assignment stops working.** `SettingWithCopyWarning` is gone. You now get a `ChainedAssignmentError` warning instead.
- **Strings get their own dtype, `str`.** It uses PyArrow when installed. Missing values become `NaN` instead of `None`.
- **Datetimes default to microseconds.** `astype("int64")` on a datetime now returns a number 1000× smaller.
- **Upgrade through 2.3 first.** The pandas team recommends fixing all warnings on 2.3 before moving to 3.0.

## Does chained assignment still work in pandas 3.0?

No. Any assignment that goes through two indexing steps now edits a temporary copy, not your DataFrame.

```python
import pandas as pd

df = pd.DataFrame({"price": [10, 20, 30], "qty": [1, 2, 3]})
df["price"][df["qty"] > 1] = 0          # column first, then mask
print(df["price"].tolist())
```

```
2.3.3 -> [10, 0, 0]
3.0.6 -> [10, 20, 30]
```

On 3.0.6 the values stay unchanged. pandas prints a `ChainedAssignmentError` warning, but the script keeps running.
The fix is a single `.loc` call, which gives `[10, 0, 0]` on both versions:

```python
df.loc[df["qty"] > 1, "price"] = 0
```

## Can editing a column change the original DataFrame?

Not anymore, and this change comes **without any warning** on either version.
If you take a column out as a Series and modify it, pandas 2 wrote the change through to the DataFrame:

```python
df = pd.DataFrame({"a": [1, 2, 3]})
col = df["a"]
col.iloc[0] = 100
print(df["a"].tolist())
```

```
2.3.3 -> [100, 2, 3]
3.0.6 -> [1, 2, 3]
```

Code that relied on this side effect breaks silently. Code that called `.copy()` just to avoid it can drop the call.

## What changed for string columns?

String data now gets the `str` dtype instead of `object`, and missing values are `NaN`:

```python
s = pd.Series(["apple", "banana", None])
print(s.dtype, repr(s[2]))
```

```
2.3.3 -> object None
3.0.6 -> str nan
```

This breaks checks like `s.dtype == object` and `s[2] is None`. In return, string columns get much
smaller and faster. Here is one million short strings on pandas 3.0.6, old `object` dtype vs new `str` dtype:

```python
import time
import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
words = np.array(["apple", "banana", "cherry", "durian", "elderberry"])
values = list(rng.choice(words, size=1_000_000))

def measure(dtype):
    s = pd.Series(values, dtype=dtype)
    mem_mb = s.memory_usage(deep=True) / 1e6
    t0 = time.perf_counter()
    for _ in range(5):
        s.str.contains("an")
    return mem_mb, (time.perf_counter() - t0) / 5 * 1000

for dtype in ["object", "str"]:
    mem, ms = measure(dtype)
    print(f"{dtype:>6}: {mem:6.1f} MB | str.contains: {ms:6.1f} ms")
```

```
object:   79.6 MB | str.contains:   87.6 ms
   str:   14.6 MB | str.contains:   27.6 ms
```

![Bar charts of a 1M-row string column: 79.6 MB as object dtype vs 14.6 MB as str dtype, and str.contains at about 88 ms vs 26 ms](/assets/img/ds/pandas_3_upgrade/string_dtype_benchmark.png)

That is about **5× less memory and 3× faster** `str.contains` on an Apple M5 (16 GB).
Timings vary by about ±1 ms between runs. Without pyarrow installed, `str` falls back to `object`
storage and you won't see this gain.

## Why are my timestamps 1000× smaller?

Parsed datetimes now default to microsecond resolution (`datetime64[us]`) instead of nanoseconds:

```python
ts = pd.to_datetime(pd.Series(["2026-09-28 12:00"]))
print(ts.dtype, ts.astype("int64").iloc[0])
print(ts.dt.as_unit("ns").astype("int64").iloc[0])
```

```
2.3.3 -> datetime64[ns] 1790596800000000000
         1790596800000000000
3.0.6 -> datetime64[us] 1790596800000000
         1790596800000000000
```

If you store epoch integers or pass them to another system, call `.dt.as_unit("ns")` before casting.

## Upgrade checklist

| Check | Search your code for | Fix |
|---|---|---|
| Chained assignment | `df["x"][...] =`, `df[...]["x"] =` | One `.loc[rows, "x"] =` |
| Relying on write-through | Editing a column Series or a slice | Assign back to `df` explicitly |
| String dtype checks | `dtype == object`, `is None` on strings | On 3.0: `pd.api.types.is_string_dtype(s)` and `pd.isna(v)` |
| Datetime to int | `.astype("int64")` on datetimes | `.dt.as_unit("ns")` first |
| Python version | Python 3.10 or older | pandas 3.0 needs Python 3.11+ |

Note that `is_string_dtype` returned `False` on 2.3.3 for the `object` Series above (it contains `None`),
and `True` on 3.0.6. Don't use it as a version-independent check.

The safest path is the one the pandas team recommends: move to 2.3, fix every warning, then move to 3.0.

## Sources

- [What's new in 3.0.0 (January 21, 2026)](https://pandas.pydata.org/docs/whatsnew/v3.0.0.html) — pandas docs, checked 2026-09-28
- [Migration guide for the new string data type](https://pandas.pydata.org/docs/user_guide/migration-3-strings.html) — pandas docs, checked 2026-09-28
- [Copy-on-Write (CoW)](https://pandas.pydata.org/docs/user_guide/copy_on_write.html) — pandas docs, checked 2026-09-28
- [pandas 3.0.6 on PyPI](https://pypi.org/project/pandas/3.0.6/) — released 2026-09-17
