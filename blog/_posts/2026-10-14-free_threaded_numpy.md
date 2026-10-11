---
layout: post
title: "Before benchmarking free-threaded Python, check what NumPy already does"
description: >
  On standard Python 3.15, four threads sped up NumPy about 3x but not a Python loop. On the free-threaded build the
  Python loop scaled about 3x, single-threaded code ran about 14% slower, and NumPy did not change.
image: /assets/img/blog/free_threaded_numpy/cover.png
sitemap:
  changefreq: daily
  priority: 1.0
---

# Before benchmarking free-threaded Python, check what NumPy already does

In our ordinary Python 3.15 build, four worker threads reduced a NumPy workload from 25.1 ms to 8.3 ms.
The GIL was still enabled. A pure Python workload did not get faster.

That is not evidence for or against free-threaded Python. It shows why “threads ran in parallel” is an insufficient
test of whether your interpreter is free-threaded: many numerical operations already release the GIL.

_Sources and both builds (standard and free-threaded Python 3.15.0) verified as of 2026-10-12._

Python 3.15.0 was released on October 9, 2026. Free-threading has existed as an optional build since 3.13; installing
3.15 does not itself mean you installed that build. [Python's release page](https://www.python.org/downloads/release/python-3150/)
and [free-threading guide](https://docs.python.org/3.15/howto/free-threading-python.html), checked 2026-10-12.

## Check the build and the running process

Run this after importing the packages your real application uses:

```python
import sys
import sysconfig
import numpy as np

print(sys.version.split()[0])
print("NumPy:", np.__version__)
print("free-threaded build:", bool(sysconfig.get_config_var("Py_GIL_DISABLED")))
print("GIL enabled now:", sys._is_gil_enabled())
```

Our executed output:

```text
3.15.0
NumPy: 2.5.3
free-threaded build: False
GIL enabled now: True
```

Build support and runtime state are different. An extension that has not declared free-threading support can cause
a free-threaded process to enable the GIL. Checking before imports can miss that change. The Python guide describes
these mechanisms; do not override compatibility warnings just to get a favorable benchmark.

## Compare work, not just thread counts

The [executed benchmark](/assets/code/hotweek/thread_benchmark.py) runs eight independent tasks with one, two, and four
workers. Each task either runs a generator expression in Python or creates its own numeric array and calls `sin` and
`sum` in NumPy. There are no shared mutable arrays and no matrix multiplication or BLAS workload.

```python
import numpy as np

def python_work(seed):
    return sum((i + seed) % 97 for i in range(1_000_000))

def numpy_work(seed):
    values = np.arange(1_000_000, dtype=np.float64) + seed
    return float(np.sin(values).sum())
```

The runner creates a thread pool outside the timed repetitions, warms it up, and checks every returned result against
the corresponding sequential result. Each configuration has seven measurements. Allocation time remains part of the
NumPy task because the task really creates an array.

```bash
python -m pip install numpy==2.5.3
python thread_benchmark.py --out threads.json
```

Tested on 2026-10-12 with Python 3.15.0, NumPy 2.5.3, and macOS 26.6.2 on Apple Silicon. Timing units below are
milliseconds; rows summarize the actual run, not a hardware-independent performance promise.

| Workload | Workers | Median | Minimum–maximum |
|---|---:|---:|---:|
| Python generator | 1 | 186.6 | 184.9–196.0 |
| Python generator | 2 | 190.0 | 185.1–196.3 |
| Python generator | 4 | 199.0 | 189.5–205.2 |
| NumPy sin + sum | 1 | 25.1 | 25.0–26.2 |
| NumPy sin + sum | 2 | 14.4 | 13.0–17.4 |
| NumPy sin + sum | 4 | 8.3 | 7.4–9.4 |

![Measured speedup versus one worker: the Python workload stays near one, while NumPy reaches about three with four workers; the process has the GIL enabled](/assets/img/blog/free_threaded_numpy/cover.png)

NumPy's documentation explains that many low-level operations release the GIL and recommends independent arrays
per worker. It also notes that operations on object arrays do not release it. Our float arrays are not a test of string
processing, pandas, or arbitrary Python objects. [NumPy thread-safety documentation](https://numpy.org/doc/stable/reference/thread_safety.html), checked 2026-10-12.

## What changes on the free-threaded build?

We installed the free-threaded build with `uv python install 3.15.0t`, added the same NumPy 2.5.3 (it installed as a
`cp315t` wheel), and ran the diagnostic above in that interpreter: `free-threaded build: True`, and `GIL enabled now:
False` even after importing NumPy. Then we ran the same benchmark script on both builds, alternating standard and
free-threaded, twice. Medians in milliseconds, first run / second run:

| Workload | Workers | Standard build (GIL on) | Free-threaded build (GIL off) |
|---|---:|---:|---:|
| Python generator | 1 | 184.5 / 182.5 | 209.1 / 209.2 |
| Python generator | 2 | 185.4 / 183.1 | 105.6 / 102.1 |
| Python generator | 4 | 184.5 / 183.6 | 62.5 / 51.7 |
| NumPy sin + sum | 1 | 24.0 / 25.2 | 24.2 / 24.2 |
| NumPy sin + sum | 2 | 12.7 / 13.0 | 12.6 / 12.7 |
| NumPy sin + sum | 4 | 7.4 / 7.5 | 9.9 / 7.4 |

- **Pure Python finally scales.** With four threads the generator workload took 52–63 ms instead of about 184 ms.
- **Single-threaded code paid for it.** One worker took 209 ms instead of about 183 ms, roughly 14% slower in this test.
- **NumPy did not care.** About 24 ms with one worker and 7.4 ms with four on both builds. The 9.9 ms median in the
  first free-threaded run did not repeat in the second.

Raw results: standard [run 1](/assets/code/hotweek/results/threads_compare/std_1.json) and [run 2](/assets/code/hotweek/results/threads_compare/std_2.json), free-threaded
[run 1](/assets/code/hotweek/results/threads_compare/ft_1.json) and [run 2](/assets/code/hotweek/results/threads_compare/ft_2.json). This is one machine, one script and two runs per build, not a
general verdict. The script does not measure memory, and for matrix operations you would also need to control
native-library thread pools, since nested parallelism can dominate the result.

The decision starts with where your time is spent. Pure Python execution and native numeric kernels expose different
bottlenecks. Our [loop-versus-vectorization experiment](/blog/2026-10-13-loop_vs_vectorization/) covers another part
of that distinction. Benchmark your workload before replacing the interpreter, and do not label a GIL-enabled run
as a free-threaded result.
