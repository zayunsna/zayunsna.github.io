"""Measure independent CPU tasks; do not infer free-threaded performance from a GIL build."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import platform
import statistics
import sys
import sysconfig
import time
from pathlib import Path

import numpy as np


def python_work(seed):
    return sum((i + seed) % 97 for i in range(1_000_000))


def numpy_work(seed):
    values = np.arange(1_000_000, dtype=np.float64) + seed
    return float(np.sin(values).sum())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    info = {'python': sys.version, 'numpy': np.__version__, 'platform': platform.platform(),
            'free_threaded_build': bool(sysconfig.get_config_var('Py_GIL_DISABLED')),
            'gil_enabled': sys._is_gil_enabled()}
    print(json.dumps(info), flush=True)
    results = []
    for name, fn in [('python', python_work), ('numpy', numpy_work)]:
        expected = [fn(i) for i in range(8)]
        for workers in (1, 2, 4):
            samples = []
            with ThreadPoolExecutor(max_workers=workers) as pool:
                list(pool.map(fn, range(8)))
                for _ in range(7):
                    start = time.perf_counter()
                    actual = list(pool.map(fn, range(8)))
                    samples.append(time.perf_counter() - start)
                    assert actual == expected
            row = {'task': name, 'workers': workers, 'median': statistics.median(samples), 'min': min(samples), 'max': max(samples)}
            results.append(row)
            print(json.dumps(row), flush=True)
    args.out.write_text(json.dumps({'environment': info, 'results': results}, indent=2))


if __name__ == '__main__':
    main()
