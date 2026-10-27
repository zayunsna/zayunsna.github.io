"""frozendict (PEP 814, Python 3.15): what it fixes and what it does not freeze.

Run: python3.15 frozendict_demo.py
"""
import copy
import functools
import json
import pickle
import sys
import timeit
from collections.abc import Mapping
from types import MappingProxyType

import pandas as pd

print(f"python {sys.version.split()[0]}, pandas {pd.__version__}")


def section(title):
    print(f"\n[{title}]")


# 1. A config dict cannot be a cache key; a frozendict can.
section("1. caching a function that takes a config")
calls = 0


@functools.cache
def train(params):
    global calls
    calls += 1
    return f"model(lr={params['lr']}, depth={params['depth']})"


try:
    train({"lr": 0.1, "depth": 6})
except TypeError as e:
    print("dict       ->", type(e).__name__ + ":", e)

train(frozendict(lr=0.1, depth=6))
train(frozendict(depth=6, lr=0.1))          # same items, different order
train(frozendict({"lr": 0.1, "depth": 6}))  # built from a dict
print("frozendict -> 3 calls, function body ran", calls, "time(s);", train.cache_info())

# 2. Equality and hashing ignore order, and a frozendict equals a dict with the same items.
section("2. equality and hashing")
a, b = frozendict(x=1, y=2), frozendict(y=2, x=1)
print("a == b:", a == b, "| hash equal:", hash(a) == hash(b), "| a == dict(x=1, y=2):", a == dict(x=1, y=2))
print("list(a):", list(a), "list(b):", list(b), "(insertion order kept)")

# 3. MappingProxyType is read-only, but it is a view: the owner can still change it, and it is not hashable.
section("3. MappingProxyType vs frozendict")
settings = {"currency": "USD"}
view = MappingProxyType(settings)
frozen = frozendict(settings)
settings["currency"] = "EUR"                # someone mutates the original dict
print("proxy sees:", view["currency"], "| frozendict sees:", frozen["currency"])
for name, obj in [("proxy", view), ("frozendict", frozen)]:
    try:
        obj["currency"] = "JPY"
    except TypeError as e:
        print(f"{name} assignment -> TypeError: {e}")
try:
    hash(view)
except TypeError as e:
    print("hash(proxy) -> TypeError:", e)
print("hash(frozendict) works:", isinstance(hash(frozen), int))

# 4. Mutable default argument.
section("4. default arguments")


def add_tag_dict(row, defaults={}):
    defaults.setdefault("seen", 0)
    defaults["seen"] += 1                   # bug: the default is shared between calls
    return {**defaults, **row}


def add_tag_frozen(row, defaults=frozendict(source="api")):
    return {**defaults, **row}


print("dict default:", [add_tag_dict({"id": i})["seen"] for i in range(3)])
print("frozendict default:", [add_tag_frozen({"id": i}) for i in range(2)])

# 5. Only the top level is frozen.
section("5. shallow immutability")
cfg = frozendict(features=["age", "income"])
cfg["features"].append("zip_code")          # the list inside is still a list
print("after append:", cfg)
try:
    hash(cfg)
except TypeError as e:
    print("hash(cfg) -> TypeError:", e)
deep = frozendict(features=("age", "income"))
print("with a tuple value, hash works:", isinstance(hash(deep), int))

# 6. "Changing" a frozendict means making a new one.
section("6. updates create new objects")
base = frozendict(lr=0.1, depth=6)
tuned = base | {"lr": 0.05}
print("base:", base, "| tuned:", tuned, "| type:", type(tuned).__name__)
alias = base
base |= {"depth": 8}                        # rebinds the name, alias keeps the old object
print("after |= : base:", base, "| alias:", alias)
print("copy.copy returns the same object:", copy.copy(alias) is alias, "| .copy():", alias.copy() is alias)
try:
    alias.update(depth=10)
except AttributeError as e:
    print("update() -> AttributeError:", e)

# 7. Where code checks for dict, frozendict is not one.
section("7. working with other code")
fd = frozendict(name="ada", score=0.93)
print("isinstance dict:", isinstance(fd, dict), "| isinstance Mapping:", isinstance(fd, Mapping))
print("json.dumps:", end=" ")
try:
    print(json.dumps(fd))
except TypeError as e:
    print("TypeError:", e)
print("json.dumps(dict(fd)):", json.dumps(dict(fd)))
print("pickle round trip equal:", pickle.loads(pickle.dumps(fd)) == fd,
      "| type:", type(pickle.loads(pickle.dumps(fd))).__name__)
print("f(**fd):", (lambda name, score: f"{name}={score}")(**fd))

section("8. pandas: works, fails, or silently wrong")
df = pd.DataFrame({"city": ["Seoul", "Busan", None], "sales": [3.0, None, 5.0]})
checks = {
    "pd.Series(fd)": lambda m: pd.Series(m(a=1, b=2)).to_dict(),
    "Series.map(fd)": lambda m: df["city"].map(m(Seoul="KR-11", Busan="KR-26")).tolist(),
    "rename(columns=fd)": lambda m: list(df.rename(columns=m(city="region")).columns),
    "astype(fd)": lambda m: str(df.astype(m(sales="float32"))["sales"].dtype),
    "fillna(fd)": lambda m: df.fillna(m(sales=0.0))["sales"].tolist(),
    "groupby.agg(fd)": lambda m: df.groupby("city").agg(m(sales="sum"))["sales"].to_dict(),
    "pd.DataFrame(fd)": lambda m: pd.DataFrame(m(a=[1, 2], b=[3, 4])).to_dict("list"),
}
for name, check in checks.items():
    results = []
    for mapping in (dict, frozendict):
        try:
            results.append(repr(check(mapping)))
        except Exception as e:
            results.append(f"{type(e).__name__}: {e}")
    same = "same" if results[0] == results[1] else "DIFFERENT"
    print(f"  {name:20s} {same}\n    dict:       {results[0]}\n    frozendict: {results[1]}")

# 8. Cost: building, lookups, hashing.
section("9. cost (timeit, best of 5, Apple M5)")
src = {f"k{i}": i for i in range(20)}
fdict = frozendict(src)
n = 1_000_000
rows = [
    ("build from a 20-key dict", "dict(src)", "frozendict(src)"),
    ("lookup d['k7']", "d['k7']", "f['k7']"),
]
g = {"src": src, "d": dict(src), "f": fdict, "frozendict": frozendict}
for label, s_dict, s_fd in rows:
    t_d = min(timeit.repeat(s_dict, globals=g, number=n, repeat=5)) / n * 1e9
    t_f = min(timeit.repeat(s_fd, globals=g, number=n, repeat=5)) / n * 1e9
    print(f"  {label:28s} dict {t_d:6.1f} ns   frozendict {t_f:6.1f} ns")
fresh = min(timeit.repeat("hash(frozendict(src))", globals=g, number=n // 10, repeat=5)) / (n // 10) * 1e9
build = min(timeit.repeat("frozendict(src)", globals=g, number=n // 10, repeat=5)) / (n // 10) * 1e9
again = min(timeit.repeat("hash(f)", globals=g, number=n, repeat=5)) / n * 1e9
print(f"  hash of a new 20-key frozendict  ~{fresh - build:6.1f} ns (build+hash {fresh:.1f} minus build {build:.1f})")
print(f"  hash of the same object again    {again:6.1f} ns")
for size in (1_000, 100_000):
    big = frozendict({i: i for i in range(size)})
    hash(big)
    t = min(timeit.repeat("hash(big)", globals={"big": big}, number=n, repeat=5)) / n * 1e9
    print(f"  hash again, {size:>7,} keys         {t:6.1f} ns")
print(f"  sys.getsizeof: dict {sys.getsizeof(dict(src))} bytes, frozendict {sys.getsizeof(fdict)} bytes")
