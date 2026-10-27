"""Cover for the frozendict post: every cell is the result of running the check below on Python 3.15.

Run: python3.15 fd_cover.py
"""
from types import MappingProxyType

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

BG, FG, MUTED, OK, BAD = "#1e1f26", "#e8e8ef", "#9a9ab0", "#4fb3bf", "#f2a65a"
KINDS = ["dict", "MappingProxyType", "frozendict"]


def make(kind, src):
    return {"dict": src, "MappingProxyType": MappingProxyType(src), "frozendict": frozendict(src)}[kind]


def can_be_cache_key(kind):
    try:
        hash(make(kind, {"lr": 0.1}))
        return True
    except TypeError:
        return False


def read_only(kind):
    try:
        make(kind, {"lr": 0.1})["lr"] = 0.2
        return False
    except TypeError:
        return True


def ignores_source_changes(kind):
    src = {"lr": 0.1}
    m = make(kind, src)
    src["lr"] = 0.2
    return m["lr"] == 0.1


def is_dict(kind):
    return isinstance(make(kind, {"lr": 0.1}), dict)


def dataframe_ok(kind):
    df = pd.DataFrame(make(kind, {"a": [1, 2], "b": [3, 4]}))
    return df.to_dict("list") == {"a": [1, 2], "b": [3, 4]}


def fillna_ok(kind):
    try:
        pd.DataFrame({"a": [None, 1.0]}).fillna(make(kind, {"a": 0.0}))
        return True
    except (TypeError, ValueError):
        return False


ROWS = [("usable as a cache key (hashable)", can_be_cache_key),
        ("m['k'] = v is blocked", read_only),
        ("unchanged when the source dict changes", ignores_source_changes),
        ("isinstance(m, dict)", is_dict),
        ("pd.DataFrame(m) builds the right table", dataframe_ok),
        ("df.fillna(m) accepted", fillna_ok)]
grid = [[check(k) for k in KINDS] for _, check in ROWS]
for (label, _), row in zip(ROWS, grid):
    print(f"{label:42s}", row)


def check_safe(fig, x0=80, x1=1120, y0=48, y1=582):
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    W, H = fig.canvas.get_width_height()
    bad = []
    for t in fig.findobj(matplotlib.text.Text):
        if not (t.get_visible() and t.get_text()):
            continue
        bb = t.get_window_extent(r)
        if bb.x0 < x0 or bb.x1 > x1 or H - bb.y1 < y0 or H - bb.y0 > y1:
            bad.append((t.get_text(), round(bb.x0), round(bb.x1), round(H - bb.y1), round(H - bb.y0)))
    print(f"cover.png: {W}x{H}, outside safe area: {bad or 'none'}")


fig = plt.figure(figsize=(12, 6.3), dpi=100, facecolor=BG)
px = lambda x, y: (x / 1200, 1 - y / 630)       # pixel position, origin top-left
fig.text(*px(80, 56), "frozendict in Python 3.15", fontsize=26, fontweight="bold", color=FG, va="top")
fig.text(*px(80, 104), "Each cell is a check run on Python 3.15.0 and pandas 3.1.0", fontsize=14, color=MUTED, va="top")
col_x = [680, 840, 1035]
for x, k in zip(col_x, KINDS):
    fig.text(*px(x, 170), k, fontsize=14, color=FG, ha="center", va="center", fontweight="bold",
             family="monospace")
for i, ((label, _), row) in enumerate(zip(ROWS, grid)):
    y = 228 + i * 58
    fig.text(*px(80, y), label, fontsize=17, color=FG, va="center")
    for x, ok in zip(col_x, row):
        fig.text(*px(x, y), "yes" if ok else "no", fontsize=20, color=OK if ok else BAD, ha="center",
                 va="center", fontweight="bold")
    fig.add_artist(plt.Line2D([80 / 1200, 1120 / 1200], [1 - (y + 29) / 630] * 2, color="#33343f", lw=1))
check_safe(fig)
fig.savefig("cover.png", dpi=100, facecolor=BG)
