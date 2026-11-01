"""Cover for the preprocessing pipeline post, drawn from results/scores.json written by pipeline_tutorial.py.

Run from the folder that contains results/: python pp_cover.py
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

BG, PANEL, FG, MUTED, NUM, CAT, WARN = "#172026", "#22303a", "#eef2f4", "#9fb0ba", "#7cc4e8", "#b9a3f0", "#f2a65a"
s = json.load(open("results/scores.json"))
W, H = 1200, 630
fig = plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor=BG)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(H, 0)  # pixel coordinates, y down
ax.axis("off")


def box(x0, y0, x1, y1, color=PANEL):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=14",
                                fc=color, ec="none"))


def arrow(x0, x1, y):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=26, lw=3, color=MUTED))


def text(x, y, s_, **kw):
    kw = {"color": FG, "fontsize": 18, "va": "top", "ha": "left"} | kw
    return ax.text(x, y, s_, **kw)


text(84, 52, "One Pipeline: raw row in, prediction out", fontsize=30, fontweight="bold")
text(84, 104, "scikit-learn ColumnTransformer on UCI Adult", fontsize=18, color=MUTED)

box(84, 160, 330, 440)
text(104, 178, "raw row", color=MUTED)
for i, (k, v, c) in enumerate([("age", "45", FG), ("workclass", "blank", WARN),
                               ("hours", "blank", WARN), ("country", "Brazil", WARN)]):
    text(104, 228 + i * 50, k, fontsize=17)
    text(312, 228 + i * 50, v, fontsize=17, ha="right", color=c, fontweight="bold")
arrow(338, 384, 300)

box(392, 160, 778, 292)
text(412, 176, f"5 numeric columns", fontsize=19, color=NUM, fontweight="bold")
text(412, 214, "fill with train median", fontsize=17)
text(412, 248, "standardize", fontsize=17)
box(392, 308, 778, 440)
text(412, 324, "7 text columns", fontsize=19, color=CAT, fontweight="bold")
text(412, 362, 'fill with "missing", one-hot', fontsize=17)
text(412, 396, "country: rare/unseen → group", fontsize=17)
arrow(786, 832, 300)

box(840, 160, 1116, 440)
text(860, 178, f"{s['n_features']} features", color=MUTED)
text(860, 212, "logistic regression", color=MUTED)
text(860, 268, "P(>50K)", fontsize=22)
text(860, 304, f"{s['new_rows_p_over_50k'][0]:.2f}", fontsize=58, fontweight="bold", color=WARN)
text(860, 386, "Brazil: never seen", fontsize=17, color=MUTED)

text(84, 478, f"Held-out test, {s['n_test']:,} rows:", fontsize=20, color=MUTED)
text(84, 516, f"accuracy {s['pipeline_accuracy']:.1%}", fontsize=28, fontweight="bold")
text(440, 522, f"vs {s['baseline_accuracy']:.1%} for always “<=50K”", fontsize=22, color=MUTED)


def check_safe(x0=80, x1=1120, y0=48, y1=582):
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    bad = []
    for t in ax.texts + ax.patches:
        bb = t.get_window_extent(r)
        if bb.x0 < x0 or bb.x1 > x1 or H - bb.y1 < y0 or H - bb.y0 > y1:
            bad.append((getattr(t, "get_text", lambda: type(t).__name__)(), round(bb.x0), round(bb.x1),
                        round(H - bb.y1), round(H - bb.y0)))
    boxes = [(t.get_text(), t.get_window_extent(r)) for t in ax.texts]
    overlaps = [(a, b) for i, (a, ba) in enumerate(boxes) for b, bb in boxes[i + 1:] if ba.overlaps(bb)]
    print(f"cover.png: {W}x{H}, {len(ax.texts)} texts + {len(ax.patches)} shapes, outside safe area: {bad or 'none'}, "
          f"overlapping texts: {overlaps or 'none'}")


check_safe()
fig.savefig("cover.png", dpi=100, facecolor=BG)
