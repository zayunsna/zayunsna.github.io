"""Cover for the Polars 2.0 post, drawn from the JSON files written by first_purchase.py."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BG, FG, MUTED, OK, BAD, GRID = "#1e1f26", "#e8e8ef", "#9a9ab0", "#4fb3bf", "#f2a65a", "#33343f"
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "axes.edgecolor": MUTED, "text.color": FG,
                     "xtick.color": FG, "ytick.color": FG, "font.size": 14})

QUERY, KEPT = "join -> first()", "join(maintain_order='left')"
ROWS = [  # label, json file, query
    ("Polars 1.44.2, 10 threads", "first_purchase_1.44.2_t10_auto.json", QUERY),
    ("Polars 2.1.0, 1 thread", "first_purchase_2.1.0_t1_auto.json", QUERY),
    ("Polars 2.1.0, 2 threads", "first_purchase_2.1.0_t2_auto.json", QUERY),
    ("Polars 2.1.0, 10 threads", "first_purchase_2.1.0_t10_auto.json", QUERY),
    ("2.1.0 + maintain_order='left'", "first_purchase_2.1.0_t10_auto.json", KEPT),
]


def check_safe(fig, x0=80, x1=1120, y0=48, y1=582):
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    W, H = fig.canvas.get_width_height()
    items = [t for t in fig.findobj(matplotlib.text.Text) if t.get_visible() and t.get_text()] + list(fig.axes)
    bad = []
    for it in items:
        bb = it.get_window_extent(r)
        if bb.x0 < x0 or bb.x1 > x1 or H - bb.y1 < y0 or H - bb.y0 > y1:
            bad.append((getattr(it, "get_text", lambda: "axes")(), round(bb.x0), round(bb.x1),
                        round(H - bb.y1), round(H - bb.y0)))
    print(f"cover.png: {W}x{H}, outside safe area: {bad or 'none'}")


fig = plt.figure(figsize=(12, 6.3), dpi=100)
fig.text(80 / 1200, 1 - 56 / 630, "Polars 2.0: same query, different answer each run", fontsize=26,
         fontweight="bold", va="top")
fig.text(80 / 1200, 1 - 104 / 630, "Customers whose \"first purchase\" came out wrong after a lazy join "
         "and first(). 10 runs each, Apple M5", fontsize=14, color=MUTED, va="top")
ax = fig.add_axes([420 / 1200, 1 - 520 / 630, 680 / 1200, 360 / 630])
for i, (label, path, query) in enumerate(ROWS):
    shares = json.load(open(path))[query]
    y = len(ROWS) - 1 - i
    color = BAD if max(shares) > 0 else OK
    ax.scatter(shares, [y] * len(shares), s=110, color=color, alpha=0.75, edgecolor=BG, zorder=3)
ax.set_yticks(range(len(ROWS)), [r[0] for r in ROWS][::-1], fontsize=15)
ax.set_xlim(-0.04, 1.04)
ax.set_ylim(-0.6, len(ROWS) - 0.4)
ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
ax.grid(axis="x", color=GRID)
ax.set_axisbelow(True)
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
ax.tick_params(axis="y", length=0)
check_safe(fig)
fig.savefig("cover.png", dpi=100)
