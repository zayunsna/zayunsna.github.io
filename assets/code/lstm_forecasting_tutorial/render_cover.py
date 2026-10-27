"""Render the tutorial cover from lstm_tutorial.py outputs, not invented data.

Usage: python render_cover.py results cover.png
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.text import Text
import pandas as pd

results, destination = Path(sys.argv[1]), Path(sys.argv[2])
data = pd.read_csv(results / "predictions.csv", parse_dates=["date"])
scores = pd.read_csv(results / "scores.csv", index_col=0).iloc[:, 0]
bg, fg, muted, teal = "#151c2c", "#e8eef8", "#9aa8bf", "#4cc5bc"
fig = plt.figure(figsize=(12, 6.3), dpi=100, facecolor=bg)
fig.text(.075, .875, "Train your first LSTM", fontsize=29, weight="bold", color=fg)
fig.text(.075, .815, "A PyTorch tutorial, from CSV to an honest forecast", fontsize=16, color=muted)
for x, title, detail in [(.075, "1  Daily CSV", "real bike rentals"),
                         (.38, "2  14-day windows", "one feature / CPU"),
                         (.685, "3  Predict tomorrow", "check a baseline")]:
    fig.add_artist(FancyBboxPatch((x, .64), .24, .11, boxstyle="round,pad=0.008",
                                  transform=fig.transFigure, facecolor="#233249", edgecolor="none"))
    fig.text(x + .012, .705, title, fontsize=15, color=fg, weight="bold")
    fig.text(x + .012, .665, detail, fontsize=12, color=muted)
for x in [.345, .65]:
    fig.text(x, .685, ">", fontsize=20, color=teal, ha="center")
fig.text(.145, .585, "Last 42 test days", fontsize=12, color=muted)
ax = fig.add_axes([.145, .22, .765, .335], facecolor=bg)
tail = data.tail(42)
ax.plot(tail.date, tail.actual, color=fg, lw=1.5, label="Actual")
ax.plot(tail.date, tail.LSTM, color=teal, lw=2, label="LSTM")
ax.set_ylabel("Rentals / day", color=muted, fontsize=12)
ax.tick_params(colors=muted, labelsize=11)
ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax.grid(alpha=.12)
for spine in ax.spines.values():
    spine.set_color("#3a4660")
ax.legend(loc="upper right", facecolor=bg, edgecolor="none", labelcolor=fg,
          fontsize=12, ncol=2)
fig.text(.075, .095, f"Test MAE: LSTM {scores['LSTM']:,.0f}  /  yesterday {scores['Yesterday']:,.0f}",
         fontsize=16, color=fg)
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
for item in fig.findobj(match=Text):
    if not item.get_visible() or not item.get_text().strip():
        continue
    bounds = item.get_window_extent(renderer)
    # Matplotlib keeps off-screen tick objects: only inspect ticks inside the axis.
    if item in ax.get_xticklabels():
        tick = ax.xaxis.get_major_ticks()[ax.get_xticklabels().index(item)]
        if not ax.get_xlim()[0] <= tick.get_loc() <= ax.get_xlim()[1]:
            continue
    if item in ax.get_yticklabels():
        tick = ax.yaxis.get_major_ticks()[ax.get_yticklabels().index(item)]
        if not ax.get_ylim()[0] <= tick.get_loc() <= ax.get_ylim()[1]:
            continue
    assert bounds.x0 >= 80 and bounds.x1 <= 1120, (item.get_text(), bounds)
    assert bounds.y0 >= 48 and bounds.y1 <= 582, (item.get_text(), bounds)
assert ax.bbox.x0 >= 80 and ax.bbox.x1 <= 1120
assert ax.bbox.y0 >= 48 and ax.bbox.y1 <= 582
destination.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(destination, dpi=100, facecolor=bg)
print("PASS: 1200x630 cover text and plot bounds inside the crop-safe area")
