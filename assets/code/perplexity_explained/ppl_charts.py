"""Charts for the perplexity post, drawn from ppl_results.json written by perplexity.py."""
import json
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

BG, FG, MUTED = "#1e1f26", "#e8e8ef", "#9a9ab0"
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "axes.edgecolor": MUTED, "axes.labelcolor": FG,
                     "xtick.color": FG, "ytick.color": FG, "text.color": FG, "font.size": 13, "axes.grid": True,
                     "grid.color": "#33343f"})
r = json.load(open("ppl_results.json"))

# Body chart: perplexity of five texts, log scale.
names, vals = list(r["perplexity"]), list(r["perplexity"].values())
fig, ax = plt.subplots(figsize=(10, 4.6))
bars = ax.barh(names[::-1], vals[::-1], color="#4fb3bf")
ax.set_xscale("log"); ax.set_xlim(1, 5000)
ax.set_xlabel("perplexity (log scale, lower = less surprised)")
for b, v in zip(bars, vals[::-1]):
    ax.text(v * 1.1, b.get_y() + b.get_height() / 2, f"{v:,.1f}", va="center", color=FG)
ax.set_title("Qwen2.5-0.5B (base), the same model scoring five texts", color=MUTED, fontsize=12)
ax.grid(axis="y", visible=False); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig("ppl_texts.png", dpi=100); plt.close(fig)

# Cover: the natural sentence, each token shaded by the probability the model gave it.
cmap = LinearSegmentedColormap.from_list("surprise", ["#e06c9f", "#3a3b48", "#4fb3bf"])
fig = plt.figure(figsize=(12, 6.3)); W, Hh = 1200, 630
fig.text(80 / W, 1 - 56 / Hh, "Perplexity: how surprised a model is, token by token",
         fontsize=24, fontweight="bold", ha="left", va="top")
r_ = fig.canvas.get_renderer()
x, y, line_h = 92, 215, 78
first = fig.text(x / W, 1 - y / Hh, "The", fontsize=24, color=MUTED, ha="left", va="center",
                 bbox=dict(boxstyle="round,pad=0.25", fc=BG, ec=MUTED, lw=1))
x += first.get_window_extent(r_).width + 22
for piece, p in zip(r["tokens"], r["probs"]):
    t = fig.text(x / W, 1 - y / Hh, piece.strip(), fontsize=24, ha="left", va="center",
                 bbox=dict(boxstyle="round,pad=0.25", fc=cmap(min(1, max(0, (math.log10(max(p, 1e-4)) + 4) / 4))), ec="none"))
    w = t.get_window_extent(r_).width
    if x + w > 1108:                       # wrap inside the safe area (box padding included)
        x, y = 92, y + line_h
        t.set_position((x / W, 1 - y / Hh))
    x += w + 22
y += 110
fig.text(80 / W, 1 - y / Hh, "teal = expected (p near 1)    dark = in between    pink = surprising (p near 0.0001)",
         fontsize=16, color=MUTED, ha="left", va="center")
fig.text(80 / W, 1 - (y + 62) / Hh, f"perplexity = exp(mean of -log p) = {r['perplexity']['natural sentence']:.1f}",
         fontsize=22, ha="left", va="center")
fig.savefig("cover.png", dpi=100); plt.close(fig)
