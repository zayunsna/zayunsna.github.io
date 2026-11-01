"""Cover for the cosine threshold post, drawn from results/cosine_threshold_results.json (cosine_threshold.py).

Run from the folder that contains results/: python ct_cover.py
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

BG, FG, MUTED, GRID, MINI, BGE = "#1b1d2a", "#eceaf4", "#a3a1b8", "#30324a", "#4fb3bf", "#e9a3c9"
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "axes.edgecolor": MUTED, "text.color": FG,
                     "xtick.color": FG, "ytick.color": FG, "axes.labelcolor": FG, "font.size": 15})
res = json.load(open("results/cosine_threshold_results.json"))
W, H = 1200, 630
fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
fig.text(84 / W, 1 - 54 / H, "Cosine score ≠ confidence", fontsize=30, fontweight="bold", va="top")
fig.text(84 / W, 1 - 106 / H, "Same 80% validation target, model-specific cutoffs", fontsize=19, color=MUTED, va="top")
ax = fig.add_axes([200 / W, 1 - 520 / H, 660 / W, 360 / H])
for key, label, color, t in [("mini_val", "all-MiniLM-L6-v2", MINI, res["t_mini"]),
                             ("bge_val", "bge-small-en-v1.5", BGE, res["t_bge"])]:
    c = res["curves"][key]
    keep = [i for i, cov in enumerate(c["coverage"]) if cov >= 0.02]  # skip cutoffs that answer < 20 questions
    ax.plot([c["cutoff"][i] for i in keep], [c["precision"][i] for i in keep], lw=4, color=color)
    ax.plot([t], [c["precision"][c["cutoff"].index(t)]], "o", ms=13, color=color, mec=BG, mew=2, zorder=5)
    fig.text(892 / W, 1 - (230 if color == MINI else 330) / H, label, fontsize=16, color=color, va="center")
    fig.text(892 / W, 1 - (262 if color == MINI else 362) / H, f"cutoff {t:.2f}", fontsize=24,
             fontweight="bold", color=color, va="center")
ax.axhline(0.8, color=MUTED, ls="--", lw=1.2)
ax.set_xlim(0.2, 0.85)
ax.set_ylim(0.3, 1.0)
ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
ax.set_yticks([0.4, 0.6, 0.8, 1.0])
ax.set_xticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
ax.set_xlabel("pass bundle if best cosine score ≥ cutoff")
ax.set_ylabel("accepted-query success")
ax.grid(color=GRID)
ax.set_axisbelow(True)
ax.spines[["top", "right"]].set_visible(False)
fig.text(892 / W, 1 - 420 / H, "1,000 SQuAD\nvalidation queries,\nhalf proxy-negative", fontsize=14,
         color=MUTED, va="top", linespacing=1.4)


def check_safe(x0=80, x1=1120, y0=48, y1=582):
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    items = [t for t in fig.findobj(matplotlib.text.Text) if t.get_visible() and t.get_text()] + [ax]
    bad = []
    for it in items:
        bb = it.get_window_extent(r)
        if bb.x0 < x0 or bb.x1 > x1 or H - bb.y1 < y0 or H - bb.y0 > y1:
            bad.append((getattr(it, "get_text", lambda: "axes")(), round(bb.x0), round(bb.x1),
                        round(H - bb.y1), round(H - bb.y0)))
    texts = [(t.get_text(), t.get_window_extent(r)) for t in items[:-1]]
    overlaps = [(a, b) for i, (a, ba) in enumerate(texts) for b, bb in texts[i + 1:] if ba.overlaps(bb)]
    print(f"cover.png: {W}x{H}, outside safe area: {bad or 'none'}, overlapping texts: {overlaps or 'none'}")


check_safe()
fig.savefig("cover.png", dpi=100)
