"""Charts for the hybrid search post, drawn from hybrid_results_mps.json written by `hybrid.py ... mps`."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BG, FG, MUTED, GRID = "#1e1f26", "#e8e8ef", "#9a9ab0", "#33343f"
COLORS = {"BM25": "#f2a65a", "embeddings": "#8f93b8", "hybrid (RRF)": "#4fb3bf"}
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "axes.edgecolor": MUTED, "axes.labelcolor": FG,
                     "xtick.color": FG, "ytick.color": FG, "text.color": FG, "font.size": 13,
                     "legend.facecolor": BG, "legend.edgecolor": MUTED})
res = json.load(open("hybrid_results_mps.json"))
N = 1_000
METHODS = list(COLORS)


def check_safe(fig, path, x0=80, x1=1120, y0=48, y1=582):
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    W, H = fig.canvas.get_width_height()
    items = [t for t in fig.findobj(matplotlib.text.Text) if t.get_visible() and t.get_text()]
    items += list(fig.axes) + [lg for ax in fig.axes if (lg := ax.get_legend())]
    bad = []
    for it in items:
        bb = it.get_window_extent(r)
        if bb.x0 < x0 or bb.x1 > x1 or H - bb.y1 < y0 or H - bb.y0 > y1:
            bad.append((getattr(it, "get_text", lambda: type(it).__name__)(), round(bb.x0), round(bb.x1),
                        round(H - bb.y1), round(H - bb.y0)))
    print(f"{path}: {W}x{H}, outside safe area: {bad or 'none'}")


def body_chart():
    ks = [1, 5, 20, 100]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for m in METHODS:
        ys = [res[m][f"hit@{k}"] for k in ks]
        ax.plot(range(len(ks)), ys, "-o", lw=3, color=COLORS[m], label=m)
    ax.set_xticks(range(len(ks)), [f"top {k}" for k in ks])
    ax.set_ylim(0.58, 1.04)
    ax.set_yticks([0.6, 0.7, 0.8, 0.9, 1.0])
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("how many results you keep")
    ax.set_ylabel("questions whose paragraph is in the results")
    ax.grid(axis="y", color=GRID)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig("hybrid_hits.png", dpi=100)
    check_safe(fig, "hybrid_hits.png", 0, 1100, 0, 550)


def cover():
    missed = {m: round((1 - res[m]["hit@20"]) * N) for m in METHODS}
    fig = plt.figure(figsize=(12, 6.3), dpi=100)
    fig.text(80 / 1200, 1 - 56 / 630, "Hybrid search: fewer questions where search misses",
             fontsize=24, fontweight="bold", va="top")
    fig.text(80 / 1200, 1 - 102 / 630, "Of 1,000 SQuAD questions, how many had their paragraph missing "
             "from the top 20 results", fontsize=14, color=MUTED, va="top")
    ax = fig.add_axes([360 / 1200, 1 - 500 / 630, 700 / 1200, 320 / 630])
    order = METHODS[::-1]
    bars = ax.barh(range(len(order)), [missed[m] for m in order], color=[COLORS[m] for m in order], height=0.6)
    for b, m in zip(bars, order):
        ax.text(b.get_width() + 1, b.get_y() + b.get_height() / 2, f"{missed[m]}", va="center",
                fontsize=22, fontweight="bold", color=COLORS[m])
    ax.set_yticks(range(len(order)), ["BM25 + embeddings\n(RRF)" if m == "hybrid (RRF)" else m for m in order],
                  fontsize=17)
    ax.set_xlim(0, 55)
    ax.set_xticks([])
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.text(360 / 1200, 1 - 540 / 630, f"After a cross-encoder reranks the top 20: "
             f"{res['embeddings + rerank']['hit@1']:.1%} → {res['hybrid (RRF) + rerank']['hit@1']:.1%} "
             f"ranked first", fontsize=15, color=FG, va="top")
    check_safe(fig, "hy_cover.png")
    fig.savefig("hy_cover.png", dpi=100)


body_chart()
cover()
