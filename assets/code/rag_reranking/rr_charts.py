"""Charts for the reranking post, drawn from rerank_results_mps.json written by `rerank.py ... mps`."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BG, FG, MUTED, BI_C, RR_C, CEIL_C = "#1e1f26", "#e8e8ef", "#9a9ab0", "#6b6f8a", "#4fb3bf", "#f2a65a"
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "axes.edgecolor": MUTED, "axes.labelcolor": FG,
                     "xtick.color": FG, "ytick.color": FG, "text.color": FG, "font.size": 13, "axes.grid": True,
                     "grid.color": "#33343f", "legend.facecolor": BG, "legend.edgecolor": MUTED})
res = json.load(open("rerank_results_mps.json"))["results"]
DEPTHS = [int(k.split()[-1]) for k in res if k.startswith("+ rerank")]
bi = res["bi-encoder only"]
rr = [res[f"+ rerank top {k}"] for k in DEPTHS]
COVER_K = 20


def check_safe(fig, path, x0=80, x1=1120, y0=48, y1=582):
    """Every text, axis and legend box must sit inside the given area (pixels, origin top-left)."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    W, H = fig.canvas.get_width_height()
    items = [t for t in fig.findobj(matplotlib.text.Text) if t.get_visible() and t.get_text()]
    items += list(fig.axes) + [lg for ax in fig.axes if (lg := ax.get_legend())]
    bad = []
    for it in items:
        bb = it.get_window_extent(r)
        left, right, top, bottom = bb.x0, bb.x1, H - bb.y1, H - bb.y0
        if left < x0 or right > x1 or top < y0 or bottom > y1:
            bad.append((getattr(it, "get_text", lambda: type(it).__name__)(), round(left), round(right),
                        round(top), round(bottom)))
    print(f"{path}: {W}x{H}, outside safe area: {bad or 'none'}")


def draw(ax, label_size):
    xs = range(len(DEPTHS))
    ax.plot(xs, [r["ceiling"] for r in rr], "--o", color=CEIL_C, lw=2,
            label="correct paragraph is among the candidates")
    ax.plot(xs, [r["hit@1"] for r in rr], "-o", color=RR_C, lw=3, label="ranked first after reranking")
    ax.axhline(bi["hit@1"], color=BI_C, lw=2.5, label="ranked first, bi-encoder only")
    for x, r in zip(xs, rr):
        ax.annotate(f"{r['hit@1']:.0%}", (x, r["hit@1"]), textcoords="offset points", xytext=(0, -24),
                    ha="center", fontsize=label_size, color=RR_C)
    ax.text(0, bi["hit@1"] + 0.012, f"{bi['hit@1']:.0%}", ha="center", va="bottom",
            fontsize=label_size, color=FG)
    ax.set_xticks(list(xs), [f"top {k}\n{r['ms_per_query']:.0f} ms" for k, r in zip(DEPTHS, rr)])
    ax.set_ylim(0.55, 1.02)
    ax.set_yticks([0.6, 0.7, 0.8, 0.9, 1.0])
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_axisbelow(True)
    ax.grid(axis="x", visible=False)


def body_chart():
    fig, ax = plt.subplots(figsize=(11, 5.5))
    draw(ax, 11)
    ax.set_xlabel("candidates passed to the cross-encoder (time per query on the M5 GPU)")
    ax.set_ylabel("share of 1,000 SQuAD questions")
    ax.legend(loc="lower right", bbox_to_anchor=(1, 0.24), fontsize=11)
    fig.tight_layout()
    fig.savefig("rerank_depth.png", dpi=100)
    check_safe(fig, "rerank_depth.png", 0, 1100, 0, 550)


def cover():
    best = res[f"+ rerank top {COVER_K}"]["hit@1"]
    fig = plt.figure(figsize=(12, 6.3), dpi=100)
    fig.suptitle(f"Reranking: {bi['hit@1']:.0%} → {best:.0%} of answers ranked first", fontsize=24,
                 fontweight="bold", x=80 / 1200, ha="left", y=1 - 56 / 630, va="top")
    fig.text(80 / 1200, 1 - 104 / 630, f"Cross-encoder re-scoring the top {COVER_K} bi-encoder results; 1,000 SQuAD questions",
             color=MUTED, fontsize=14, va="top")
    ax = fig.add_axes([150 / 1200, 1 - 510 / 630, 950 / 1200, 350 / 630])
    draw(ax, 13)
    ax.tick_params(labelsize=12)
    ax.legend(loc="lower right", bbox_to_anchor=(1, 0.24), fontsize=12)
    check_safe(fig, "cover.png")
    fig.savefig("cover.png", dpi=100)


body_chart()
cover()
