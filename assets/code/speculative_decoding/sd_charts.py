"""Charts for the speculative decoding post, drawn from sd_results_mps.json written by `speculative.py mps`."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BG, FG, MUTED = "#1e1f26", "#e8e8ef", "#9a9ab0"
COLORS = {"1.5B alone": "#6b6f8a", "1.5B + 0.5B draft": "#4fb3bf", "1.5B + prompt lookup": "#f2a65a"}
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "axes.edgecolor": MUTED, "axes.labelcolor": FG,
                     "xtick.color": FG, "ytick.color": FG, "text.color": FG, "font.size": 13, "axes.grid": True,
                     "grid.color": "#33343f", "legend.facecolor": BG, "legend.edgecolor": MUTED})
res = json.load(open("sd_results_mps.json"))
PROMPTS = ["explain", "code", "rewrite", "story"]
METHODS = list(COLORS)


def check_safe(fig, path, x0=80, x1=1120, y0=48, y1=582):
    """Every text, axis and legend box must sit inside the cover's safe area (pixels, origin top-left)."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    W, H = fig.canvas.get_width_height()
    items = [t for t in fig.findobj(matplotlib.text.Text) if t.get_visible() and t.get_text()]
    items += list(fig.axes)
    bad = []
    for it in items:
        bb = it.get_window_extent(r)
        left, right, top, bottom = bb.x0, bb.x1, H - bb.y1, H - bb.y0
        if left < x0 or right > x1 or top < y0 or bottom > y1:
            bad.append((getattr(it, "get_text", lambda: type(it).__name__)(), round(left), round(right),
                        round(top), round(bottom)))
    print(f"{path}: {W}x{H}, outside safe area: {bad or 'none'}")


def speedup_chart():
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(PROMPTS))
    for off, method in [(-0.2, "1.5B + 0.5B draft"), (0.2, "1.5B + prompt lookup")]:
        sp = [res[p][method]["tok_per_s"] / res[p]["1.5B alone"]["tok_per_s"] for p in PROMPTS]
        tpc = [res[p][method]["tokens"] / res[p][method]["target_calls"] for p in PROMPTS]
        bars = ax.bar(x + off, sp, 0.38, color=COLORS[method], label=method.replace("1.5B + ", ""))
        ax.bar_label(bars, labels=[f"{s:.2f}x\n{t:.1f}/pass" for s, t in zip(sp, tpc)], fontsize=10,
                     label_type="center", color=BG, fontweight="bold")
    ax.axhline(1, color=FG, lw=1.2, ls="--")
    ax.text(3.62, 1.02, "1.5B alone", color=FG, fontsize=11, ha="right", va="bottom")
    ax.set_xticks(x, ["explain\n(hash table)", "code\n(is_palindrome)", "rewrite\n(copy facts)", "story\n(open-ended)"])
    ax.set_ylabel("speed vs 1.5B alone (tokens/s ratio)")
    ax.set_ylim(0, 1.75)
    ax.set_axisbelow(True)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", fontsize=11, title="drafting method", title_fontsize=11)
    ax.set_title("Apple M5 GPU (MPS), float32, greedy, 128 new tokens max; median of 5 runs. "
                 "In bars: speed ratio, tokens accepted per 1.5B pass", color=MUTED, fontsize=11)
    fig.tight_layout()
    fig.savefig("speedup_by_prompt.png", dpi=100)


def cover():
    p = res["code"]
    fig = plt.figure(figsize=(12, 6.3), dpi=100)
    fig.suptitle("Speculative decoding: 10x fewer passes, same speed", fontsize=24, fontweight="bold",
                 x=80 / 1200, ha="left", y=1 - 56 / 630, va="top")
    fig.text(80 / 1200, 1 - 106 / 630, "Qwen2.5-1.5B with a 0.5B draft, writing the same 128 code tokens on a laptop GPU",
             color=MUTED, fontsize=14, va="top")
    labels = ["1.5B alone", "+ 0.5B draft", "+ prompt lookup"]
    colors = [COLORS[m] for m in METHODS]
    panels = [("forward passes of the 1.5B model", [p[m]["target_calls"] for m in METHODS], "{:.0f}", 150),
              ("tokens per second", [p[m]["tok_per_s"] for m in METHODS], "{:.1f}", 20)]
    for (title, vals, fmt, top), left in zip(panels, [120, 640]):
        ax = fig.add_axes([left / 1200, 1 - 520 / 630, 440 / 1200, 320 / 630])
        bars = ax.bar(labels, vals, 0.6, color=colors)
        ax.bar_label(bars, labels=[fmt.format(v) for v in vals], fontsize=15, padding=4, fontweight="bold")
        ax.set_title(title, fontsize=16, loc="left")
        ax.set_ylim(0, top)
        ax.tick_params(axis="x", labelsize=12)
        ax.tick_params(axis="y", labelsize=11)
        ax.set_axisbelow(True)
        ax.grid(axis="x", visible=False)
    check_safe(fig, "cover.png")
    fig.savefig("cover.png", dpi=100)


speedup_chart()
cover()
