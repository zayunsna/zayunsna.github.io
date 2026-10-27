"""Charts for the target encoding post, drawn from the CSV files written by target_encoding.py."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BG, FG, MUTED = "#1e1f26", "#e8e8ef", "#9a9ab0"
MEAN_C, TE_C = "#f2a65a", "#4fb3bf"
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "axes.edgecolor": MUTED, "axes.labelcolor": FG,
                     "xtick.color": FG, "ytick.color": FG, "text.color": FG, "font.size": 13, "axes.grid": True,
                     "grid.color": "#33343f", "legend.facecolor": BG, "legend.edgecolor": MUTED})


def check_safe(fig, path, x0=80, x1=1120, y0=48, y1=582):
    """Every text, axis and legend box must sit inside the cover's safe area (pixels, origin top-left)."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    W, H = fig.canvas.get_width_height()
    items = [t for t in fig.findobj(matplotlib.text.Text) if t.get_visible() and t.get_text()]
    items += [ax for ax in fig.axes] + [lg for ax in fig.axes if (lg := ax.get_legend())]
    bad = []
    for it in items:
        bb = it.get_window_extent(r)
        left, right, top, bottom = bb.x0, bb.x1, H - bb.y1, H - bb.y0
        if left < x0 or right > x1 or top < y0 or bottom > y1:
            bad.append((getattr(it, "get_text", lambda: type(it).__name__)(), round(left), round(right),
                        round(top), round(bottom)))
    print(f"{path}: {W}x{H}, outside safe area: {bad or 'none'}")


def rare_chart():
    df = pd.read_csv("te_rare.csv")
    groups = ["none", "~3 rows per ID", "~1.5 rows per ID"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, model in zip(axes, ["HGB", "Ridge"]):
        d = df[df.model == model].set_index("random_id").loc[groups]
        x = np.arange(len(groups))
        for off, col, color, label in [(-0.2, "mean test", MEAN_C, "plain mean encoding"),
                                       (0.2, "TE test", TE_C, "TargetEncoder (cross fitting)")]:
            bars = ax.bar(x + off, d[col], 0.38, color=color, label=label)
            ax.bar_label(bars, labels=[f"{v:,.0f}" for v in d[col]], fontsize=11, padding=3)
        ax.set_xticks(x, ["no random ID", "random ID,\n~3 rows each", "random ID,\n~1.5 rows each"])
        ax.set_title("HistGradientBoosting" if model == "HGB" else "Ridge regression", fontsize=13)
        ax.set_ylim(0, d[["mean test", "TE test"]].to_numpy().max() * 1.18)
        ax.grid(axis="x", visible=False)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("test MAE, USD (5-fold CV)")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=2, fontsize=12, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig("rare_categories.png", dpi=100)


def cover():
    df = pd.read_csv("te_random_id_codes.csv")
    fig = plt.figure(figsize=(12, 6.3), dpi=100)
    fig.suptitle("A random ID column, target-encoded two ways", fontsize=24, fontweight="bold",
                 x=80 / 1200, ha="left", y=1 - 56 / 630, va="top")
    fig.text(80 / 1200, 1 - 108 / 630, "Training rows: salary vs the encoded value of a column with no information",
             color=MUTED, fontsize=14, va="top")
    panels = [("mean encoding", MEAN_C, "plain mean: looks predictive"),
              ("TargetEncoder fit_transform", TE_C, "TargetEncoder: no signal")]
    for (col, color, title), left in zip(panels, [160, 660]):
        ax = fig.add_axes([left / 1200, 1 - 515 / 630, 430 / 1200, 320 / 630])
        ax.scatter(df[col] / 1000, df.salary / 1000, s=4, alpha=0.35, color=color, linewidths=0)
        r = np.corrcoef(df[col], df.salary)[0, 1]
        ax.set_title(f"{title}  (r = {r:.2f})", fontsize=15, color=color, loc="left")
        ax.set_xlabel("encoded random_id, $1,000s", fontsize=12)
        ax.set_xlim(0, 310); ax.set_ylim(0, 310)
        ax.set_xticks([0, 100, 200, 300]); ax.set_yticks([0, 100, 200, 300])
        ax.tick_params(labelsize=11)
    fig.axes[0].set_ylabel("salary, $1,000s", fontsize=12)
    check_safe(fig, "cover.png")
    fig.savefig("cover.png", dpi=100)


rare_chart()
cover()
