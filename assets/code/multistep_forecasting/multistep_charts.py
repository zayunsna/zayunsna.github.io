"""Charts for the multi-step forecasting post, drawn from mae_by_h.npy written by multistep.py."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BG, FG, MUTED = "#1e1f26", "#e8e8ef", "#9a9ab0"
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "axes.edgecolor": MUTED, "axes.labelcolor": FG,
                     "xtick.color": FG, "ytick.color": FG, "text.color": FG, "font.size": 13, "axes.grid": True,
                     "grid.color": "#33343f", "legend.facecolor": BG, "legend.edgecolor": MUTED})

mae = np.load("mae_by_h.npy")
names = open("mae_by_h_names.txt").read().split("\n")
STYLE = {  # name -> (label, color, line style)
    "one-step, real lag_1 (not a forecast)": ("one-step score (real lag_1 every hour)", "#9a9ab0", "--"),
    "recursive": ("recursive: feed forecasts back as lag_1", "#4fb3bf", "-"),
    "direct": ("direct: one model per horizon", "#f2a65a", "-"),
    "lag_24 + lag_168 only": ("no short lag: lag_24 + lag_168", "#e06c9f", "-"),
    "no lags": ("no lags: calendar + weather", "#6b6f8a", ":"),
}


def fig_mae(w, h, title=None):
    fig, ax = plt.subplots(figsize=(w, h))
    hs = np.arange(1, mae.shape[1] + 1)
    for name, row in zip(names, mae):
        label, color, ls = STYLE[name]
        ax.plot(hs, row, ls, color=color, lw=2.5, marker="o" if ls == "-" else None, ms=4, label=label)
    ax.set_xlabel("hours ahead (forecast horizon)")
    ax.set_ylabel("MAE, bikes per hour")
    ax.set_xticks([1, 3, 6, 9, 12, 15, 18, 21, 24])
    ax.set_ylim(0, 60)
    ax.legend(loc="lower right", fontsize=12)
    if not title:
        ax.set_title("Holdout: last 8 weeks, every hour as a forecast origin", color=MUTED, fontsize=12)
        fig.tight_layout()
        return fig
    # Cover: keep everything 80 px from the sides and 48 px from top/bottom (16:9 center crop on the site).
    fig.suptitle(title, fontsize=22, fontweight="bold", x=80 / 1200, ha="left", y=1 - 56 / 630, va="top")
    fig.subplots_adjust(left=150 / 1200, right=1105 / 1200, bottom=1 - 515 / 630, top=1 - 130 / 630)
    return fig


fig_mae(11, 5.5).savefig("mae_by_horizon.png", dpi=100)
fig_mae(12, 6.3, "Forecast error: 24.5 one step ahead, 45.7 over a day").savefig("cover.png", dpi=100)
