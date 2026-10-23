"""Chart and cover for the threshold post, drawn from threshold_curve.csv written by threshold.py."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

BG, FG, MUTED = "#1e1f26", "#e8e8ef", "#9a9ab0"
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "axes.edgecolor": MUTED, "axes.labelcolor": FG,
                     "xtick.color": FG, "ytick.color": FG, "text.color": FG, "font.size": 13, "axes.grid": True,
                     "grid.color": "#33343f", "legend.facecolor": BG, "legend.edgecolor": MUTED})
c = pd.read_csv("threshold_curve.csv")
at = lambda t: c.cost[(c.threshold - t).abs().idxmin()]


def fig_curve(w, h, title=None):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.plot(c.threshold, c.cost, color="#4fb3bf", lw=2.5, label="model, average over 20 test splits")
    ax.axhline(0.7, color="#e06c9f", ls="--", lw=2, label="reject every applicant: 0.700")
    for t, label, dy in [(0.5, "default 0.5", 0.06), (1 / 6, "1/(1+5) = 0.167", -0.09)]:
        ax.plot([t], [at(t)], "o", color="#f2a65a", ms=9, zorder=3)
        ax.annotate(f"{label}: {at(t):.3f}", (t, at(t)), xytext=(t + 0.03, at(t) + dy), color="#f2a65a")
    ax.set_xlabel("decision threshold on P(bad)")
    ax.set_ylabel("cost per applicant")
    ax.set_xlim(0, 1); ax.set_ylim(0.3, 1.6)
    ax.legend(loc="lower right", fontsize=12)
    if not title:
        ax.set_title("German Credit, cost 5 for approving a bad applicant, 1 for rejecting a good one",
                     color=MUTED, fontsize=12)
        fig.tight_layout()
        return fig
    # Cover: keep everything 80 px from the sides and 48 px from top/bottom (16:9 center crop on the site).
    fig.suptitle(title, fontsize=22, fontweight="bold", x=80 / 1200, ha="left", y=1 - 56 / 630, va="top")
    fig.subplots_adjust(left=150 / 1200, right=1105 / 1200, bottom=1 - 515 / 630, top=1 - 130 / 630)
    return fig


fig_curve(10, 5.2).savefig("threshold_cost.png", dpi=100)
fig_curve(12, 6.3, "Threshold 0.5 cost more than rejecting everyone").savefig("threshold_cover.png", dpi=100)
