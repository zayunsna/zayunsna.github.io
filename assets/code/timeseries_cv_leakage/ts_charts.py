"""Cover and result chart for the time-series leakage post. Numbers come from ts_leakage.py functions."""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold, TimeSeriesSplit

from ts_leakage import HOLDOUT, SCHEMES, cv_folds, holdout_mae, load

BG, FG, MUTED = "#121a2a", "#e8eef8", "#9aa8bf"
BLUE, ORANGE, GREEN, GREY = "#8cb8ff", "#f4c27a", "#8fd6a5", "#3a4660"
out = sys.argv[1] if len(sys.argv) > 1 else "."

X, y = load()
X_dev, y_dev, X_test, y_test = X.iloc[:-HOLDOUT], y.iloc[:-HOLDOUT], X.iloc[-HOLDOUT:], y.iloc[-HOLDOUT:]
lags = [c for c in X if c.startswith("lag_")]
cases = {
    "no lag features": (X_dev.drop(columns=lags), X_test.drop(columns=lags)),
    "with lag_1, lag_24, lag_168": (X_dev, X_test),
}
results = {}
for label, (xd, xt) in cases.items():
    results[label] = ({n: cv_folds(xd, y_dev, cv).mean() for n, cv in SCHEMES.items()},
                      holdout_mae(xd, y_dev, xt, y_test))
    print(label, {k: round(v, 1) for k, v in results[label][0].items()}, round(results[label][1], 1))

# ---- result chart: CV estimate per scheme vs the real holdout error
fig, axes = plt.subplots(1, 2, figsize=(12, 5.0), facecolor=BG, sharex=True)
colors = [ORANGE, GREY, BLUE, GREEN]
for ax, (label, (cv, hold)) in zip(axes, results.items()):
    ax.set_facecolor(BG)
    names = list(cv)
    ax.barh(range(len(names)), [cv[n] for n in names], color=colors, height=0.6)
    for i, n in enumerate(names):
        ax.text(cv[n] + 0.8, i, f"{cv[n]:.1f}", va="center", color=FG, fontsize=11,
                bbox=dict(facecolor=BG, edgecolor="none", pad=1))
    ax.axvline(hold, color="#ff7b7b", ls="--", lw=2)
    ax.set_yticks(range(len(names)), names, color=FG, fontsize=11)
    ax.invert_yaxis()
    ax.set_title(f"{label}\nholdout (next 8 weeks, dashed line): {hold:.1f}", color=FG, fontsize=12, loc="left")
    ax.tick_params(colors=MUTED)
    ax.set_xlabel("cross-validated MAE (bikes per hour)", color=MUTED)
    for s in ax.spines.values():
        s.set_visible(False)
axes[1].set_yticks(range(4), [""] * 4)
axes[0].set_xlim(0, 62)
fig.tight_layout()
fig.savefig(f"{out}/cv_vs_holdout.png", dpi=100, facecolor=BG)

# ---- cover: real demand series + where each split puts its test rows
fig = plt.figure(figsize=(12, 6.3), dpi=100, facecolor=BG)
n_dev, n_all = len(X_dev), len(X)
ax = fig.add_axes([0.0667, 0.47, 0.8666, 0.24], facecolor=BG)
daily = y.groupby(np.arange(n_all) // 24).mean()
ax.plot(np.arange(len(daily)) * 24, daily.values, color=MUTED, lw=1)
ax.axvspan(n_dev, n_all, color="#ff7b7b", alpha=0.18)
ax.text(n_dev - 150, daily.max() * 1.02, "the future (holdout)", color="#ff9b9b", fontsize=12, ha="right")
ax.set_xlim(0, n_all)
ax.axis("off")
ax.text(0, daily.max() * 1.02, "hourly bike rentals, 2011-2012 (mean of each 24 rows)", color=MUTED, fontsize=11)


def strip(ypos, cv, label, color):
    a = fig.add_axes([0.0667, ypos, 0.8666, 0.07], facecolor=BG)
    tr, te = list(cv.split(X_dev))[2]  # third fold of each scheme
    mask = np.zeros(n_all)
    mask[tr] = 1
    mask[te] = 2
    mask[n_dev:] = 3
    cmap = matplotlib.colors.ListedColormap([BG, GREY, color, "#5a2f3a"])
    a.imshow(mask[None, :], aspect="auto", cmap=cmap, vmin=0, vmax=3, interpolation="nearest")
    a.axis("off")
    fig.text(0.0667, ypos + 0.085, label, color=FG, fontsize=13)


strip(0.30, KFold(5, shuffle=True, random_state=0), "KFold(shuffle=True): test hours scattered between training hours", ORANGE)
strip(0.12, TimeSeriesSplit(5), "TimeSeriesSplit: train on the past, test on the next block", BLUE)
no_lag_cv, no_lag_hold = results["no lag features"]
fig.text(0.0667, 0.87, f"Shuffled CV said MAE {no_lag_cv['KFold(5, shuffle=True)']:.1f}.", color=ORANGE, fontsize=28, weight="bold")
fig.text(0.0667, 0.80, f"The next 8 weeks said {no_lag_hold:.1f}.", color="#ff9b9b", fontsize=28, weight="bold")
fig.savefig(f"{out}/cover.png", dpi=100, facecolor=BG)
