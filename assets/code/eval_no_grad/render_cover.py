"""Render observed Dropout masks beside a labeled autograd schematic."""
import json
from pathlib import Path
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch
from matplotlib.text import Text

data = json.loads(Path(sys.argv[1]).read_text())
train, evaluation = data["matrix"][0], data["matrix"][2]
assert train["output_requires_grad"] and evaluation["output_requires_grad"]
assert not data["matrix"][3]["output_requires_grad"]
bg, fg, muted, teal, purple = "#19182c", "#f1effc", "#b7b3ce", "#68dcc3", "#c4a6ff"
fig = plt.figure(figsize=(12, 6.3), dpi=100, facecolor=bg)
fig.text(.08, .865, "Two independent switches", fontsize=34, weight="bold", color=fg)
for left in [.08, .515]:
    fig.add_artist(FancyBboxPatch((left, .215), .405, .535, boxstyle="round,pad=0.005",
                                  transform=fig.transFigure, facecolor="#25243e", edgecolor="none"))
fig.text(.105, .665, "model.eval()", fontsize=27, color=teal, family="monospace", weight="bold")
fig.text(.105, .597, "Layer behavior", fontsize=20, color=fg)
for y, row in [(.455, train), (.325, evaluation)]:
    fig.text(.105, y + .055, row["mode"] + " Dropout", fontsize=16, color=muted)
    values = sum(row["first_output"], [])
    for i, value in enumerate(values):
        fig.add_artist(Circle((.125 + i * .044, y), .012, transform=fig.transFigure,
                              facecolor=teal if value else "#52516a", edgecolor="none"))
fig.text(.54, .665, "torch.no_grad()", fontsize=25, color=purple, family="monospace", weight="bold")
fig.text(.54, .597, "Gradient recording", fontsize=20, color=fg)
for y, label, active in [(.455, "normal: graph recorded", True), (.325, "no_grad: no graph", False)]:
    fig.text(.54, y + .055, label, fontsize=16, color=muted)
    xs = [.575, .70, .825]
    for x in xs:
        fig.add_artist(Circle((x, y), .014, transform=fig.transFigure,
                              facecolor=purple if active else "#52516a", edgecolor="none"))
    if active:
        for first, second in zip(xs[:-1], xs[1:]):
            fig.add_artist(plt.Line2D([first+.018, second-.018], [y, y],
                                      transform=fig.transFigure, color=purple, lw=2))
fig.text(.08, .12, "Observed Dropout mask", fontsize=15, color=muted)
fig.text(.54, .12, "Autograd schematic", fontsize=15, color=muted)
fig.canvas.draw()
for item in fig.findobj(match=Text):
    if item.get_visible() and item.get_text().strip():
        b = item.get_window_extent(fig.canvas.get_renderer())
        assert b.x0 >= 80 and b.x1 <= 1120 and b.y0 >= 48 and b.y1 <= 582, (item.get_text(), b)
destination = Path(sys.argv[2])
destination.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(destination, dpi=100, facecolor=bg)
print("PASS: 1200x630, all text inside crop-safe bounds; visual QA still required")
