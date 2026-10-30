"""Render the confirmed findings; the image is not the sole source of results."""
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.text import Text

root = Path(__file__).resolve().parent
functions = ["page", "discounted_price", "cache_key", "p95", "word_count", "mean_or_none"]
tests = ["test_page", "test_discount", "test_cache_key", "test_p95", "test_words", "test_mean"]
baseline = (root / "baseline_out.txt").read_text()
assert "3 failed, 5 passed" in baseline
reports = [(root / f"review_{label}.txt").read_text() for label in ["a", "b"]]
assert all(all(name in report for name in functions) for report in reports)
assert "PASS: 10 counterexamples reproduced; 2 control probes passed" in (root / "verified_findings.txt").read_text()
bg, fg, muted, teal, amber = "#10252b", "#edf4ee", "#b0c8c5", "#69dbc4", "#ffb76b"
fig = plt.figure(figsize=(12, 6.3), dpi=100, facecolor=bg)
fig.text(.08, .865, "AI review vs tests", fontsize=35, weight="bold", color=fg)
fig.text(.08, .805, "Six seeded bugs. Which checks noticed them?", fontsize=18, color=muted)
for x, label in zip([.53, .715, .875], ["Baseline", "Review A", "Review B"]):
    fig.text(x, .705, label, ha="center", color=fg, fontsize=18)
for i, (function, test) in enumerate(zip(functions, tests)):
    y = .625 - i * .078
    fig.text(.09, y, function, color=fg, fontsize=18, va="center", family="monospace")
    found = f"::{test} " in baseline
    for x, hit in zip([.53, .715, .875], [found, True, True]):
        fig.add_artist(Circle((x, y), .014, transform=fig.transFigure,
                              facecolor=teal if hit else bg,
                              edgecolor=teal if hit else muted, lw=2))
fig.text(.08, .105, "Second review: 0 new bugs. Preserve findings as tests.", fontsize=20, color=amber)
fig.canvas.draw()
for item in fig.findobj(match=Text):
    if item.get_visible() and item.get_text().strip():
        b = item.get_window_extent(fig.canvas.get_renderer())
        assert b.x0 >= 80 and b.x1 <= 1120 and b.y0 >= 48 and b.y1 <= 582, (item.get_text(), b)
destination = Path(sys.argv[1])
destination.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(destination, dpi=100, facecolor=bg)
print("PASS: 1200x630, all text inside crop-safe bounds; visual QA still required")
