"""Render four topic-specific covers from the accompanying experiment outputs.

Usage: python render_figures.py pandas|metrics|padding|lstm DATA OUTPUT.png
DATA is the experiment JSON, or the output directory for lstm.
Packages: matplotlib==3.11.0, pandas==3.1.0, pillow==13.0.0.
"""
import json
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.text import Text
import pandas as pd

kind, data_path, output = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
bg, fg, muted = '#152133', '#eef3fc', '#b5c1d3'
teal, amber, violet = '#67d8bf', '#ffc078', '#c4adff'
fig = plt.figure(figsize=(12, 6.3), dpi=100, facecolor=bg)
axes = []


def text(x, y, value, size=20, color=fg, **kwargs):
    return fig.text(x, y, value, fontsize=size, color=color, **kwargs)


if kind == 'pandas':
    data = json.loads(data_path.read_text())
    text(.08, .86, 'select() chooses the output', 33, weight='bold')
    text(.08, .79, 'Same calculations. Different columns survive.', 19, muted)
    for x, name, fields in [(.08, 'assign()', data['assign']), (.53, 'select()', data['select'])]:
        text(x+.02, .65, name, 27, color=amber if name == 'assign()' else teal, family='monospace')
        for j, label in enumerate(fields):
            yy = .545 - j*.064
            fig.add_artist(FancyBboxPatch((x+.015, yy-.018), .37, .052,
                boxstyle='round,pad=0.003', transform=fig.transFigure,
                facecolor='#2a3850', edgecolor='none'))
            text(x+.03, yy-.002, label, 18, color=teal if label in data['select'] else muted, family='monospace')
    text(.55, .31, 'Explicit output schema', 20, teal)
    text(.08, .11, 'pandas 3.1 / measured on a three-row order table', 17, muted)
elif kind == 'metrics':
    data = json.loads(data_path.read_text())
    text(.08, .86, 'Which result is better?', 34, weight='bold')
    text(.08, .79, 'One perfect answer first, or two relevant answers later?', 18, muted)
    for j, row in enumerate(data['rows'][:2]):
        y = .62 - j*.23
        text(.08, y+.03, 'Query ' + str(j+1), 20, color=teal if j == 0 else amber)
        for i, doc in enumerate(row['ranking'][:3]):
            x = .25+i*.095
            grade = data['grades'][doc]
            fig.add_artist(Rectangle((x, y-.025), .073, .095, transform=fig.transFigure,
                                    facecolor=teal if grade else '#3a4961', edgecolor='none'))
            text(x+.0365, y+.005, doc, 24, color=bg if grade else fg, ha='center', weight='bold')
        text(.59, y+.037, f"Recall@3  {row['recall']:.3f}", 20, color=teal)
        text(.59, y-.022, f"RR@3       {row['rr']:.3f}", 20, color=amber)
    text(.08, .145, 'Known relevant documents: A, B, C / constructed examples', 17, muted)
elif kind == 'padding':
    data = json.loads(data_path.read_text())
    text(.08, .86, 'The last position matters', 34, weight='bold')
    text(.08, .79, 'Actual final 10 input positions of the shorter prompt', 18, muted)
    for y, name in [(.59, 'left+mask'), (.39, 'right+mask')]:
        row = data[name]
        text(.08, y+.055, name, 21, family='monospace')
        mask = row['mask'][0][-10:]
        for j, bit in enumerate(mask):
            x = .34 + j*.055
            fig.add_artist(Rectangle((x, y), .043, .058, transform=fig.transFigure,
                 facecolor=teal if bit else amber, edgecolor=fg if j == 9 else 'none', lw=2))
        text(.34, y-.045, 'last token ID: ' + str(row['last_ids'][0]), 16, muted)
    text(.08, .18, 'Teal: real token / amber: padding / outline: generation position', 16, muted)
    text(.08, .105, 'An attention mask does not move the final position.', 21, color=amber)
elif kind == 'lstm':
    data = json.loads((data_path / 'summary.json').read_text())
    text(.08, .86, '7-day LSTM forecast', 34, weight='bold')
    text(.08, .79, 'One input window. Seven outputs. No updates inside the forecast.', 17, muted)
    ax = fig.add_axes([.16, .24, .73, .44], facecolor=bg)
    axes.append(ax)
    for name, color in [('Last value', amber), ('Last week', violet), ('LSTM', teal)]:
        ax.plot(range(1,8), data['scores'][name]['horizon'], marker='o', lw=2, color=color, label=name)
    ax.set_xlabel('Days ahead', color=muted, fontsize=14)
    ax.set_ylabel('Test MAE / rentals', color=muted, fontsize=14)
    ax.set_xticks(range(1,8))
    ax.set_ylim(850, 1500)
    ax.tick_params(colors=muted, labelsize=12)
    ax.grid(alpha=.15)
    ax.legend(loc='lower right', ncol=3, facecolor=bg, labelcolor=fg, fontsize=12)
    for spine in ax.spines.values():
        spine.set_color('#53617a')
    text(.08, .105, '104 rolling origins / default seed 0 / CPU experiment', 18, muted)
else:
    raise ValueError(kind)

fig.canvas.draw()
renderer = fig.canvas.get_renderer()
for item in fig.findobj(match=Text):
    if not item.get_visible() or not item.get_text().strip():
        continue
    bounds = item.get_window_extent(renderer)
    if bounds.width == 0 or bounds.height == 0:
        continue
    assert bounds.x0 >= 80 and bounds.x1 <= 1120 and bounds.y0 >= 48 and bounds.y1 <= 582, (item.get_text(), bounds)
for ax in axes:
    assert ax.bbox.x0 >= 80 and ax.bbox.x1 <= 1120 and ax.bbox.y0 >= 48 and ax.bbox.y1 <= 582
output.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(output, dpi=100, facecolor=bg)
print('PASS: 1200x630 text/axis crop-safe bounds; visual inspection still required:', kind)
