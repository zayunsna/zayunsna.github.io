"""Render an original hybrid-ranking concept cover with checked crop margins."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.transforms import Bbox
from PIL import Image


def render(output):
    bg, ink, teal = '#131b2a', '#f1f5f9', '#72ddcf'
    fig = plt.figure(figsize=(12, 6.3), dpi=100, facecolor=bg)
    ax = fig.add_axes([0, 0, 1, 1], xlim=(0, 1200), ylim=(0, 630))
    ax.axis('off')
    labels, boxes = [], []

    def text(x, y, value, size, **kwargs):
        label = ax.text(x, y, value, fontsize=size, color=ink, **kwargs)
        labels.append(label)
        return label

    def node(x, y, w, h, value, size, color=teal):
        patch = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=16',
                              facecolor='#22344b', edgecolor=color, linewidth=2.5)
        ax.add_patch(patch)
        label = text(x+w/2, y+h/2, value, size, ha='center', va='center', linespacing=1.4)
        boxes.append((patch, label))

    def arrow(start, end):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=23,
                                    linewidth=2.8, color=teal))

    text(104, 520, 'Search ranking', 34, weight='bold')
    text(104, 83, 'CONCEPT DIAGRAM', 22)
    for y, value in [(350, 'GBDT leaves'), (250, 'FM output'), (150, 'Features')]:
        node(104, y, 220, 70, value, 23)
    node(475, 220, 260, 170, 'Neural\nranker', 29)
    node(885, 225, 210, 160, 'Ranked\nlist', 26, color='#b8a4ee')
    for start, end in [((337, 385), (461, 325)), ((337, 285), (461, 305)),
                       ((337, 185), (461, 285)), ((751, 305), (871, 305))]:
        arrow(start, end)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    safe = Bbox.from_extents(80, 48, 1120, 582)
    elements = []
    for artist in [*labels, *ax.patches]:
        bounds = artist.get_window_extent(renderer).padded(3)
        assert safe.contains(bounds.x0, bounds.y0) and safe.contains(bounds.x1, bounds.y1), bounds
        elements.append(dict(type=type(artist).__name__, bounds=list(bounds.extents)))
    for patch, label in boxes:
        p = patch.get_window_extent(renderer)
        b = label.get_window_extent(renderer).padded(6)
        assert p.contains(b.x0, b.y0) and p.contains(b.x1, b.y1)
    for i, left in enumerate(labels):
        for right in labels[i+1:]:
            assert not left.get_window_extent(renderer).overlaps(right.get_window_extent(renderer))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=100, facecolor=bg)
    plt.close(fig)
    with Image.open(output) as png:
        assert png.size == (1200, 630), png.size
    print(json.dumps(dict(output=output.name, size=[1200, 630], safe_region=[80, 48, 1120, 582],
                          safe_bounds_pass=True, labels_fit=True, text_overlap=False, elements=elements), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path,
                        default=Path(__file__).resolve().parents[2]/'img/ds/airbnb/cover.png')
    render(parser.parse_args().output)
