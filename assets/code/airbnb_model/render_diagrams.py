"""Draw original conceptual diagrams; no paper images or plotted data are used."""
from pathlib import Path
import argparse
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

BG = '#131b2a'
FG = '#f1f5f9'
ACCENT = '#72ddcf'
CHECKS = []


def canvas(title):
    fig, ax = plt.subplots(figsize=(12, 5.4), dpi=120)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_position([0, 0, 1, 1])
    ax.set(xlim=(0, 12), ylim=(0, 5.4))
    ax.axis('off')
    ax.text(.6, 4.8, title, color=FG, fontsize=22, weight='bold')
    ax.text(.6, .35, 'Original conceptual diagram | Haldar et al., arXiv:1810.09591',
            color='#b7c5d9', fontsize=12)
    return fig, ax, []


def box(ax, boxes, x, y, w, h, text):
    patch = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.03,rounding_size=0.12',
                          edgecolor=ACCENT, facecolor='#22344b', linewidth=1.6)
    ax.add_patch(patch)
    label = ax.text(x+w/2, y+h/2, text, ha='center', va='center', color=FG,
                    fontsize=16, linespacing=1.5)
    boxes.append((patch, label))


def arrow(ax, start, end):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=18,
                                linewidth=1.8, color=ACCENT))


def save(fig, ax, boxes, path):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bounds = fig.bbox
    for label in ax.texts:
        b = label.get_window_extent(renderer)
        assert bounds.contains(b.x0, b.y0) and bounds.contains(b.x1, b.y1), label.get_text()
    for patch in ax.patches:
        b = patch.get_window_extent(renderer)
        assert bounds.contains(b.x0, b.y0) and bounds.contains(b.x1, b.y1), 'Shape outside canvas'
    for patch, label in boxes:
        p = patch.get_window_extent(renderer)
        b = label.get_window_extent(renderer)
        assert p.contains(b.x0-4, b.y0-4) and p.contains(b.x1+4, b.y1+4), label.get_text()
    for i, left in enumerate(ax.texts):
        for right in ax.texts[i+1:]:
            assert not left.get_window_extent(renderer).overlaps(right.get_window_extent(renderer))
    fig.savefig(path, facecolor=BG)
    CHECKS.append(dict(file=path.name, text_in_canvas=True, shapes_in_canvas=True, labels_in_boxes=True,
                       text_overlap=False, width=int(bounds.width), height=int(bounds.height)))
    plt.close(fig)


def main(out):
    out.mkdir(parents=True, exist_ok=True)
    fig, ax, boxes = canvas('Ranking model changes')
    labels = ['Simple NN', 'LambdaRank\nNN', 'GBDT + FM\nfeatures in NN', 'Deep NN\nmore data']
    for i, text in enumerate(labels):
        x = .6 + i*2.85
        box(ax, boxes, x, 2.15, 2.25, 1.3, text)
        if i < 3:
            arrow(ax, (x+2.3, 2.8), (x+2.79, 2.8))
    ax.text(.6, 1.15, 'Sequence of approaches; box sizes carry no performance values.', color=FG, fontsize=15)
    save(fig, ax, boxes, out/'model_changes_concept.png')

    fig, ax, boxes = canvas('Inputs to the hybrid ranker')
    for y, text in [(3.25, 'GBDT leaf IDs'), (2.05, 'FM prediction'), (.85, 'Other features')]:
        box(ax, boxes, .6, y, 3.1, .85, text)
        arrow(ax, (3.8, y+.425), (5.25, 2.6))
    box(ax, boxes, 5.4, 1.95, 2.5, 1.3, 'Neural\nnetwork')
    box(ax, boxes, 9.1, 2.05, 2.3, 1.1, 'Ranking\nscore')
    arrow(ax, (8.0, 2.6), (8.98, 2.6))
    save(fig, ax, boxes, out/'hybrid_inputs_concept.png')

    fig, ax, boxes = canvas('Listing ID and overfitting')
    box(ax, boxes, .6, 2.05, 2.4, 1.3, 'Sparse\nbookings')
    box(ax, boxes, 4.0, 2.05, 2.9, 1.3, 'Trainable\nlisting-ID vectors')
    arrow(ax, (3.1, 2.7), (3.87, 2.7))
    box(ax, boxes, 8.0, 3.05, 3.4, 1.1, 'Training fit improves')
    box(ax, boxes, 8.0, 1.15, 3.4, 1.1, 'Test gain absent')
    arrow(ax, (7.0, 2.7), (7.87, 3.6))
    arrow(ax, (7.0, 2.7), (7.87, 1.7))
    save(fig, ax, boxes, out/'listing_id_concept.png')

    fig, ax, boxes = canvas('TopBot feature inspection')
    box(ax, boxes, .6, 2.05, 2.5, 1.3, 'Rank listings\nper test query')
    box(ax, boxes, 4.1, 3.05, 2.4, 1.1, 'Top group')
    box(ax, boxes, 4.1, 1.15, 2.4, 1.1, 'Bottom group')
    arrow(ax, (3.2, 2.7), (3.97, 3.6))
    arrow(ax, (3.2, 2.7), (3.97, 1.7))
    box(ax, boxes, 8.0, 2.05, 3.4, 1.3, 'Compare price\nand review count')
    arrow(ax, (6.6, 3.6), (7.87, 2.7))
    arrow(ax, (6.6, 1.7), (7.87, 2.7))
    save(fig, ax, boxes, out/'topbot_workflow_concept.png')
    print(json.dumps(CHECKS, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path,
                        default=Path(__file__).resolve().parents[2]/'img/ds/airbnb')
    main(parser.parse_args().output_dir)
