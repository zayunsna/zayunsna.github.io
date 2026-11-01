"""Plot separate measured experiments, never a fabricated total memory figure."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.text import Text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(Path(__file__).with_name('memory.json').read_text())
    plt.rcParams.update({'text.color':'#edf4f8', 'axes.labelcolor':'#edf4f8',
                         'xtick.color':'#bdc8d4', 'ytick.color':'#bdc8d4', 'font.size':18})
    fig = plt.figure(figsize=(12, 6.3), dpi=100, facecolor='#101e2c')
    fig.text(.08, .85, 'Small weights. Growing cache.', fontsize=34, weight='bold')
    fig.text(.08, .77, 'Four-bit is a weight format, not a request budget.', fontsize=22, color='#a8c6d6')
    fig.text(.10, .65, 'Packed matrix', fontsize=24, weight='bold')
    fig.text(.10, .57, '0.50 MiB payload + 0.0625 MiB scales', fontsize=18)
    left = fig.add_axes([.10, .34, .31, .16], facecolor='#101e2c')
    p = data['packing']
    left.barh([0], [p['packed_bytes']/2**20], color='#63d7c0', height=.55)
    left.barh([0], [p['scale_bytes']/2**20], left=[p['packed_bytes']/2**20], color='#ffb366', height=.55)
    left.set_xlim(0, .6)
    left.set_axis_off()
    fig.text(.10, .29, 'Restored FP32 array: 4 MiB', fontsize=20, color='#ffb366')
    fig.text(.10, .23, 'Simple unpacking demo only', fontsize=17, color='#a8c6d6')
    fig.text(.57, .65, 'Real FP16 KV payload', fontsize=24, weight='bold')
    right = fig.add_axes([.61, .30, .27, .27], facecolor='#101e2c')
    vals = [r['tensor_bytes']/2**20 for r in data['cache']]
    right.bar(range(3), vals, color=['#7196c9','#7fb8ce','#63d7c0'], width=.6)
    for i,v in enumerate(vals):
        right.text(i, v+.3, f'{v:g}', ha='center', fontsize=20)
    right.set_ylim(0,15)
    right.set_xticks(range(3), ['1×128','1×512','2×512'], fontsize=17)
    right.set_yticks([])
    right.tick_params(axis='x', length=0, pad=8)
    for spine in right.spines.values():
        spine.set_visible(False)
    fig.text(.65, .21, 'Batch × tokens  ·  MiB', fontsize=17, color='#a8c6d6')
    fig.text(.08, .10, 'Separate bounded experiments — not total process memory', fontsize=20)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for item in fig.findobj(Text):
        if item.get_visible() and item.get_text():
            box = item.get_window_extent(renderer)
            assert box.x0 >= 80 and box.x1 <= 1120 and box.y0 >= 48 and box.y1 <= 582, (item.get_text(), box)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=100, facecolor=fig.get_facecolor())
    print('PASS: 1200x630 text safe area; inspect original, CSS crop, and theme separately.')


if __name__ == '__main__':
    main()
