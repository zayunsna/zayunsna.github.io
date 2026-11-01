"""Measured warm latency and observed process ranges; not vendor benchmarks."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.text import Text


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    rows = json.loads(Path(__file__).with_name('summary.json').read_text())
    plt.rcParams.update({'text.color':'#f3f0ed', 'axes.labelcolor':'#f3f0ed',
                         'xtick.color':'#c7c0b8', 'ytick.color':'#f3f0ed', 'font.size':20})
    fig = plt.figure(figsize=(12,6.3), dpi=100, facecolor='#241e1c')
    fig.text(.08,.85,'PyTorch 2.15 on an M5',fontsize=34,weight='bold')
    fig.text(.08,.76,'Same MLP. Different batch sizes. Modest gains.',fontsize=22,color='#d8c6b8')
    ax = fig.add_axes([.24,.28,.62,.36],facecolor='#241e1c')
    for y,batch in enumerate([1,64]):
        for offset,version,color in [(.16,'2.14.0','#b0a29a'),(-.16,'2.15.0','#ffa96b')]:
            row = next(r for r in rows if r['version']==version and r['batch']==batch)
            value=row['median_ms']
            ax.barh(y+offset,value,height=.26,color=color,label=version if y==0 else None)
            ax.errorbar(value,y+offset,xerr=[[value-row['min_ms']],[row['max_ms']-value]],
                        fmt='none',ecolor='#ffffff',capsize=5,lw=2)
            ax.text(row['max_ms']+.014,y+offset,f'{value:.3f} ms',va='center',fontsize=20)
    ax.set_xlim(0,.59)
    ax.set_yticks([0,1],['Batch 1','Batch 64'])
    ax.invert_yaxis()
    ax.set_xticks([0,.2,.4])
    ax.tick_params(length=0,pad=9)
    ax.set_xlabel('Warm forward time · lower is better',labelpad=10,fontsize=18)
    for spine in ax.spines.values(): spine.set_visible(False)
    ax.legend(loc='upper left',bbox_to_anchor=(.02,1.30),frameon=False,ncols=2,fontsize=18)
    fig.text(.08,.10,'Three processes/version · whiskers: observed range, not a confidence interval',fontsize=17)
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    for item in fig.findobj(Text):
        if item.get_visible() and item.get_text():
            box=item.get_window_extent(renderer)
            assert box.x0>=80 and box.x1<=1120 and box.y0>=48 and box.y1<=582,(item.get_text(),box)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.output,dpi=100,facecolor=fig.get_facecolor())
    print('PASS: 1200x630 text safe area; inspect saved image and actual crops separately.')


if __name__=='__main__': main()
