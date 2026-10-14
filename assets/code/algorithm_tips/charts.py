"""Charts and 1200x630 covers, parsed from the saved experiment outputs (no retyped numbers)."""
import json, re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BG, FG, MUTED, A, B, C = "#1e1f26", "#e8e8ef", "#9a9ab0", "#4fb3bf", "#f2a65a", "#e06c9f"
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "axes.edgecolor": MUTED, "axes.labelcolor": FG,
                     "xtick.color": FG, "ytick.color": FG, "text.color": FG, "font.size": 13, "axes.grid": True,
                     "grid.color": "#33343f", "legend.facecolor": BG, "legend.edgecolor": MUTED})

def cover(fig, path):
    fig.savefig(path, dpi=100); plt.close(fig)

# 1. HNSW
txt = open("hnsw_out.txt").read()
blocks = re.findall(r"n=([\d,]+): exact ([\d.]+) ms/query.*?ef=40\s+recall@10 ([\d.]+)\s+([\d.]+) ms", txt, re.S)
ns = [int(b[0].replace(",", "")) for b in blocks]
ex = [float(b[1]) for b in blocks]; hn = [float(b[3]) for b in blocks]; rc = [float(b[2]) for b in blocks]
walk = [(int(e), float(v)) for e, v in re.findall(r"ef=(\d+): ([\d.]+)", txt.split("greedy walk")[1])]
def hnsw_fig(w, h, title=None):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(w, h))
    a1.plot(ns, ex, "o-", color=B, lw=2.5, label="exact (every vector)")
    a1.plot(ns, hn, "o-", color=A, lw=2.5, label="HNSW, ef=40")
    for x, y, r in zip(ns, hn, rc):
        a1.annotate(f"recall {r:.2f}", (x, y), textcoords="offset points", xytext=(0, 10), ha="center", color=A, fontsize=11)
    a1.set_xlabel("vectors in the index"); a1.set_ylabel("ms per query"); a1.legend(loc="upper left")
    a1.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:.0f}k"))
    a1.set_title("Query time as the index grows", color=FG)
    a2.plot([e for e, _ in walk], [v * 100 for _, v in walk], "o-", color=C, lw=2.5)
    a2.set_xscale("log", base=2); a2.set_xticks([e for e, _ in walk]); a2.set_xticklabels([str(e) for e, _ in walk])
    a2.set_ylim(0, 100); a2.set_xlabel("ef (candidates kept during the walk)"); a2.set_ylabel("true nearest neighbour found (%)")
    a2.set_title("ef=1 is a pure greedy walk", color=FG)
    if title: fig.suptitle(title, fontsize=22, fontweight="bold", x=0.04, ha="left", y=0.97)
    fig.tight_layout(rect=(0, 0, 1, 0.9 if title else 1))
    return fig
cover(hnsw_fig(12, 5), "img/hnsw_chart.png")
cover(hnsw_fig(12, 6.3, f"Vector search on {ns[-1]:,} real docstrings: exact vs HNSW"), "img/hnsw_cover.png")

# 2. KV cache
rows = [tuple(map(float, m)) for m in re.findall(r"(\d+) new tokens: cache on\s+([\d.]+) s \| cache off\s+([\d.]+) s", open("kv_out_1.txt").read())]
def kv_fig(w, h, title=None):
    fig, ax = plt.subplots(figsize=(w, h))
    x = [r[0] for r in rows]
    ax.plot(x, [r[2] for r in rows], "o-", color=B, lw=2.5, label="cache off: recompute every token")
    ax.plot(x, [r[1] for r in rows], "o-", color=A, lw=2.5, label="cache on: reuse keys and values")
    for r in rows:
        ax.annotate(f"{r[2] / r[1]:.1f}x", (r[0], r[2]), textcoords="offset points", xytext=(-6, 8), ha="right", color=B)
    ax.set_xlabel("new tokens generated"); ax.set_ylabel("seconds"); ax.set_xticks(x); ax.legend(loc="upper left")
    ax.set_title("Qwen2.5-0.5B-Instruct, greedy, CPU float32: identical output tokens in every run", color=MUTED, fontsize=12)
    if title: fig.suptitle(title, fontsize=22, fontweight="bold", x=0.04, ha="left", y=0.97)
    fig.tight_layout(rect=(0, 0, 1, 0.9 if title else 1))
    return fig
cover(kv_fig(11, 5), "img/kv_chart.png")
cover(kv_fig(12, 6.3, "KV cache: the same 256 tokens, 7.5 s instead of 34 s"), "img/kv_cover.png")

# 3. Decoding
steps = json.load(open("decoding_steps.json"))
def dec_fig(w, h, title=None):
    fig, ax = plt.subplots(figsize=(w, h))
    for label, color in (("greedy", B), ("beam 8", A)):
        lp = steps[label]["logprobs"]; cum = [sum(lp[:i + 1]) for i in range(len(lp))]
        ax.plot(range(1, len(cum) + 1), cum, "o-", color=color, lw=2.5, label=f"{label}: total {cum[-1]:.2f}")
        i = next(k for k, (a, b) in enumerate(zip(steps["greedy"]["tokens"], steps["beam 8"]["tokens"])) if a != b)
        word = steps[label]["tokens"][i].strip()
        ax.annotate(f'token {i + 1}: "{word}" ({lp[i]:.2f})', (i + 1, cum[i]), textcoords="axes fraction",
                    xytext=(0.45, 0.88 if label == "beam 8" else 0.76), color=color, fontsize=13,
                    arrowprops=dict(arrowstyle="->", color=color))
    ax.set_xlabel("token"); ax.set_ylabel("cumulative log-probability"); ax.legend(loc="lower left")
    ax.set_title('"The main difference between a list and a tuple in Python is ..."', color=MUTED, fontsize=12)
    if title: fig.suptitle(title, fontsize=22, fontweight="bold", x=0.04, ha="left", y=0.97)
    fig.tight_layout(rect=(0, 0, 1, 0.9 if title else 1))
    return fig
cover(dec_fig(11, 5), "img/decoding_chart.png")
cover(dec_fig(12, 6.3, "Greedy picked the likelier next word, and lost overall"), "img/decoding_cover.png")

# 4. Formats: the real bit patterns of 0.1
lay = dict(re.findall(r"(fp32|fp16|bf16): .*?0\.1 = ([01 ]+)$", open("formats_out.txt").read(), re.M))
sums = dict(re.findall(r"(fp32|fp16|bf16): 1 \+ 0\.001 = [\d.]+ \| running sum = ([\d.]+)", open("formats_out.txt").read()))
def fmt_fig(w, h, title=None):
    fig, ax = plt.subplots(figsize=(w, h)); ax.axis("off"); ax.set_xlim(0, 33); ax.set_ylim(-0.5, 3.6)
    cols = ("#e06c9f", "#4fb3bf", "#6b6f8a")
    for row, name in enumerate(("fp32", "fp16", "bf16")):
        y = 2.6 - row * 1.1
        ax.text(0, y + 0.35, name, fontsize=18, fontweight="bold", va="center")
        x = 2.6
        for part, col in zip(lay[name].strip().split(" "), cols):
            for bit in part:
                ax.add_patch(plt.Rectangle((x, y), 0.9, 0.7, color=col, alpha=0.9))
                ax.text(x + 0.45, y + 0.35, bit, ha="center", va="center", fontsize=12, color="#111")
                x += 0.95
        ax.text(x + 0.3, y + 0.35, f"sum of 0.001 x 10,000 = {float(sums[name]):g}", va="center", fontsize=13, color=FG)
    ax.text(2.6, -0.3, "pink: sign   teal: exponent (range)   grey: fraction (precision)       each row stores 0.1",
            fontsize=12, color=MUTED)
    if title: fig.suptitle(title, fontsize=22, fontweight="bold", x=0.04, ha="left", y=0.96)
    fig.tight_layout(rect=(0, 0, 1, 0.9 if title else 1))
    return fig
cover(fmt_fig(13, 4.2), "img/formats_chart.png")
cover(fmt_fig(12, 6.3, "fp32, fp16, bf16: where the 16 bits go"), "img/formats_cover.png")
print(ns, ex, hn, rc, walk, rows, lay, sums, sep="\n")
