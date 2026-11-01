"""Cosine similarity is not confidence: choosing a retrieval threshold. Full script for the blog post.

Data: SQuAD v1.1 dev set (CC BY-SA 4.0), https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json
Run:  python cosine_threshold.py   (expects dev-v1.1.json in the current directory; writes results/)
The steps match the post. This file adds version logging, the score chart, a 10-split robustness check
and results/cosine_threshold_results.json.
"""
import platform
import sys
import time
from pathlib import Path

STARTED = time.perf_counter()
import json
import random

import numpy as np
from sentence_transformers import SentenceTransformer
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sentence_transformers
import torch
import transformers

print(f"python {sys.version.split()[0]}, sentence-transformers {sentence_transformers.__version__}, "
      f"transformers {transformers.__version__}, torch {torch.__version__}, numpy {np.__version__}, "
      f"{platform.system()} {platform.machine()}")
OUT = Path("results")
OUT.mkdir(exist_ok=True)

articles = json.load(open("dev-v1.1.json"))["data"]
order = list(range(len(articles)))
random.Random(0).shuffle(order)
groups = {"val": (order[:16], order[32:40]), "test": (order[16:32], order[40:48])}

corpus, questions = [], {}
for split, (in_corpus, left_out) in groups.items():
    answerable, unanswerable = [], []
    for a in in_corpus:
        for p in articles[a]["paragraphs"]:
            corpus.append(p["context"])
            answerable += [(qa["question"], len(corpus) - 1) for qa in p["qas"]]
    for a in left_out:
        unanswerable += [(qa["question"], -1) for p in articles[a]["paragraphs"] for qa in p["qas"]]
    rng = random.Random(1)
    questions[split] = rng.sample(answerable, 500) + rng.sample(unanswerable, 500)
print("corpus paragraphs:", len(corpus))
print({split: len(q) for split, q in questions.items()})

# ---- embed and keep the best score ----
def retrieve(model_name, revision, query_prefix=""):
    model = SentenceTransformer(model_name, revision=revision, device="cpu")
    docs = model.encode(corpus, batch_size=64, normalize_embeddings=True)
    result = {}
    for split, qs in questions.items():
        q = model.encode([query_prefix + text for text, _ in qs], batch_size=64, normalize_embeddings=True)
        sims = q @ docs.T  # cosine similarity, because the vectors are normalized
        top3 = np.argsort(-sims, axis=1)[:, :3]
        gold = np.array([g for _, g in qs])
        result[split] = {"score": sims.max(axis=1), "correct": (top3 == gold[:, None]).any(axis=1),
                         "answerable": gold >= 0}
    return result

mini = retrieve("sentence-transformers/all-MiniLM-L6-v2", "1110a243fdf4706b3f48f1d95db1a4f5529b4d41")
v = mini["val"]
for name, mask in [("right one in top 3", v["correct"]),
                   ("right one missed", v["answerable"] & ~v["correct"]),
                   ("no answer in corpus", ~v["answerable"])]:
    print(f"{name:20s} n={mask.sum():3d}  top-1 score median {np.median(v['score'][mask]):.3f}")

# ---- precision, coverage and the cutoff ----
def precision_coverage(t, r):
    if t is None:
        raise ValueError("No validation cutoff met the target; do not evaluate a missing threshold.")
    answered = r["score"] >= t
    precision = r["correct"][answered].mean() if answered.any() else float("nan")
    return precision, answered.mean()

def choose_threshold(r, target=0.80):
    for t in np.round(np.arange(0.0, 1.0, 0.01), 2):
        if precision_coverage(t, r)[0] >= target:
            return t
    return None

for t in [0.3, 0.4, 0.5, 0.6, 0.7]:
    p, c = precision_coverage(t, mini["val"])
    print(f"t={t:.1f}  precision {p:.3f}  coverage {c:.3f}")
t_mini = choose_threshold(mini["val"])
p, c = precision_coverage(t_mini, mini["test"])
print(f"chosen on validation: t={t_mini:.2f}  ->  test precision {p:.3f}, coverage {c:.3f}")


def plot_scores(r, t, path):
    """Histogram of top-1 cosine scores on validation, by outcome, with the chosen cutoff."""
    bg, fg, muted = "#1e1f26", "#e8e8ef", "#9a9ab0"
    plt.rcParams.update({"figure.facecolor": bg, "axes.facecolor": bg, "axes.edgecolor": muted,
                         "axes.labelcolor": fg, "xtick.color": fg, "ytick.color": fg,
                         "text.color": fg, "font.size": 12, "legend.facecolor": bg, "legend.edgecolor": muted})
    bins = np.arange(0.1, 0.91, 0.025)
    groups = [("right paragraph in top 3", r["correct"], "#4fb3bf"),
              ("answerable, right one missed", r["answerable"] & ~r["correct"], "#f2a65a"),
              ("excluded gold article (proxy negative)", ~r["answerable"], "#9a8fd0")]
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.hist([r["score"][m] for _, m, _ in groups], bins=bins, stacked=True,
            color=[c for _, _, c in groups], label=[f"{n} ({m.sum()})" for n, m, _ in groups])
    ax.set_ylim(0, ax.get_ylim()[1] * 1.45)
    ax.set_xlim(0.15, 0.9)
    ax.axvline(t, color=fg, ls="--", lw=1.2)
    ax.text(t + 0.005, ax.get_ylim()[1] * 0.95, f"cutoff {t:.2f}: answer \u2192", va="top", color=fg)
    ax.text(t - 0.005, ax.get_ylim()[1] * 0.95, "\u2190 say \"no answer\"", va="top", ha="right", color=fg)
    ax.set_xlabel("top-1 cosine score (all-MiniLM-L6-v2), 1,000 validation questions")
    ax.set_ylabel("questions")
    ax.legend(loc="upper right", bbox_to_anchor=(1, 0.86))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


plot_scores(mini["val"], t_mini, OUT / "score_hist.png")

# ---- score bins ----
r = mini["val"]
for lo in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
    in_bin = (r["score"] >= lo) & (r["score"] < lo + 0.1)
    if in_bin.any():
        print(f"score {lo:.1f}-{lo + 0.1:.1f}: n={in_bin.sum():3d}  right paragraph in top 3: "
              f"{r['correct'][in_bin].mean():.0%}")

# ---- another model ----
bge = retrieve("BAAI/bge-small-en-v1.5", "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
               query_prefix="Represent this sentence for searching relevant passages: ")
print(f"bge median top score, no answer in corpus: {np.median(bge['val']['score'][~bge['val']['answerable']]):.3f}")
t_bge = choose_threshold(bge["val"])
for label, t in [(f"MiniLM's cutoff {t_mini:.2f}", t_mini), (f"its own cutoff {t_bge:.2f}", t_bge)]:
    p, c = precision_coverage(t, bge["test"])
    print(f"bge with {label}: test precision {p:.3f}, coverage {c:.3f}")

# ---- a different question mix ----
r = mini["test"]
keep = np.r_[np.arange(500), 500 + np.arange(56)]
mostly_answerable = {key: value[keep] for key, value in r.items()}
p, c = precision_coverage(t_mini, mostly_answerable)
print(f"t={t_mini:.2f} with ~10% unanswerable: precision {p:.3f}, coverage {c:.3f}")


# ---- robustness: repeat the MiniLM threshold choice on 10 different article splits ----
mini_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2",
                                 revision="1110a243fdf4706b3f48f1d95db1a4f5529b4d41", device="cpu")
split_runs = []
print("\nMiniLM, 80% target, other random article splits (seed 0 is the worked example above):")
for split_seed in range(10):
    order = list(range(len(articles)))
    random.Random(split_seed).shuffle(order)
    groups = {"val": (order[:16], order[32:40]), "test": (order[16:32], order[40:48])}
    corpus, questions = [], {}
    for split, (in_corpus, left_out) in groups.items():
        answerable, unanswerable = [], []
        for a in in_corpus:
            for p in articles[a]["paragraphs"]:
                corpus.append(p["context"])
                answerable += [(qa["question"], len(corpus) - 1) for qa in p["qas"]]
        for a in left_out:
            unanswerable += [(qa["question"], -1) for p in articles[a]["paragraphs"] for qa in p["qas"]]
        rng = random.Random(1)
        questions[split] = rng.sample(answerable, 500) + rng.sample(unanswerable, 500)
    docs = mini_model.encode(corpus, batch_size=64, normalize_embeddings=True)
    res = {}
    for split, qs in questions.items():
        q = mini_model.encode([text for text, _ in qs], batch_size=64, normalize_embeddings=True)
        sims = q @ docs.T
        top3 = np.argsort(-sims, axis=1)[:, :3]
        gold = np.array([g for _, g in qs])
        res[split] = {"score": sims.max(axis=1), "correct": (top3 == gold[:, None]).any(axis=1),
                      "answerable": gold >= 0}
    t = choose_threshold(res["val"])
    (pv, cv), (pt, ct) = precision_coverage(t, res["val"]), precision_coverage(t, res["test"])
    split_runs.append({"split_seed": split_seed, "cutoff": float(t), "val_precision": float(pv),
                       "val_coverage": float(cv), "test_precision": float(pt), "test_coverage": float(ct)})
    print(f"split seed {split_seed}: cutoff {t:.2f}  validation precision {pv:.3f}  "
          f"test precision {pt:.3f}  test coverage {ct:.3f}")
tp = [s["test_precision"] for s in split_runs]
print(f"test precision over 10 splits: min {min(tp):.3f}, median {np.median(tp):.3f}, max {max(tp):.3f}")


def curve(r):
    ts = np.round(np.arange(0.20, 0.86, 0.01), 2)
    pc = [precision_coverage(t, r) for t in ts]
    return {"cutoff": ts.tolist(), "precision": [float(p) if np.isfinite(p) else None for p, _ in pc],
            "coverage": [float(c) for _, c in pc]}


summary = {
    "t_mini": float(t_mini), "t_bge": float(t_bge),
    "splits": split_runs,
    "curves": {f"{name}_{split}": curve(res_[split]) for name, res_ in [("mini", mini), ("bge", bge)]
               for split in ("val", "test")},
    "top_score_median_val": {name: {group: float(np.median(res_["val"]["score"][mask(res_["val"])]))
                                    for group, mask in [("right_in_top3", lambda r: r["correct"]),
                                                        ("answerable_missed", lambda r: r["answerable"] & ~r["correct"]),
                                                        ("no_answer", lambda r: ~r["answerable"])]}
                             for name, res_ in [("mini", mini), ("bge", bge)]},
}
(OUT / "cosine_threshold_results.json").write_text(json.dumps(summary, indent=1, allow_nan=False) + "\n")
with open(OUT / "top_scores.csv", "w") as f:
    f.write("model,split,question,top_score,right_in_top3,answerable\n")
    for name, res_ in [("mini", mini), ("bge", bge)]:
        for split in ("val", "test"):
            r_ = res_[split]
            for i in range(len(r_["score"])):
                f.write(f"{name},{split},{i},{r_['score'][i]:.6f},{int(r_['correct'][i])},{int(r_['answerable'][i])}\n")
print(f"total run time: {time.perf_counter() - STARTED:.1f} s")
