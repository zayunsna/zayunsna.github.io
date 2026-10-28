"""BM25 vs embeddings vs hybrid (reciprocal rank fusion), on SQuAD v1.1 dev (CC BY-SA 4.0).

Same corpus and questions as rerank.py: 2,067 paragraphs, 1,000 random questions (seed 0).
Run: python hybrid.py path/to/dev-v1.1.json [cpu|mps]
"""
import json
import platform
import random
import re
import sys
import time

import numpy as np
import sentence_transformers
import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

BI, CROSS = "sentence-transformers/all-MiniLM-L6-v2", "cross-encoder/ms-marco-MiniLM-L6-v2"
N_QUERIES, SEED, POOL, RRF_K, RERANK_K = 1_000, 0, 100, 60, 20
DEVICE = sys.argv[2] if len(sys.argv) > 2 else "cpu"


def load_squad(path):
    paragraphs, questions = [], []
    for article in json.load(open(path))["data"]:
        for p in article["paragraphs"]:
            paragraphs.append(p["context"])
            questions += [(qa["question"], len(paragraphs) - 1) for qa in p["qas"]]
    return paragraphs, questions


def tokenize(text):
    return re.findall(r"\w+", text.lower())


def rrf(*rankings, k=RRF_K):
    """Reciprocal rank fusion: each list adds 1 / (k + rank) to a document's score."""
    scores = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=scores.get, reverse=True)


def rank_of(order, gold):
    order = list(order)
    return order.index(gold) + 1 if gold in order else None


def hits(ranks, k):
    return np.mean([r is not None and r <= k for r in ranks])


def main():
    print(f"python {sys.version.split()[0]}, torch {torch.__version__}, "
          f"sentence-transformers {sentence_transformers.__version__}, rank-bm25 0.2.2, "
          f"{platform.machine()}, device {DEVICE}")
    paragraphs, questions = load_squad(sys.argv[1])
    sample = random.Random(SEED).sample(questions, N_QUERIES)
    q_texts = [q for q, _ in sample]
    golds = [g for _, g in sample]
    print(f"corpus {len(paragraphs):,} paragraphs, sampled {N_QUERIES:,} questions")

    bm25 = BM25Okapi([tokenize(p) for p in paragraphs])
    t0 = time.perf_counter()
    bm25_top = [np.argsort(-bm25.get_scores(tokenize(q)))[:POOL].tolist() for q in q_texts]
    bm25_ms = (time.perf_counter() - t0) / N_QUERIES * 1000

    bi = SentenceTransformer(BI, device=DEVICE)
    doc_vecs = bi.encode(paragraphs, batch_size=64, normalize_embeddings=True, convert_to_numpy=True)
    t0 = time.perf_counter()
    q_vecs = bi.encode(q_texts, batch_size=64, normalize_embeddings=True, convert_to_numpy=True)
    dense_top = np.argsort(-(q_vecs @ doc_vecs.T), axis=1)[:, :POOL].tolist()
    dense_ms = (time.perf_counter() - t0) / N_QUERIES * 1000

    hybrid_top = [rrf(b, d)[:POOL] for b, d in zip(bm25_top, dense_top)]

    runs = {"BM25": bm25_top, "embeddings": dense_top, "hybrid (RRF)": hybrid_top}
    ranks = {name: [rank_of(top, g) for top, g in zip(tops, golds)] for name, tops in runs.items()}
    print(f"\n{'method':26s} {'hit@1':>6} {'hit@5':>6} {'hit@20':>6} {'hit@100':>7}")
    for name, r in ranks.items():
        print(f"{name:26s} {hits(r, 1):6.3f} {hits(r, 5):6.3f} {hits(r, 20):6.3f} {hits(r, 100):7.3f}")
    print(f"search time per query: BM25 {bm25_ms:.1f} ms (rank-bm25, pure Python), "
          f"embeddings {dense_ms:.2f} ms (query encoding + dot product)")

    # Where each one wins.
    b1 = [r == 1 for r in ranks["BM25"]]
    d1 = [r == 1 for r in ranks["embeddings"]]
    print(f"\nrank 1 correct: both {sum(a and b for a, b in zip(b1, d1))}, "
          f"BM25 only {sum(a and not b for a, b in zip(b1, d1))}, "
          f"embeddings only {sum(b and not a for a, b in zip(b1, d1))}, "
          f"neither {sum(not a and not b for a, b in zip(b1, d1))}")
    examples = {"BM25 only": [], "embeddings only": []}
    for i, (a, b) in enumerate(zip(b1, d1)):
        key = "BM25 only" if a and not b else "embeddings only" if b and not a else None
        if key and len(examples[key]) < 4:
            examples[key].append(f"{q_texts[i]}  (BM25 rank {ranks['BM25'][i]}, "
                                 f"embeddings rank {ranks['embeddings'][i]})")
    for key, qs in examples.items():
        print(f"\n{key}, first 4:")
        print("\n".join(f"- {q}" for q in qs))

    # The candidate list feeds a reranker: does a better list help after reranking?
    cross = CrossEncoder(CROSS, device=DEVICE)
    print(f"\nrerank the top {RERANK_K} with {CROSS}:")
    reranked = {}
    for name in ("embeddings", "hybrid (RRF)"):
        r = []
        for q, top, g in zip(q_texts, runs[name], golds):
            cand = top[:RERANK_K]
            scores = cross.predict([(q, paragraphs[j]) for j in cand], batch_size=RERANK_K)
            r.append(rank_of([cand[j] for j in np.argsort(-scores)], g))
        reranked[name] = r
        print(f"{name + ' + rerank':26s} hit@1 {hits(r, 1):.3f}   "
              f"gold in top {RERANK_K} before reranking {hits(ranks[name], RERANK_K):.3f}")

    json.dump({name: {"hit@1": hits(r, 1), "hit@5": hits(r, 5), "hit@20": hits(r, 20), "hit@100": hits(r, 100)}
               for name, r in ranks.items()}
              | {f"{name} + rerank": {"hit@1": hits(r, 1)} for name, r in reranked.items()},
              open(f"hybrid_results_{DEVICE}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
