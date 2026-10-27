"""Bi-encoder retrieval vs bi-encoder + cross-encoder reranking, on SQuAD v1.1 dev (CC BY-SA 4.0).

Corpus: every unique paragraph of the dev set (2,067). Queries: 1,000 random questions (seed 0).
A question counts as answered when its own source paragraph is ranked first.
Run: python rerank.py path/to/dev-v1.1.json [cpu|mps]
"""
import json
import platform
import random
import statistics
import sys
import time

import numpy as np
import sentence_transformers
import torch
import transformers
from sentence_transformers import CrossEncoder, SentenceTransformer

BI, CROSS = "sentence-transformers/all-MiniLM-L6-v2", "cross-encoder/ms-marco-MiniLM-L6-v2"
N_QUERIES, DEPTHS, SEED = 1_000, [5, 10, 20, 50, 100], 0
DEVICE = sys.argv[2] if len(sys.argv) > 2 else "cpu"


def load_squad(path):
    paragraphs, questions = [], []
    for article in json.load(open(path))["data"]:
        for p in article["paragraphs"]:
            paragraphs.append(p["context"])
            questions += [(qa["question"], len(paragraphs) - 1) for qa in p["qas"]]
    return paragraphs, questions


def rank_of(order, gold):
    hits = np.where(order == gold)[0]
    return int(hits[0]) + 1 if len(hits) else None


def summarize(ranks):
    hit = lambda k: np.mean([r is not None and r <= k for r in ranks])
    mrr = np.mean([1 / r if r is not None and r <= 10 else 0 for r in ranks])
    return {"hit@1": hit(1), "hit@3": hit(3), "hit@10": hit(10), "mrr@10": mrr}


def main():
    print(f"python {sys.version.split()[0]}, torch {torch.__version__}, transformers {transformers.__version__}, "
          f"sentence-transformers {sentence_transformers.__version__}, {platform.machine()}, device {DEVICE}")
    paragraphs, questions = load_squad(sys.argv[1])
    sample = random.Random(SEED).sample(questions, N_QUERIES)
    print(f"corpus {len(paragraphs):,} paragraphs, {len(questions):,} questions, sampled {N_QUERIES:,}")

    bi = SentenceTransformer(BI, device=DEVICE)
    cross = CrossEncoder(CROSS, device=DEVICE)
    doc_vecs = bi.encode(paragraphs, batch_size=64, normalize_embeddings=True, convert_to_numpy=True)
    q_texts = [q for q, _ in sample]
    golds = np.array([g for _, g in sample])

    # Warm up both models so the timings below do not include first-call setup.
    bi.encode(q_texts[:8], normalize_embeddings=True)
    cross.predict([(q_texts[0], paragraphs[0])] * 8)

    t0 = time.perf_counter()
    q_vecs = bi.encode(q_texts, batch_size=64, normalize_embeddings=True, convert_to_numpy=True)
    order = np.argsort(-(q_vecs @ doc_vecs.T), axis=1)
    bi_ms = (time.perf_counter() - t0) / N_QUERIES * 1000
    bi_ranks = [rank_of(order[i], golds[i]) for i in range(N_QUERIES)]
    results = {"bi-encoder only": {**summarize(bi_ranks), "ms_per_query": bi_ms}}
    print(f"\n{'method':28s} {'hit@1':>6} {'hit@3':>6} {'hit@10':>6} {'MRR@10':>7} {'ms/query':>9}  pairs scored")
    r = results["bi-encoder only"]
    print(f"{'bi-encoder only':28s} {r['hit@1']:6.3f} {r['hit@3']:6.3f} {r['hit@10']:6.3f} {r['mrr@10']:7.3f} "
          f"{bi_ms:9.2f}  0")

    examples = []
    for k in DEPTHS:
        ranks, secs = [], []
        for i, q in enumerate(q_texts):
            cand = order[i, :k]
            t0 = time.perf_counter()
            scores = cross.predict([(q, paragraphs[j]) for j in cand], batch_size=k)
            secs.append(time.perf_counter() - t0)
            reranked = cand[np.argsort(-scores)]
            ranks.append(rank_of(reranked, golds[i]))
            if k == 20 and len(examples) < 3 and bi_ranks[i] and bi_ranks[i] > 1 and ranks[-1] == 1:
                examples.append({"question": q, "bi_rank": bi_ranks[i], "bi_top": paragraphs[order[i, 0]],
                                 "gold": paragraphs[golds[i]]})
        ceiling = np.mean([br is not None and br <= k for br in bi_ranks])
        name = f"+ rerank top {k}"
        results[name] = {**summarize(ranks), "ms_per_query": bi_ms + statistics.mean(secs) * 1000,
                         "rerank_ms_median": statistics.median(secs) * 1000, "ceiling": ceiling}
        r = results[name]
        print(f"{name:28s} {r['hit@1']:6.3f} {r['hit@3']:6.3f} {r['hit@10']:6.3f} {r['mrr@10']:7.3f} "
              f"{r['ms_per_query']:9.2f}  {k}   (gold in top {k} before reranking: {ceiling:.3f})")

    # What the reranker cannot fix: the right paragraph was never retrieved.
    k = 20
    missed = sum(br is None or br > k for br in bi_ranks)
    print(f"\nwith top {k}: {missed} of {N_QUERIES} questions never had their paragraph in the candidate list")

    helped = sum(1 for i in range(N_QUERIES) if bi_ranks[i] and bi_ranks[i] > 1 and bi_ranks[i] <= 20)
    print(f"bi-encoder rank 2-20 (reranker can still lift to #1): {helped}")

    print("\nexamples where the bi-encoder ranked the wrong paragraph first and top-20 reranking fixed it:")
    for ex in examples:
        print(f"- Q: {ex['question']}\n  bi-encoder #1: {ex['bi_top'][:160]!r}...\n"
              f"  correct (bi-encoder rank {ex['bi_rank']}): {ex['gold'][:160]!r}...")
    json.dump({"results": results, "examples": examples}, open(f"rerank_results_{DEVICE}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
