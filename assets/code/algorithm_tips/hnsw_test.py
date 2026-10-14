"""Exact search vs HNSW on 134k real docstring paragraphs: recall and query time, with held-out queries."""
import importlib.metadata as md, sys, time
import numpy as np
import hnswlib

K = 10
emb = np.load("big.npy")
rng = np.random.default_rng(0)
perm = rng.permutation(len(emb))
queries, pool = emb[perm[:500]], emb[perm[500:]]

def exact(db, qs, k):
    scores = qs @ db.T                                   # cosine similarity: vectors are normalized
    top = np.argpartition(-scores, k, axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(scores, top, axis=1), axis=1)
    return np.take_along_axis(top, order, axis=1)

def best_time(fn, repeat=3):
    best = float("inf")
    for _ in range(repeat):
        t = time.perf_counter(); out = fn(); best = min(best, time.perf_counter() - t)
    return out, best

def recall(found, truth):
    return np.mean([len(set(a) & set(b)) / truth.shape[1] for a, b in zip(found, truth)])

for n in (10_000, 50_000, len(pool)):
    db = pool[:n]
    truth, t_exact = best_time(lambda: exact(db, queries, K))
    index = hnswlib.Index(space="ip", dim=db.shape[1])
    index.init_index(max_elements=n, M=16, ef_construction=200, random_seed=100)
    t = time.perf_counter(); index.add_items(db, np.arange(n)); build = time.perf_counter() - t
    index.set_num_threads(1)
    print(f"n={n:,}: exact {t_exact / 500 * 1e3:.3f} ms/query (recall 1.000), HNSW build {build:.1f} s")
    for ef in (10, 40, 160):
        index.set_ef(ef)
        (found, _), t = best_time(lambda: index.knn_query(queries, k=K))
        print(f"   HNSW ef={ef:<4} recall@10 {recall(found, truth):.3f}  {t / 500 * 1e3:.3f} ms/query")
    if n == len(pool):
        print("   greedy walk, top-1 correct:", "  ".join(
            f"ef={ef}: {np.mean(index.knn_query(queries, k=1)[0][:, 0] == truth[:, 0]):.3f}"
            for ef in (1, 2, 4, 8, 16, 32) if not index.set_ef(ef)))
print("Python", sys.version.split()[0], "| numpy", np.__version__, "| hnswlib", md.version("hnswlib"),
      "| sentence-transformers", md.version("sentence-transformers"))
