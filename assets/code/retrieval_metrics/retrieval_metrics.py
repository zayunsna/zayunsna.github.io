"""Hand-check retrieval metrics on constructed rankings, not a benchmark."""
import json
import math
from pathlib import Path
import sys
import numpy as np
from sklearn.metrics import ndcg_score


def metrics(ranking, grades, k):
    """Unique doc IDs; full known relevance set; grades are nonnegative."""
    assert k > 0 and len(ranking) == len(set(ranking))
    assert set(ranking) <= set(grades)
    relevant = {doc for doc, grade in grades.items() if grade > 0}
    if not relevant:
        raise ValueError('No relevant documents: choose and document a policy')
    top = ranking[:k]
    recall = len(set(top) & relevant) / len(relevant)
    rr = next((1 / i for i, doc in enumerate(top, 1) if doc in relevant), 0.)
    dcg = sum((2 ** grades[doc] - 1) / math.log2(i + 1)
              for i, doc in enumerate(top, 1))
    ideal = sorted(grades.values(), reverse=True)[:k]
    idcg = sum((2 ** grade - 1) / math.log2(i + 1)
               for i, grade in enumerate(ideal, 1))
    return dict(recall=recall, rr=rr, ndcg=dcg / idcg)


grades = {'A': 3, 'B': 1, 'C': 1, 'X': 0, 'Y': 0}
rankings = {'one answer first': ['A', 'X', 'Y', 'B', 'C'],
            'more answers later': ['X', 'B', 'C', 'A', 'Y'],
            'best answer later': ['B', 'C', 'A', 'X', 'Y'],
            'best answer first': ['A', 'B', 'C', 'X', 'Y']}
rows = []
print('k=3; relevant grades > 0; NDCG gain=2**grade-1')
print('ranking                 Recall@3  RR@3  NDCG@3')
for name, ranking in rankings.items():
    row = dict(name=name, ranking=ranking, **metrics(ranking, grades, 3))
    # sklearn uses the supplied y_true as gain, so transform grades explicitly.
    docs = list(grades)
    gains = np.array([[2 ** grades[d] - 1 for d in docs]])
    scores = np.array([[len(ranking) - ranking.index(d) for d in docs]])
    assert np.isclose(row['ndcg'], ndcg_score(gains, scores, k=3))
    rows.append(row)
    print(f"{name:24s} {row['recall']:.3f}    {row['rr']:.3f}  {row['ndcg']:.3f}")
print('MRR@3 across these 4 artificial queries:', f"{np.mean([r['rr'] for r in rows]):.3f}")
assert metrics(['X', 'Y'], grades, 2) == dict(recall=0., rr=0., ndcg=0.)
try:
    metrics(['X'], {'X': 0}, 1)
except ValueError as exc:
    print(type(exc).__name__ + ':', str(exc))
print('PASS: NDCG matches sklearn with transformed gains; no-hit and no-relevance cases checked')
if len(sys.argv) > 1:
    Path(sys.argv[1]).write_text(json.dumps(dict(grades=grades, k=3, rows=rows), indent=2) + '\n')
