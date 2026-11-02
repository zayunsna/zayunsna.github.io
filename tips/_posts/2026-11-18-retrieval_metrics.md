---
layout: post
title: "NDCG vs MRR vs Recall@k: what each metric rewards"
description: >
  Recall rewards finding more relevant documents, MRR rewards the first relevant hit, and NDCG rewards graded
  relevance near the top. Calculate all three on small rankings and match NDCG against scikit-learn.
image: /assets/img/tips/retrieval_metrics/cover.png
sitemap:
  changefreq: daily
  priority: 1.0
---

# NDCG vs MRR vs Recall@k: what each metric rewards

A retriever can improve Recall@3 while making its first useful result worse. It can also keep both recall and
reciprocal rank unchanged while moving the best document down. These are not contradictory results: the metrics
ask different questions.

Here are four constructed query results. For each query, the five-document corpus has the same relevance pattern:
**A has grade 3, B and C have grade 1, X and Y have grade 0.** Any grade above zero counts as relevant. These are
arithmetic examples, not measurements from a trained search model.

| Query result | Top 3 documents | Recall@3 | RR@3 | NDCG@3 |
|---|---|---:|---:|---:|
| One answer first | A, X, Y | 0.333 | 1.000 | 0.861 |
| More answers later | X, B, C | 0.667 | 0.500 | 0.139 |
| Best answer later | B, C, A | 1.000 | 1.000 | 0.631 |
| Best answer first | A, B, C | 1.000 | 1.000 | 1.000 |

The top two rows disagree on what “better” means. Recall prefers finding B and C. Reciprocal rank prefers any
relevant document at position 1. NDCG, with the gain convention below, strongly prefers placing the grade-3
document first. The last two rows show why reciprocal rank is not enough when relevance has levels.

_Executed on 2026-10-30 with Python 3.12.13, NumPy 2.5.4 and scikit-learn 1.9.1 on CPU. Complete
[script](/assets/code/retrieval_metrics/retrieval_metrics.py),
[output](/assets/code/retrieval_metrics/reference_output.txt) and
[run instructions](/assets/code/retrieval_metrics/README.txt). No dataset download or model is required._

## Recall@k: how much relevant material did you retrieve?

Recall@k is the number of relevant documents among the top k, divided by the number of relevant documents in
the judged corpus. The denominator here is **3**, not k: A, B and C are relevant.

`A, X, Y` contains one of them, so Recall@3 is `1/3`. `X, B, C` contains two, so it is `2/3`. Rearranging a fixed
top-k set cannot change recall. Moving a relevant document in from outside the cutoff can.

This is useful when a first-stage retriever must give a downstream reranker enough candidates. A reranker cannot
promote a document it never received. The [reranking experiment](/tips/2026-10-29-rag_reranking/) measures that
candidate ceiling with one designated paragraph per question. In that special one-relevant-document setup,
recall per query is 0 or 1 and its mean equals hit rate. That equivalence does not hold for multiple relevant documents.

## RR and MRR: where is the first relevant result?

For one query, reciprocal rank is `1 / first_relevant_rank`. With a cutoff k, it is zero if no relevant document
appears by k. The examples use this truncated RR@3 convention.

MRR@3 is the **mean** of those RR@3 values across queries. It is not a second score computed independently for
one ranking. Treating the four constructed rows above as four toy queries gives:

```text
(1 + 0.5 + 1 + 1) / 4 = 0.875
```

RR ignores everything after the first relevant hit. `B, C, A` and `A, B, C` both score 1 even though only one puts
the highly relevant document first. This can be appropriate for “find one acceptable answer,” but less informative
for a task that needs several pieces of evidence.

## NDCG rewards useful material near the top

Discounted cumulative gain adds a relevance gain at each rank, discounted by `log2(rank + 1)`. Divide by the
best possible DCG at the same cutoff to get **normalized** DCG.

This example uses `gain = 2**grade - 1`: grades 3, 1 and 0 give gains 7, 1 and 0. For `A, X, Y`, DCG@3 is 7.
The ideal order is `A, B, C`, whose DCG is `7 + 1/log2(3) + 1/log2(4)`. Their ratio is about **0.861**.

Here is a compact version of the calculation behind the table. The function assumes unique document IDs, known
nonnegative grades and k greater than zero; the full file checks those conditions relevant to this example.

```python
import math

def score(ranking, grades, k):
    relevant = {doc for doc, grade in grades.items() if grade > 0}
    if not relevant:
        raise ValueError('No relevant documents: choose and document a policy')
    top = ranking[:k]
    recall = len(set(top) & relevant) / len(relevant)
    rr = next((1 / i for i, doc in enumerate(top, 1) if doc in relevant), 0.)
    dcg = sum((2**grades[doc] - 1) / math.log2(i + 1)
              for i, doc in enumerate(top, 1))
    ideal = sorted(grades.values(), reverse=True)[:k]
    idcg = sum((2**grade - 1) / math.log2(i + 1)
               for i, grade in enumerate(ideal, 1))
    return recall, rr, dcg / idcg

grades = {'A': 3, 'B': 1, 'C': 1, 'X': 0, 'Y': 0}
print(tuple(round(v, 3) for v in score(['A', 'X', 'Y', 'B', 'C'], grades, 3)))
```

```text
(0.333, 1.0, 0.861)
```

**Specify the gain convention.** scikit-learn's
[`ndcg_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ndcg_score.html) uses the
values supplied as `y_true` directly as gains. To check the exponential-gain calculation above, pass
`2**grades - 1`, not the raw grades. The downloadable script does that for every row, using distinct predicted
scores to avoid ties, and all four comparisons pass. LightGBM's default
[`label_gain`](https://lightgbm.readthedocs.io/en/stable/Parameters.html#label_gain) follows the exponential pattern;
the [LambdaRank walkthrough](/blog/2026-10-12-lightgbm_lambdarank/) shows why mixing conventions changes a report.
References checked 2026-10-30.

## Decide the evaluation policy before comparing retrievers

Keep k, the query set, the relevance threshold, the gain function and the judged corpus fixed. Decide how to
handle duplicate chunks, tied scores and queries with no known relevant documents. This example raises for
the last case; other evaluations exclude those queries or assign zero, changing the reported average.

Incomplete judgments also limit the meaning of recall: unjudged is not automatically irrelevant. A larger
unjudged corpus does not magically supply the missing denominator. Evaluate against the labels you have and
state their coverage.

For a first-stage RAG retriever, I would inspect recall at the number of candidates actually passed downstream.
For the final ordering, I would add MRR if one useful result is enough, or NDCG if relevance is graded. None
of these metrics says whether the language model's final answer is supported by those documents.

Sentence Transformers' [retrieval evaluator](https://www.sbert.net/docs/package_reference/sentence_transformer/evaluation.html#informationretrievalevaluator)
reports these metric families too. The cover uses the first two constructed rankings, not model predictions;
the [renderer](/assets/code/tue_wed_figures/render_figures.py) reads their recorded scores.
