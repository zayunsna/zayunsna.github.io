---
layout: post
title: "Hybrid search for RAG: BM25 plus embeddings, measured on 1,000 questions"
description: >
  Keyword search (BM25) and embedding search miss different questions. Merging their result lists with reciprocal
  rank fusion cut the questions whose paragraph was missing from the top 20 from 45 to 16 out of 1,000, and raised
  top-1 accuracy after reranking from 85.2% to 87.3%. BM25 alone was better at rank 1 than embeddings on this test.
image: /assets/img/tips/hybrid_search_bm25/cover.png
sitemap:
  changefreq: daily
  priority: 1.0
---

# Hybrid search for RAG: BM25 plus embeddings, measured on 1,000 questions

Embedding search finds text with the same meaning. Keyword search finds text with the same words. Hybrid search
runs both and merges the two result lists. I measured all three on the same test as the
[reranking post](/tips/2026-10-29-rag_reranking/): 1,000 SQuAD questions, 2,067 paragraphs, and one right
paragraph per question.

| Method | Right paragraph ranked 1st | In top 5 | In top 20 | In top 100 |
|---|---|---|---|---|
| BM25 (keywords) | **74.5%** | 90.7% | 95.8% | 98.7% |
| Embeddings (all-MiniLM-L6-v2) | 63.6% | 86.4% | 95.5% | 99.0% |
| Hybrid (reciprocal rank fusion) | 73.4% | **92.5%** | **98.4%** | **99.9%** |

Hybrid was not the best at rank 1. It was the best at putting the right paragraph *somewhere* in the list. A RAG
system usually sends several results to the model or to a reranker, so the deeper columns matter more.

_Tested on 2026-10-29 with rank-bm25 0.2.2, sentence-transformers 6.2.0 and torch 2.14.1 (Python 3.12.13) on an
Apple M5 laptop. Script and full output: [hybrid.py](/assets/code/hybrid_search_bm25/hybrid.py),
[hybrid_out_mps.txt](/assets/code/hybrid_search_bm25/hybrid_out_mps.txt)._

## What BM25 is

BM25 scores a document by the query words it contains. A rare word (a name, an error code) counts much more than a
common word, and repeating a word gives less extra credit each time. It needs no model and no GPU. With the
[rank-bm25](https://github.com/dorianbrown/rank_bm25) package (Apache-2.0), the index takes a few lines:

```python
import re

from rank_bm25 import BM25Okapi

def tokenize(text):
    return re.findall(r"\w+", text.lower())

bm25 = BM25Okapi([tokenize(p) for p in paragraphs])
scores = bm25.get_scores(tokenize(question))  # one score per paragraph
```

The embedding side is the same bi-encoder as in the reranking post: `sentence-transformers/all-MiniLM-L6-v2`
(Apache-2.0), cosine similarity between the question vector and each paragraph vector. If embeddings are new to
you, start with [What are embeddings?](/tips/2026-10-01-embeddings_semantic_search/).

## The two methods miss different questions

At rank 1, the two methods were both right on 540 questions. BM25 alone was right on 205, embeddings alone on 96,
and neither on 159. Some of each (ranks from the run above):

| Question | BM25 rank | Embeddings rank |
|---|---|---|
| Why is giving a defiant speech sometimes more harmful for the individual? | 1 | 14 |
| What could anyone do to Martin Luther without legal consequence? | 1 | 6 |
| What year was the first class taught at the University of Chicago? | 12 | 1 |
| Which two treaties provided more formal institutions of the European Union? | 4 | 1 |

When the question repeats rare words from the paragraph ("defiant speech"), BM25 wins. When the paragraph says the
same thing in other words, embeddings win. Hybrid search uses this: a paragraph that one method misses can still
come in through the other.

## Merging the lists: reciprocal rank fusion

The two scores are on different scales (BM25 scores have no upper limit, cosine similarity is between -1 and 1), so
you can't add them directly. Reciprocal rank fusion (RRF) ignores the scores and uses only the positions:

```python
def rrf(*rankings, k=60):
    """Reciprocal rank fusion: each list adds 1 / (k + rank) to a document's score."""
    scores = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=scores.get, reverse=True)

hybrid = rrf(bm25_top_100, embedding_top_100)
```

A paragraph that is 3rd in one list and 50th in the other gets 1/63 + 1/110. One that is 2nd in one list and
missing from the other gets only 1/62. `k=60` is the value from the
[original RRF paper](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) (Cormack, Clarke and Büttcher,
SIGIR 2009). I did not tune it.

![Line chart of the share of 1,000 questions whose paragraph is in the results, for top 1, 5, 20 and 100. BM25: 74.5%, 90.7%, 95.8%, 98.7%. Embeddings: 63.6%, 86.4%, 95.5%, 99.0%. Hybrid: 73.4%, 92.5%, 98.4%, 99.9%](/assets/img/tips/hybrid_search_bm25/hybrid_hits.png)

## Hybrid search before a reranker

A reranker can only reorder the candidates it gets. In the reranking post, 45 of the 1,000 questions had no right
paragraph in the embedding top 20, so no reranker could fix them. With the hybrid top 20, that number fell to 16.

I ran the same cross-encoder (`cross-encoder/ms-marco-MiniLM-L6-v2`) on both top-20 lists:

| Candidates for the reranker | Right paragraph in top 20 | Ranked 1st after reranking |
|---|---|---|
| Embeddings top 20 | 95.5% | 85.2% |
| Hybrid top 20 | 98.4% | **87.3%** |

The reranker costs the same either way, because it scores 20 pairs. BM25 added about 1 ms per query here
(rank-bm25 is pure Python; a search engine with a BM25 index is faster).

## Limits of this test

- **SQuAD favors keywords.** People wrote the questions while reading the paragraph, so questions often reuse its
  words. That is why BM25 beat embeddings at rank 1. Questions from real users, written without seeing the
  document, may favor embeddings more. Measure on your own questions.
- The embedding model was trained on data that includes the SQuAD 2.0 training set (per its model card), so this
  test may also be easier for it than for your data.
- One corpus of 2,067 English paragraphs, one embedding model, plain lowercase word tokens with no stemming or
  stop-word removal, and RRF with the default `k=60`.
- SQuAD v1.1 is licensed CC BY-SA 4.0. The question texts above come from it.

Hybrid search is worth trying when your documents contain names, codes or exact terms that people type, or when you
feed a reranker. If you only show the single top result and your users paraphrase, compare it with embeddings alone
first.
