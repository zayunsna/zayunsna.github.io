---
layout: post
title: "Reranking for RAG: a cross-encoder lifted top-1 retrieval from 64% to 85%"
description: >
  Bi-encoder vs cross-encoder, measured on 1,000 SQuAD questions with two small local models. Re-scoring the top 20
  search results moved the right paragraph to first place 85% of the time instead of 64%, for 67 ms per query on a
  laptop GPU. Past 20 candidates the gain flattened while the cost kept rising.
image: /assets/img/tips/rag_reranking/cover.png
sitemap:
  changefreq: daily
  priority: 1.0
---

# Reranking for RAG: a cross-encoder lifted top-1 retrieval from 64% to 85%

A RAG system first **retrieves** the passages most similar to the question and then pastes them into the prompt
([RAG in 50 lines](/tips/2026-10-05-rag_in_50_lines/)). If the right passage is ranked 7th and you only paste the
top 3, the model never sees it. A **reranker** is a second, slower model that re-scores the top results before you
cut the list.

I measured how much it helps with two small free models on 1,000 real questions:

| Setup | Right paragraph ranked 1st | Right paragraph in top 3 | Time per query (M5 GPU) |
|---|---:|---:|---:|
| Bi-encoder only | 63.6% | 80.4% | 0.3 ms |
| + rerank top 10 | 83.7% | 91.6% | 31 ms |
| + rerank top 20 | 85.2% | 93.4% | 67 ms |
| + rerank top 100 | 86.6% | 95.3% | 519 ms |

Reranking the top 20 turned 64% into 85%. Going from 20 to 100 candidates added only 1.4 points for almost 8 times
the cost.

_Tested on 2026-10-27 with sentence-transformers 6.2.0, transformers 5.21.0 and torch 2.14.1 (Python 3.12.13) on an
Apple M5 laptop. Script and output: [rerank.py](/assets/code/rag_reranking/rerank.py),
[output](/assets/code/rag_reranking/rerank_out_mps.txt)._

## Bi-encoder vs cross-encoder

Both are small transformer models. The difference is **when the question and the passage meet**.

- A **bi-encoder** turns the question and each passage into vectors separately and compares the vectors
  ([embeddings explained](/tips/2026-10-01-embeddings_semantic_search/)). Passage vectors are computed once, ahead of
  time, so a search is just one question encoding plus a fast vector comparison. The model never reads the question
  and the passage together.
- A **cross-encoder** reads the question and one passage **as a single input** and outputs a relevance score. Every word
  of the question can attend to every word of the passage, which makes it more precise. Nothing can be precomputed,
  though: scoring 20 passages means running the model 20 times (in one batch).

That is why they are used together. The bi-encoder narrows thousands of passages down to a few dozen, and the
cross-encoder sorts those few dozen carefully.

```python
from sentence_transformers import CrossEncoder, SentenceTransformer

bi = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
cross = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")

doc_vecs = bi.encode(paragraphs, normalize_embeddings=True)  # once, offline

def search(question, k=20):
    # Stage 1: fast and approximate, compare vectors
    q_vec = bi.encode(question, normalize_embeddings=True)
    candidates = (doc_vecs @ q_vec).argsort()[::-1][:k]
    # Stage 2: slow and precise, read each (question, passage) pair
    scores = cross.predict([(question, paragraphs[i]) for i in candidates])
    return candidates[scores.argsort()[::-1]]
```

Both models are free, run locally, and are Apache-2.0 licensed
([bi-encoder](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2),
[cross-encoder](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2)).

## The test

- **Passages:** all 2,067 paragraphs of the [SQuAD v1.1](https://rajpurkar.github.io/SQuAD-explorer/) dev set
  (Wikipedia, CC BY-SA 4.0).
- **Questions:** 1,000 questions sampled at random (seed 0) from the 10,570 in the same set. Each question was written
  about one paragraph, so we know the right answer.
- **Score:** how often that paragraph ends up in first place (hit@1), and in the top 3. Time is measured per question
  on the M5's GPU (MPS) after a warm-up.

## What reranking fixed

Here is one question where the bi-encoder put the wrong paragraph first:

> **Q:** When did the UK formally subscribe to the Agreement on Social Policy?
>
> **Bi-encoder #1:** "The Social Charter was subsequently adopted in 1989 by 11 of the then 12 member states. The UK refused to sign the Social Charter…"
>
> **Correct paragraph (bi-encoder rank 2, cross-encoder rank 1):** "Following the election of the UK Labour Party to government in 1997, the UK formally subscribed to the Agreement on Social Policy…"

Both paragraphs are about the UK and the social policy agreement, so their vectors are close. Only the second one
answers *when the UK subscribed*. The cross-encoder reads the question and the paragraph together and catches that
difference. In another example, "Why is giving a defiant speech sometimes more harmful?", the right paragraph was
14th for the bi-encoder and 1st after reranking.

## How many candidates should you rerank?

![Line chart over rerank depth 5, 10, 20, 50 and 100: the share of questions whose correct paragraph is among the candidates rises from 86% to 99%, the share ranked first after reranking rises from 80% to 84%, 85%, 86% and 87%, while the bi-encoder alone stays at 64%. Time per query on the M5 GPU grows from 16 ms to 31, 67, 187 and 519 ms](/assets/img/tips/rag_reranking/rerank_depth.png)

Two lines matter here:

- **The dashed line is a ceiling.** The reranker only reorders what the bi-encoder gave it. With the top 20, 45 of the
  1,000 questions never had their paragraph among the candidates, so no reranker could fix them. A better reranker
  can't help these questions. A bigger `k` or a better first-stage model can.
- **The gap between the lines is what the cross-encoder still gets wrong.** At top 100, the right paragraph was there
  99% of the time, but the reranker put it first only 87% of the time.

The cost grows faster than `k`, because the batch gets bigger: 16 ms for 5 passages, 519 ms for 100. On the CPU of
the same laptop (4 threads), scoring 10, 20 and 50 SQuAD paragraphs took 77, 187 and 606 ms per question (median of 3
passes over 100 questions; [timing script](/assets/code/rag_reranking/cpu_timing.py)). In a chat app,
reranking 20 candidates on CPU adds about a fifth of a second.

A practical default from this test: **retrieve 20 to 50, rerank, keep the top 3 to 5 for the prompt.**

## Limits of this test

- **SQuAD is friendly to both models.** Questions reuse many words from their paragraph. On real user questions
  with vaguer wording, the numbers will be different, and should be measured on your own data.
- **The bi-encoder has seen this kind of data.** Its training data includes the SQuAD 2.0 training set (87,599
  pairs, per the [model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)). The dev set uses
  different Wikipedia articles, but the question style is familiar to it. The cross-encoder was trained on MS MARCO
  (web search queries).
- **2,067 passages is small.** With millions of passages, the first stage gets harder and the dashed ceiling drops,
  which makes the choice of `k` more important ([exact search vs HNSW](/tips/2026-10-19-vector_search_hnsw/)).
- **These are small models** (both MiniLM, 6 layers). Larger rerankers are usually more accurate and slower; I didn't
  test them here.
- One run, one sample of questions. Timings are from a single machine and change with passage length.

If retrieval is missing the right passage entirely, look at how the documents were split first
([RAG chunking](/tips/2026-10-08-rag_chunking/)). Reranking can't recover text that a bad chunk cut in half.
