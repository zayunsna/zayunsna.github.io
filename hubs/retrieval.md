---
layout: page
title: Retrieval and RAG
short: chunking, embeddings, BM25, reranking and thresholds, measured on the same question set
description: >
  Measured retrieval experiments for RAG: embeddings, chunking, HNSW, hybrid BM25, reranking and score thresholds.
permalink: /retrieval/
hub: retrieval
hub_order: 3
---

The reranking and hybrid search posts share one setup: 1,000 SQuAD v1.1 questions retrieved against 2,067 paragraphs,
so their numbers can be compared directly. On that set, embeddings alone put the right paragraph first 63.6% of the time, BM25 74.5%,
and a cross-encoder reranker over the top 20 lifted the embedding result to 85.2%.

SQuAD questions reuse words from their paragraphs, which favors BM25. Each post states its own limits next to the result.

{% include hub.html %}
