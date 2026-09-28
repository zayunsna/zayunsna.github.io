---
layout: post
title: "What are embeddings? Semantic search with cosine similarity in Python"
description: >
  An embedding turns text into a vector so that similar meanings land close together. A 30-line example with a free
  local model finds "My laptop gets very hot" for "Why does my notebook overheat?", with no shared words.
image: /assets/img/tips/embeddings_semantic_search/cover.png
sitemap:
  changefreq: daily
  priority: 1.0
---

# What are embeddings? Semantic search with cosine similarity in Python

**TL;DR** — An embedding model turns a piece of text into a list of numbers (a vector) so that texts with similar meaning
get similar vectors. Comparing vectors with **cosine similarity** lets you search by meaning instead of by matching words.
In our test, keyword search ranked a sourdough sentence first for a laptop question; embedding search found the right answers.

_Tested on 2026-09-28 with sentence-transformers 6.1.0 and the free `all-MiniLM-L6-v2` model, running locally (no API key)._

## Key points

- **An embedding** is a fixed-length vector for a piece of text. This model produces 384 numbers per sentence.
- **Cosine similarity** measures the angle between two vectors: 1 means the same direction, 0 means unrelated.
- **Semantic search** embeds the query and all documents, then returns the documents with the highest similarity.
- **It works without shared words:** "notebook overheat" matched "laptop gets very hot".
- **Embeddings are the retrieval step of RAG** (retrieval-augmented generation), which feeds found text to an LLM.

## Why does keyword search fail?

Keyword search only sees words, not meaning. Here are 12 short documents and one question:

```python
import re
import numpy as np
from sentence_transformers import SentenceTransformer

docs = [
    "My laptop gets very hot when I play games.",
    "How to reduce CPU temperature on a notebook computer.",
    "The fan in my computer is making a loud noise.",
    "Best budget laptops for students in college.",
    "How to bake sourdough bread at home.",
    "My sourdough starter smells like vinegar.",
    "Tips for running your first marathon.",
    "How to prevent knee pain when jogging.",
    "Python list comprehension examples.",
    "How to speed up a slow Python loop.",
    "The stock market fell sharply today.",
    "How interest rates affect bond prices.",
]
query = "Why does my notebook overheat?"

# 1) keyword search: count shared words
words = lambda s: set(re.findall(r"[a-z]+", s.lower()))
kw = [len(words(query) & words(d)) for d in docs]

# 2) embedding search: cosine similarity between vectors
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
emb = model.encode(docs, normalize_embeddings=True)          # unit-length vectors
q = model.encode([query], normalize_embeddings=True)[0]
cos = emb @ q                                                 # dot product of unit vectors = cosine similarity

print("embedding size:", emb.shape[1])
print("query:", query)
print("\nkeyword search (shared words):")
for i in np.argsort(kw)[::-1][:3]:
    print(f"  {kw[i]}  {docs[i]}")
print("\nembedding search (cosine similarity):")
for i in np.argsort(cos)[::-1][:3]:
    print(f"  {cos[i]:.2f}  {docs[i]}")
```

```
embedding size: 384
query: Why does my notebook overheat?
keyword search (shared words):
  1  My sourdough starter smells like vinegar.
  1  The fan in my computer is making a loud noise.
  1  How to reduce CPU temperature on a notebook computer.

embedding search (cosine similarity):
  0.65  How to reduce CPU temperature on a notebook computer.
  0.64  My laptop gets very hot when I play games.
  0.41  The fan in my computer is making a loud noise.
```

- **Keyword search** found three documents that share one word each, so they tie. The sourdough sentence is among them only because it shares the word "my".
- **Embedding search** put the two relevant documents on top. "My laptop gets very hot when I play games" shares **no words** with the question, but its meaning is close (0.64).

## What does cosine similarity look like across topics?

Comparing all 12 documents with each other shows that sentences on the same topic have higher similarity:

![Heatmap of cosine similarity between 12 sentences: the three computer-overheating sentences score 0.34 to 0.54 with each other, and the sourdough, running, Python and finance pairs each score 0.21 to 0.39, while unrelated topics stay near 0](/assets/img/tips/embeddings_semantic_search/similarity_heatmap.png)

- The laptop-heat sentences (1–3) form a block with similarities of 0.34 to 0.54.
- Each two-sentence topic pair scores 0.21 to 0.39: bread (5–6), running (7–8), Python (9–10), finance (11–12).
- Unrelated pairs sit near 0, for example the laptop sentence and the bread sentence at −0.08.

Scores are relative: 0.4 can be a strong match with one model and a weak one with another. Compare rankings, not raw numbers across models.

## How do you use this in practice?

| Step | Tip |
|---|---|
| Choose a model | Start with a small general model like `all-MiniLM-L6-v2`. It is English-focused, so pick a multilingual model for other languages |
| Embed documents once | Store the vectors; only the query needs embedding at search time |
| Normalize | With `normalize_embeddings=True`, a plain dot product is the cosine similarity |
| Split long documents | Embed paragraphs or chunks, not whole books. One vector can't hold everything |
| Scale up | For millions of vectors, use a vector index (for example FAISS or a vector database) instead of a full matrix product |

## Sources

- [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) — model card, checked 2026-09-28
- [Sentence Transformers documentation](https://sbert.net/)

Related: [What is a token, and why do LLMs count tokens instead of words?](/tips/2026-09-29-what_is_a_token/)
