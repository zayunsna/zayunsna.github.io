---
layout: post
title: "How should you chunk documents for RAG? A chunk-size experiment"
description: >
  We split a fictional handbook six ways and measured how often retrieval found the chunk with the answer. One-sentence chunks
  dropped from 96% to 33% once the text used pronouns; paragraph chunks stayed at 100% in both cases.
image: /assets/img/tips/rag_chunking/cover.png
sitemap:
  changefreq: daily
  priority: 1.0
---

# How should you chunk documents for RAG? A chunk-size experiment

**TL;DR** — Split documents along their natural boundaries, such as paragraphs or sections, so that each chunk makes sense on its own.
In our test, one-sentence chunks worked (96%) only while every sentence repeated the team name. With natural writing ("It is led by…"),
they found the right chunk just 33% of the time. Paragraph chunks found it 100% of the time in both versions.

_Tested on 2026-09-28 with sentence-transformers 6.1.0 (`all-MiniLM-L6-v2`) and the Qwen2.5 tokenizer for token counts, locally, no API key.
The handbook and questions are fictional and small (8 teams, 24 questions), so treat the numbers as an illustration._

## Key points

- **Chunking** is how you split documents before embedding them. Retrieval can only return whole chunks.
- **A chunk must carry its own context.** "It is led by Ines Moreau" means nothing without the team name.
- **Chunks that ignore document structure mix topics.** Three-sentence windows crossed paragraph boundaries and scored worst.
- **Bigger chunks find more but send more tokens.** 256-token chunks sent about 3× the tokens of paragraph chunks and still found less.
- **Test your chunking** with a few questions whose answers you know, as below.

## How did we test chunking?

Eight teams, one paragraph each, with four facts and three general sentences. For each team we ask three questions (floor, lead, meeting day)
and check whether the top-ranked chunk contains the answer (**hit@1**) or one of the top three does (**hit@3**). We also count how many tokens
the top three chunks would add to the prompt. `STYLE` switches between two ways of writing the same facts:

```python
import os, re
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

# A fictional handbook: 8 teams, each with a paragraph of facts and general sentences
teams = {
    "finance":    ("4th floor", "Ines Moreau", "Tuesday", "#fin-ops"),
    "marketing":  ("2nd floor", "Tom Becker", "Monday", "#mkt-daily"),
    "data":       ("5th floor", "Aiko Tanaka", "Thursday", "#data-help"),
    "legal":      ("6th floor", "Rui Almeida", "Wednesday", "#legal-desk"),
    "sales":      ("1st floor", "Nora Lind", "Friday", "#sales-wins"),
    "security":   ("basement", "Omar Haddad", "Monday", "#sec-alerts"),
    "hr":         ("3rd floor", "Lena Vogel", "Thursday", "#people-team"),
    "support":    ("ground floor", "Mateo Ruiz", "Wednesday", "#support-queue"),
}
general = ["The team shares its roadmap at the start of each quarter.",
           "New members receive a laptop and an access badge on day one.",
           "Holiday plans should be shared with the team two weeks ahead."]
STYLE = os.environ.get("STYLE", "explicit")
paragraphs = []
for team, (floor, lead, day, channel) in teams.items():
    if STYLE == "explicit":   # every sentence names the team
        facts = [f"The {team} team sits on the {floor}.", f"The {team} team is led by {lead}.",
                 f"The {team} team holds its weekly meeting every {day}.", f"Questions for the {team} team go to the {channel} channel."]
    else:                     # natural writing: only the first sentence names the team
        facts = [f"The {team} team sits on the {floor}.", f"It is led by {lead}.",
                 f"Its weekly meeting is every {day}.", f"Questions go to the {channel} channel."]
    paragraphs.append(" ".join([facts[0], general[0], facts[1], facts[2], general[1], facts[3], general[2]]))
document = "\n\n".join(paragraphs)

questions = []
for team, (floor, lead, day, channel) in teams.items():
    questions += [(f"Which floor is the {team} team on?", floor), (f"Who leads the {team} team?", lead),
                  (f"On which day does the {team} team meet?", day)]

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
sentences = re.split(r"(?<=\.)\s+", document.replace("\n\n", " "))

def by_sentences(n):          # n sentences per chunk
    return [" ".join(sentences[i:i + n]) for i in range(0, len(sentences), n)]

def by_tokens(size, overlap):  # fixed token windows with overlap
    ids, step = tok.encode(document), size - overlap
    return [tok.decode(ids[i:i + size]) for i in range(0, len(ids), step)]

strategies = {
    "1 sentence": by_sentences(1),
    "3 sentences": by_sentences(3),
    "1 paragraph": paragraphs,
    "64 tokens, 16 overlap": by_tokens(64, 16),
    "256 tokens, 32 overlap": by_tokens(256, 32),
    "whole document": [document],
}

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
q_vecs = model.encode([q for q, _ in questions], normalize_embeddings=True)
print(f"{len(questions)} questions, document of {len(tok.encode(document))} tokens")
print(f"{'strategy':<24}{'chunks':>7}{'hit@1':>7}{'hit@3':>7}{'tokens sent (top 3)':>21}")
results = {}
for name, chunks in strategies.items():
    c_vecs = model.encode(chunks, normalize_embeddings=True)
    ranks = np.argsort(-(q_vecs @ c_vecs.T), axis=1)
    hit1 = np.mean([answer in chunks[ranks[i, 0]] for i, (_, answer) in enumerate(questions)])
    hit3 = np.mean([any(answer in chunks[j] for j in ranks[i, :3]) for i, (_, answer) in enumerate(questions)])
    sent = np.mean([sum(len(tok.encode(chunks[j])) for j in ranks[i, :3]) for i in range(len(questions))])
    results[name] = (len(chunks), hit1, hit3, sent)
    print(f"{name:<24}{len(chunks):>7}{hit1:>7.0%}{hit3:>7.0%}{sent:>21.0f}")
```

Every sentence names the team (`STYLE=explicit`):

```
24 questions, document of 653 tokens
strategy                 chunks  hit@1  hit@3  tokens sent (top 3)
1 sentence                   56    96%   100%                   34
3 sentences                  19    75%    88%                  102
1 paragraph                   8   100%   100%                  244
64 tokens, 16 overlap        14    92%   100%                  189
256 tokens, 32 overlap        3   100%   100%                  717
whole document                1   100%   100%                  653
```

Natural writing, where only the first sentence names the team (`STYLE=natural`):

```
24 questions, document of 581 tokens
strategy                 chunks  hit@1  hit@3  tokens sent (top 3)
1 sentence                   56    33%    46%                   29
3 sentences                  19    38%    58%                   94
1 paragraph                   8   100%   100%                  217
64 tokens, 16 overlap        13    71%    96%                  180
256 tokens, 32 overlap        3    83%   100%                  645
whole document                1   100%   100%                  581
```

![Grouped bar chart of hit@1 for six chunking strategies in two writing styles: one sentence 96% vs 33%, three sentences 75% vs 38%, one paragraph 100% in both, 64-token windows 92% vs 71%, 256-token windows 100% vs 83%, whole document 100% in both](/assets/img/tips/rag_chunking/chunking_hit_rate.png)

## What do the results mean?

- **One-sentence chunks depend on how the text is written.** With "It is led by…", the chunk holding the answer doesn't mention the team, so the question can't match it: 96% fell to 33%.
- **Paragraph chunks were the only strategy at 100% in both styles.** Each paragraph is about one team, so every fact stays next to the team name.
- **Three-sentence windows scored low in both styles** (75% and 38%). Fixed windows cut across paragraphs and mix two teams in one chunk.
- **Token windows landed in between.** 64-token chunks sent fewer tokens than paragraphs but found less; 256-token chunks sent 645 tokens and still missed 17%.
- **"Whole document" scores 100% only because it sends everything.** With real document collections that don't fit in the prompt, this isn't an option.

## How should you chunk your own documents?

| Document type | Good chunk boundary |
|---|---|
| Handbooks, wikis, docs with headings | One section or paragraph per chunk; prepend the heading |
| Long unstructured text | Token windows (for example 200–500 tokens) with some overlap |
| FAQs | One question-and-answer pair per chunk |
| Tables and records | One row or record per chunk, written out with its column names |
| Code | One function or class per chunk |

A cheap fix for context-dependent chunks is to prepend the title or heading to every chunk, for example "Finance team: It is led by Ines Moreau."
We tried it on the natural-writing handbook: one-sentence chunks with the team name prepended went from 33% to 92% hit@1.
Whatever you choose, keep a small set of questions with known answers and re-run a check like the one above after every change.

## Sources

- [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) — embedding model, checked 2026-09-28

Related: [RAG explained in 50 lines of Python](/tips/2026-10-05-rag_in_50_lines/) · [What is a context window, and what happens when the input is too long?](/tips/2026-10-07-context_window/)
