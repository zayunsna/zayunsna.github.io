---
layout: post
title: "How do you evaluate LLM answers? Exact match vs key facts vs LLM-as-judge"
description: >
  We graded 12 answers from a small RAG system three ways and compared each grader with a human label. Exact match agreed on
  5 of 12, a key-fact check on 11 of 12, and a small LLM judge on 8 of 12, even though its total score looked right.
image: /assets/img/tips/evaluate_llm_answers/cover.png
sitemap:
  changefreq: daily
  priority: 1.0
---

# How do you evaluate LLM answers? Exact match vs key facts vs LLM-as-judge

**TL;DR** — Build a small test set of questions with known answers, grade every answer automatically, and check the grader against
your own judgment on a sample. In our test, **exact match** was far too strict (it matched the human verdict on 5 of 12 answers),
a **key-fact check** matched 11 of 12, and an **LLM judge** matched only 8 of 12, even though its total score happened to equal the human total.

_Tested on 2026-09-28 with the RAG system from our [RAG post](/tips/2026-10-05-rag_in_50_lines/) (Qwen2.5-0.5B-Instruct and all-MiniLM-L6-v2, local, no API key).
The judge was the same 0.5B model; larger judge models do better, but the lesson about checking them still applies._

## Key points

- **You need a test set:** questions, reference answers, and the fact a correct answer must contain. Include questions that should get "I don't know".
- **Exact match** fails on natural sentences: "Nordwind Analytics was founded in 2019." is correct but doesn't equal "2019".
- **A key-fact check** (does the answer contain "2019"?) is cheap and was the best grader here, but it can't tell complete from incomplete.
- **An LLM judge** reads the answer like a person would, but it makes its own mistakes. Its total can look right while individual verdicts are wrong.
- **Look at failures by stage:** one wrong answer here came from the generator, not the retriever.

## How did we test it?

Twelve questions about the fictional handbook: ten answered by it, two not. We ran the RAG system with the "say I don't know" rule, graded each answer three ways,
and then labeled each answer by hand after reading it:

```python
import runpy, contextlib, io, re

with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    rag = runpy.run_path("a4_rag.py")             # the RAG system from the RAG post
retrieve, ask, tok, llm = rag["retrieve"], rag["ask"], rag["tok"], rag["llm"]
rule = " If the answer is not in the context, say: I don't know."

# A small test set: question, reference answer, and the key fact a correct answer must contain
tests = [
    ("How many vacation days do employees get?", "28 days", "28"),
    ("Where are the company offices?", "Leipzig and Porto", "porto"),
    ("When was the company founded?", "2019", "2019"),
    ("Which day must everyone be in the office?", "Wednesday", "wednesday"),
    ("What is the GPU server called?", "Kestrel", "kestrel"),
    ("How many GPUs does the server have?", "four A100 cards", "four"),
    ("Where do I book a GPU job longer than 12 hours?", "the #gpu-queue channel", "gpu-queue"),
    ("When are expense reports due?", "by the 5th of the following month", "5th"),
    ("Who cleans the coffee machine?", "the facilities team", "facilities"),
    ("Where is the company retreat held?", "in the Harz mountains", "harz"),
    ("What is the CEO's name?", "I don't know", "don't know"),            # not in the handbook
    ("What is the office Wi-Fi password?", "I don't know", "don't know"),  # not in the handbook
]

def normalize(s):
    return re.sub(r"[^a-z0-9# ]", "", s.lower()).strip()

def judge(question, reference, answer):
    """Ask the same small model to grade the answer (LLM-as-judge)."""
    prompt = (f"Question: {question}\nReference answer: {reference}\nCandidate answer: {answer}\n"
              "Is the candidate answer correct according to the reference? Reply with yes or no only.")
    ids = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True,
                                  return_tensors="pt", return_dict=True)
    out = llm.generate(**ids, max_new_tokens=3, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True).strip().lower().startswith("yes")

rows = []
for question, reference, key in tests:
    answer = ask(question, retrieve(question), rule)
    rows.append({"q": question, "ref": reference, "answer": answer,
                 "exact": normalize(answer) == normalize(reference),
                 "contains": key in answer.lower(),
                 "judge": judge(question, reference, answer)})

for r in rows:
    print(f"{r['q']:<48} | {r['answer'][:75]}")

# Labels from reading every answer above (True = correct and complete)
human = [True, False, True, True, True, True, False, True, True, True, True, True]
print()
for metric in ["exact", "contains", "judge"]:
    scored = sum(r[metric] for r in rows)
    agree = sum(r[metric] == h for r, h in zip(rows, human))
    print(f"{metric:<9} says {scored:>2}/12 correct | matches the human label on {agree:>2}/12")
print(f"{'human':<9} says {sum(human):>2}/12 correct")
print("\nwhere the judge disagreed with the human label:")
for r, h in zip(rows, human):
    if r["judge"] != h:
        print(f"  judge={'yes' if r['judge'] else 'no ':<3} human={'correct' if h else 'wrong  '} | {r['q']} -> {r['answer'][:60]}")
```

```
How many vacation days do employees get?         | The given text states that employees get 28 days of paid vacation per year,
Where are the company offices?                   | Porto
When was the company founded?                    | Nordwind Analytics was founded in 2019.
Which day must everyone be in the office?        | Wednesdays.
What is the GPU server called?                   | Kestrel
How many GPUs does the server have?              | The server has four A100 GPUs.
Where do I book a GPU job longer than 12 hours?  | I don't know.
When are expense reports due?                    | The expense report is due by the 5th of the following month.
Who cleans the coffee machine?                   | The coffee machine on the second floor is cleaned every Friday by the facil
Where is the company retreat held?               | The annual company retreat takes place in the Harz mountains in September.
What is the CEO's name?                          | I don't know.
What is the office Wi-Fi password?               | I don't know.

exact     says  3/12 correct | matches the human label on  5/12
contains  says 11/12 correct | matches the human label on 11/12
judge     says 10/12 correct | matches the human label on  8/12
human     says 10/12 correct

where the judge disagreed with the human label:
  judge=yes human=wrong   | Where are the company offices? -> Porto
  judge=yes human=wrong   | Where do I book a GPU job longer than 12 hours? -> I don't know.
  judge=no  human=correct | What is the CEO's name? -> I don't know.
  judge=no  human=correct | What is the office Wi-Fi password? -> I don't know.
```

Two answers were wrong by our reading: "Porto" leaves out Leipzig, and "I don't know" for the GPU-queue question ignores a fact that is in the handbook.

![Bar chart of three graders on 12 answers: exact match marks 3 correct and agrees with the human label on 5; the key-fact check marks 11 and agrees on 11; the LLM judge marks 10 and agrees on 8; the human reference marks 10](/assets/img/tips/evaluate_llm_answers/grader_agreement.png)

## What did each grader get wrong?

| Grader | Verdicts matching the human | Typical failure |
|---|---|---|
| Exact match | 5 / 12 | Rejects correct full-sentence answers like "The server has four A100 GPUs." |
| Contains key fact | 11 / 12 | Accepted "Porto", which is only half of "Leipzig and Porto" |
| LLM judge (same 0.5B model) | 8 / 12 | Got every "I don't know" backwards: accepted it where the handbook had the answer, rejected it where "I don't know" was the reference |

The judge's total (10/12) equals the human total, so a single score would have looked perfect. Only the row-by-row comparison shows that it got four verdicts wrong, which happened to cancel out.

## Was it a retrieval or a generation error?

For the GPU-queue question, the retriever returned the right sentence first:

```python
# continues from the code above (retrieve)
q = "Where do I book a GPU job longer than 12 hours?"
print("retrieved for the failed question:"); [print("  ", c) for c in retrieve(q)]
```

```
retrieved for the failed question:
   GPU jobs longer than 12 hours must be booked in the #gpu-queue channel.
   The data team uses a shared GPU server called 'Kestrel' with four A100 cards.
```

The generator still said "I don't know". So the fix belongs in the prompt or the model, not in chunking or embeddings. Logging the retrieved chunks next to each answer makes this visible in seconds.

## How should you evaluate your own system?

| Step | Tip |
|---|---|
| Build a test set | 20–100 real questions with reference answers and the key fact each answer needs; include unanswerable ones |
| Grade automatically | Start with key-fact checks; add an LLM judge for open-ended answers |
| Check the grader | Label a sample by hand and measure agreement, row by row, not just totals |
| Split by stage | Log retrieved chunks, so you can tell retrieval misses from generation misses |
| Re-run on every change | Chunk size, prompt, model and temperature changes can all move the score |

## Sources

- [Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685) — Zheng et al., 2023, on the strengths and biases of LLM judges

Related: [Why is 99% accuracy useless? Precision, recall and F1 explained](/tips/2026-09-30-accuracy_is_misleading/) · [How should you chunk documents for RAG?](/tips/2026-10-08-rag_chunking/)
