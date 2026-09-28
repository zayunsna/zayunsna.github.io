---
layout: post
title: "What is a context window, and what happens when the input is too long?"
description: >
  A context window is the maximum number of tokens a model can read at once. We hid a code in long documents: a small local model
  found it at 8,000 tokens, but answering took 35x longer than at 570, and when truncation removed the code, it made one up.
image: /assets/img/tips/context_window/cover.png
sitemap:
  changefreq: daily
  priority: 1.0
---

# What is a context window, and what happens when the input is too long?

**TL;DR** — The **context window** is the maximum number of tokens a model can take in at once: your prompt, any documents, the chat history and the answer.
Longer inputs are slower and more expensive, and they grew faster than linearly in our test (570 tokens: 3 s, 8,069 tokens: 110 s).
If your input doesn't fit and gets truncated, the model doesn't know what it lost, and it may confidently invent the missing fact.

_Tested on 2026-09-28 with Qwen2.5-0.5B-Instruct (context window 32,768 tokens) and transformers 5.17.0, running on the CPU of an Apple M5, no API key._

## Key points

- **Everything counts toward the window:** system prompt, instructions, retrieved documents, previous messages and the generated answer.
- **Longer input costs more than proportionally.** 14× more tokens took 35× more time here.
- **When input is too long, something must go.** Where you truncate decides what the model can no longer see.
- **The model doesn't flag missing information.** With the key sentence removed, it answered "123456".
- **Put the question and key facts where they won't be truncated,** and send only what's relevant (that's what RAG is for).

## Can a model find one fact in a long document?

We hid one sentence, "The secret access code for the archive room is 7342.", in filler text of about 500, 2,000 and 8,000 tokens,
at the start, the middle and the end, and asked for the code:

```python
import time
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

name = "Qwen/Qwen2.5-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name).eval()
print("max_position_embeddings:", AutoConfig.from_pretrained(name).max_position_embeddings)

filler = [
    "The quarterly report covers sales in three regions.", "Weather conditions were mild for most of the week.",
    "The team reviewed the onboarding checklist again.", "Several customers asked about delivery times.",
    "The warehouse inventory was counted on Tuesday.", "Marketing prepared a draft for the spring campaign.",
    "The office printer needed new toner twice.", "A new supplier offered lower prices for packaging.",
]
needle = "The secret access code for the archive room is 7342."

def haystack(n_tokens, position):
    sents, total = [], 0
    while total < n_tokens:
        s = filler[len(sents) % len(filler)]
        sents.append(s); total += len(tok.encode(" " + s))
    sents.insert(int(len(sents) * position), needle)
    return " ".join(sents)

def ask(context):
    msgs = [{"role": "user", "content": context + "\n\nWhat is the secret access code for the archive room? Answer with the number only."}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=8, do_sample=False, pad_token_id=tok.eos_token_id)
    return ids["input_ids"].shape[1], tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()

if __name__ == "__main__":
    print(f"{'target':>7} {'prompt tokens':>14} {'needle at':>10} {'answer':>10} {'ok':>4} {'seconds':>8}")
    for n in [500, 2_000, 8_000]:
        for pos in [0.0, 0.5, 1.0]:
            t0 = time.perf_counter()
            n_prompt, ans = ask(haystack(n, pos))
            ok = "7342" in ans
            print(f"{n:>7} {n_prompt:>14,} {pos:>10.0%} {ans!r:>10} {'yes' if ok else 'no':>4} {time.perf_counter() - t0:>8.1f}")
```

```
max_position_embeddings: 32768
 target  prompt tokens  needle at     answer   ok  seconds
    500            570         0%     '7342'  yes      3.1
    500            570        50%     '7342'  yes      3.1
    500            570       100%     '7342'  yes      3.1
   2000          2,064         0%     '7342'  yes     14.6
   2000          2,064        50%     '7342'  yes     15.0
   2000          2,064       100%     '7342'  yes     15.1
   8000          8,069         0%     '7342'  yes    109.8
   8000          8,069        50%     '7342'  yes    111.2
   8000          8,069       100%     '7342'  yes    110.2
```

The model found the code in all nine cases. This is an easy test, though: the filler repeats, so the one unusual sentence stands out.
Research on harder tasks, such as [Lost in the Middle](https://arxiv.org/abs/2307.03172), found that models use information in the middle of long contexts less reliably. We did not reproduce that here.

## How does input length affect speed?

Time grew much faster than the prompt did:

![Line chart of seconds to answer versus prompt length: 3 s at 570 tokens, 6 s at 1,069, 15 s at 2,064, 38 s at 4,070 and 110 s at 8,069, far above the dashed line showing proportional growth, which would reach about 44 s](/assets/img/tips/context_window/time_vs_length.png)

| Prompt tokens | Seconds | vs. 570 tokens |
|---|---|---|
| 570 | 3.1 | 1× |
| 2,064 | 14.6 | 4.7× |
| 8,069 | 109.9 | 35× |

Going from 570 to 8,069 tokens is 14× more input but took 35× longer. Attention compares every token with every other token, so its cost grows with the square of the length.
Hosted APIs are much faster than a laptop CPU, but they bill every input token, so long prompts still cost more on every request.

## What happens when the input is truncated to fit?

Suppose the model only accepted 2,000 tokens and the document had 8,000. We kept either the first or the last 2,000 tokens:

```python
import runpy
ns = runpy.run_path("a9_needle.py", run_name="helpers")     # reuse model, tokenizer, haystack() and ask()
tok, haystack, ask = ns["tok"], ns["haystack"], ns["ask"]

limit = 2_000                                             # pretend the model only accepts 2,000 tokens
for pos in [0.0, 1.0]:
    doc = tok.encode(haystack(8_000, pos))
    for side, kept in [("keep the start", doc[:limit]), ("keep the end", doc[-limit:])]:
        n_prompt, ans = ask(tok.decode(kept))
        print(f"needle at {pos:>4.0%} | {side:<14} | needle kept: {'7342' in tok.decode(kept)!s:<5} | answer: {ans!r}")
```

```
needle at   0% | keep the start | needle kept: True  | answer: '7342'
needle at   0% | keep the end   | needle kept: False | answer: '123456'
needle at 100% | keep the start | needle kept: False | answer: '123456'
needle at 100% | keep the end   | needle kept: True  | answer: '7342'
```

Whenever the code was truncated away, the model answered **"123456"**, a made-up number, instead of saying it couldn't find it.

## How do you stay within the context window?

| Situation | What to do |
|---|---|
| Long documents | Retrieve only the relevant parts (RAG) instead of pasting everything |
| Long chats | Summarize old turns; keep the system prompt and latest messages |
| You must truncate | Remove the least important part, and keep the question at the end |
| You're near the limit | Count tokens first; remember the answer also needs room |
| Cost or latency matters | Shorter prompts are faster and cheaper on every call |

## Sources

- [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) — model card, checked 2026-09-28
- [Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172) — Liu et al., 2023

Related: [What is a token, and why do LLMs count tokens instead of words?](/tips/2026-09-29-what_is_a_token/) · [RAG explained in 50 lines of Python](/tips/2026-10-05-rag_in_50_lines/)
