---
layout: post
title: "What does temperature do in an LLM? Temperature, top-k and top-p explained"
description: >
  Temperature reshapes the next-token probabilities; top-k and top-p cut off unlikely tokens. Tested on a small open
  model: at temperature 2.0 the output turned into noise, and top-k fixed it while top-p did not.
image: /assets/img/tips/llm_temperature/cover.png
sitemap:
  changefreq: daily
  priority: 1.0
---

# What does temperature do in an LLM? Temperature, top-k and top-p explained

**TL;DR** — An LLM picks each next token by sampling from a probability distribution. **Temperature** sharpens (below 1)
or flattens (above 1) that distribution. **Top-k** keeps only the k most likely tokens; **top-p** keeps the smallest set
whose probabilities add up to p. Low temperature gives safe, repetitive text; high temperature gives variety, and past a point, noise.

_Tested on 2026-09-28 with the open model Qwen2.5-0.5B-Instruct (transformers 5.17.0, PyTorch 2.14.0), running locally with no API key.
The model ran in its default bfloat16 precision; probabilities were computed in float32. A small model's text is weaker than a commercial model's,
but the sampling settings work the same way._

## Key points

- **The model outputs a score (logit) for every token** in its vocabulary, 151,936 of them for this model.
- **Temperature divides those scores before softmax.** Below 1, the top token gets more probability; above 1, probability spreads to unlikely tokens.
- **Top-k** cuts the candidates to a fixed number. **Top-p** (nucleus sampling) cuts them to a probability budget.
- **Order matters.** In Hugging Face transformers, temperature is applied first, then top-k, then top-p.
- **For factual or code tasks, use low temperature.** For brainstorming, use a moderate one, around 0.7–1.0, with top-p or top-k.

## How does temperature change the probabilities?

Here are the real next-token probabilities after "My favorite programming language is" at three temperatures:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

name = "Qwen/Qwen2.5-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name)
model.eval()

prompt = "My favorite programming language is"
ids = tok(prompt, return_tensors="pt").input_ids
with torch.no_grad():
    logits = model(ids).logits[0, -1].float()    # scores for every possible next token (float32)

top = torch.topk(logits, 6).indices
print(f"prompt: {prompt!r}")
print(f"{'next token':<14}" + "".join(f"{f'T={t}':>9}" for t in [0.3, 1.0, 2.0]))
probs = {t: torch.softmax(logits / t, dim=-1) for t in [0.3, 1.0, 2.0]}
for i in top:
    print(f"{tok.decode(i)!r:<14}" + "".join(f"{probs[t][i].item():>9.3f}" for t in [0.3, 1.0, 2.0]))
```

```
prompt: 'My favorite programming language is'
next token        T=0.3    T=1.0    T=2.0
' Python'         0.854    0.297    0.017
' C'              0.131    0.169    0.013
' Java'           0.013    0.085    0.009
' JavaScript'     0.000    0.029    0.005
' Ruby'           0.000    0.028    0.005
' not'            0.000    0.026    0.005
```

![Grouped bar chart of next-token probabilities after "My favorite programming language is": Python has 0.85 at temperature 0.3, 0.30 at 1.0 and 0.02 at 2.0; C, Java, JavaScript, Ruby and "not" all shrink toward 0.01 at temperature 2.0](/assets/img/tips/llm_temperature/next_token_probs.png)

At temperature 0.3, " Python" gets 85% of the probability. At 2.0 it gets 1.7%, and the six most likely tokens together get only 5.4%.
The rest is spread across tens of thousands of unlikely tokens:

```python
# continues from the code above (logits)
print(f"vocabulary size: {logits.numel():,}")
print(f"{'T':>4}{'top-6 mass':>12}{'tokens for top-p 0.9':>22}")
for t in [0.3, 1.0, 2.0]:
    p = torch.sort(torch.softmax(logits / t, dim=-1), descending=True).values
    n90 = int((torch.cumsum(p, 0) < 0.9).sum()) + 1
    print(f"{t:>4}{p[:6].sum().item():>12.3f}{n90:>22,}")
```

```
vocabulary size: 151,936
   T  top-6 mass  tokens for top-p 0.9
 0.3       1.000                     2
 1.0       0.635                    64
 2.0       0.054                43,942
```

To cover 90% of the probability, you need 2 tokens at temperature 0.3, 64 at 1.0, and 43,942 at 2.0.

## What does the generated text look like?

Three samples per temperature, each with a fixed random seed:

```python
# continues from the code above (model, tok, ids)
for t in [0.3, 1.0, 2.0]:
    print(f"--- temperature {t}")
    for seed in range(3):
        torch.manual_seed(seed)
        out = model.generate(ids, do_sample=True, temperature=t, top_k=0, top_p=1.0,
                             max_new_tokens=12, pad_token_id=tok.eos_token_id)
        print("  ", repr(tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)))
```

```
--- temperature 0.3
   ' Python. I am learning to code and have been trying to'
   ' Python. What are some of the benefits and drawbacks of using'
   " C++. I'm trying to learn Python. Can you provide"
--- temperature 1.0
   " Python. Let's say I have two lists of integers:\n\n"
   ':\nrunning\n\nThis justifies why we use "running"'
   ' C#. This language reacts promptly with Linq queries, so'
--- temperature 2.0
   ' ZAAA.\\憩 שנית consumptionkubectl品尝pellier啦 CFRAs'
   ':\nrunning South Hercules Üniversitesi \ndiscussion allowed;\nlongitude 필 관'
   ' Dart弨 recorder reacts bc subtotal MSR序列ues对 yüz'
```

`top_k=0, top_p=1.0` turns both filters off, so only temperature is at work. At 2.0, the model samples from the long tail and mixes random words from many languages.

## Do top-k and top-p fix a high temperature?

Top-k did; top-p did not:

```python
# continues from the code above (model, tok, ids)
settings = {"T=2.0, top_k=20": dict(temperature=2.0, top_k=20, top_p=1.0),
            "T=2.0, top_p=0.9": dict(temperature=2.0, top_k=0, top_p=0.9)}
for label, kw in settings.items():
    print(f"--- {label}")
    for seed in range(3):
        torch.manual_seed(seed)
        out = model.generate(ids, do_sample=True, max_new_tokens=12, pad_token_id=tok.eos_token_id, **kw)
        print("  ", repr(tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)))
```

```
--- T=2.0, top_k=20
   ' Python. My colleagues have said there is one key reason their'
   ' Go. But now with new tools for the latest releases like'
   ' C#. This year, more and more businesses adopt Java in'
--- T=2.0, top_p=0.9
   ' ZAAA.\\憩民愈趋品尝各种各样啦 CFRAs'
   ':\nrunning South Hercules Üniversitesi \ndiscussion allowed;\nlongitude是多少显示'
   ' Dart– pirate recorder reacts promptly)], Fury tidues rapid yüz'
```

The reason is the order. transformers applies temperature first, then top-k, then top-p (we checked its logits processor list:
`TemperatureLogitsWarper`, `TopKLogitsWarper`, `TopPLogitsWarper`). After temperature 2.0 has flattened the distribution,
"the top 90%" still contains about 44,000 tokens, so top-p removes almost nothing. Top-k=20 keeps exactly 20, no matter how flat the distribution is.

## Which settings should you use?

| Task | Temperature | Filter | Why |
|---|---|---|---|
| Extracting facts, classification, code | 0–0.3 | — | You want the single most likely answer, every time |
| General chat, explanations | 0.7–1.0 | top-p 0.9 or top-k 40 | Natural variety without drifting |
| Brainstorming, creative writing | 1.0–1.2 | top-k or top-p | More variety; check the output |
| Anything | above ~1.5 | — | Tends toward noise, as shown above |

Different APIs use different defaults and may apply these settings in a different order, so test on your own prompts.

## Sources

- [Generation strategies](https://huggingface.co/docs/transformers/generation_strategies) — Hugging Face transformers docs, checked 2026-09-28
- [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) — model card

Related: [What is a token, and why do LLMs count tokens instead of words?](/tips/2026-09-29-what_is_a_token/)
