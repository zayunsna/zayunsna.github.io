---
layout: post
title: "Greedy decoding vs beam search: why the most likely next word can give a worse answer"
description: >
  Greedy decoding picks the most likely token at every step. We compared it with beam search on a local
  Qwen2.5-0.5B for 10 prompts: beam search found a more probable continuation in 8 of 10, and on one prompt greedy's
  locally best word led to a wrong answer. Higher probability was not always better text, either.
image: /assets/img/tips/greedy_vs_beam_search/cover.png
sitemap:
  changefreq: daily
  priority: 1.0
---

# Greedy decoding vs beam search: why the most likely next word can give a worse answer

When `do_sample=False`, a language model uses **greedy decoding**: at every step it takes the single most likely
next token and never looks back. That is a greedy algorithm in the exact sense of
[our greedy algorithms post](/blog/2026-10-21-greedy_visualized/), and it fails the same way: the best-looking
step now can lead to a worse result overall.

**Beam search** keeps the `n` best partial sequences at each step instead of one, and picks the most probable
complete sequence at the end. We compared both on a local model, scoring every 24-token continuation by its total
log-probability (higher, meaning closer to 0, is more likely according to the model).

| Prompt | Greedy | Beam 4 | Beam 8 |
|---|---:|---:|---:|
| The capital of France is | -17.65 | **-10.97** | -11.43 |
| In machine learning, overfitting means | -10.69 | **-10.38** | **-10.38** |
| To make a good cup of coffee, you should | -23.54 | -19.14 | **-17.79** |
| The main difference between a list and a tuple in Python is | -10.96 | **-9.32** | **-9.32** |
| Once upon a time, a small robot | -35.59 | **-19.83** | **-19.83** |
| The best way to learn a new programming language is | -22.54 | -23.10 | **-17.64** |
| A hash table stores data by | **-12.25** | -12.81 | **-12.25** |
| The weather today is | -19.13 | -13.36 | **-10.05** |
| My favorite thing about weekends is | -30.05 | -19.82 | **-19.53** |
| The first step in debugging a program is | -18.85 | -16.48 | **-12.68** |

Beam 4 beat greedy on 8 of 10 prompts. It lost twice, so beam search is not a guarantee either: it is still an
approximation, just a less short-sighted one.

_Tested on 2026-10-14 with Qwen2.5-0.5B-Instruct used as a plain text continuer (no chat template),
transformers 5.19.0 and PyTorch 2.14.1 on CPU, no API key._

## Where greedy goes wrong: one word at token 3

The list-versus-tuple prompt shows the failure clearly. Both methods agree on the first two tokens, " that a". At
token 3 the model gives " list" a log-probability of -0.59 and " tuple" -0.83. Greedy takes " list", the better
step. Beam search keeps both and continues.

![Cumulative log-probability over 24 tokens for greedy and beam 8: the lines split at token 3, where greedy takes "list" (-0.59) and beam search keeps "tuple" (-0.83); greedy ends at -10.96 and beam search at -9.32](/assets/img/tips/greedy_vs_beam_search/chart.png)

Where they ended up:

- **greedy**: " that a list is an ordered collection of items, while a tuple is an unordered collection of items."
  Total -10.96, and **wrong**: tuples are ordered too.
- **beam 8**: " that a tuple is immutable, while a list is mutable." Total -9.32, and correct.

Once greedy had written "a list is", the most likely way to continue was to describe lists, and it never got back
to the real difference. The model itself rated the correct sentence as more likely overall. Greedy just could not
see that far ahead.

Greedy also got stuck on the coffee prompt: "first brew the coffee beans, then grind them, then store them, then
grind them again, then store them, and". Repetition loops like this are a common weakness of greedy decoding.

## More probable is not always better

Look at what beam search wrote for the capital of France:

```text
greedy  -17.65 | ' Paris. It is the largest city in Europe and the third largest city in the world. ...'
beam 4  -10.97 | ':\nA. Paris\nB. London\nC. Rome\nD. Berlin\n\nA. Paris is the capital of'
```

The multiple-choice format scores as more probable, probably because this kind of text is common in training data.
It is also not what you wanted. Beam search turned the weekend prompt into a quiz too, and so did beam 8 for the
debugging prompt. Beam search finds what the model considers likely, and likely text can be generic or oddly
formatted. That is one reason chat models usually sample with a moderate
[temperature](/tips/2026-10-03-llm_temperature/) instead of using beam search.

## A gotcha: `do_sample=False` was not pure greedy

Our first run gave strange numbers: greedy did not always pick the most likely token. The model's own generation
settings include `"repetition_penalty": 1.1`, and `generate()` applies it even with `do_sample=False`. It lowers the
score of tokens already in the text, which changed greedy's choices. The
[Hugging Face generation docs](https://huggingface.co/docs/transformers/generation_strategies) note that settings
in a model's `GenerationConfig` are used unless you override them (checked 2026-10-14).

So we passed `repetition_penalty=1.0` and checked that greedy picks the top token at every step:

```python
out = model.generate(ids, max_new_tokens=24, min_new_tokens=24, do_sample=False,
                     repetition_penalty=1.0, num_beams=beams, pad_token_id=tok.eos_token_id)
...
assert torch.equal(logp.argmax(-1), cont[0]), "greedy must pick the top token at every step"
```

If you compare decoding methods, print `model.generation_config` first.

## Which one to use

| Task | Decoding |
|---|---|
| Short, factual output where the model is usually right (classification, extraction) | Greedy: fast and repeatable |
| Translation or summarization with a fixed format | Beam search (4–8 beams) is a common choice |
| Open-ended chat or writing | Sampling with temperature, top-p |
| Any comparison of methods | Fix the length, check the generation config, compare total log-probabilities |

Beam search costs more: 8 beams means about 8 sequences to compute at every step.

Full scripts and output: [decoding.py](/assets/code/algorithm_tips/decoding.py),
[decoding_steps.py](/assets/code/algorithm_tips/decoding_steps.py), [output](/assets/code/algorithm_tips/decoding_out.txt).

Related: [What does temperature do in an LLM?](/tips/2026-10-03-llm_temperature/) · [What is a KV cache?](/tips/2026-10-20-kv_cache_memoization/)
