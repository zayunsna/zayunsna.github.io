---
layout: post
title: "Speculative decoding, measured on a laptop: 10x fewer model passes, no speedup"
description: >
  A 0.5B draft model cut Qwen2.5-1.5B's forward passes from 128 to 13 and produced exactly the same text, but
  tokens per second did not move. Prompt lookup decoding was 1.30x faster when the answer copied the prompt and
  slower otherwise. Local transformers test on an Apple M5, with the reasons.
image: /assets/img/tips/speculative_decoding/cover.png
sitemap:
  changefreq: daily
  priority: 1.0
---

# Speculative decoding, measured on a laptop: 10x fewer model passes, no speedup

A language model writes one token per forward pass, and for big models each pass is slow. **Speculative decoding**
lets a small, fast **draft model** guess several tokens ahead, then has the big **target model** check all of the
guesses in a single pass. Every guess the target agrees with is a token it did not need a pass of its own for.
With greedy decoding the output is exactly what the target would have written alone.

I tried it on a laptop with Hugging Face transformers: Qwen2.5-1.5B-Instruct as the target, Qwen2.5-0.5B-Instruct
as the draft. On a code prompt the 1.5B model needed **13 passes instead of 128**. The text was identical, token
for token. The speed was identical too: **16.8 tokens per second both ways.**

_Tested on 2026-10-27 with PyTorch 2.14.1, transformers 5.17.0 and Python 3.12.13 on an Apple M5 (16 GB), Apple GPU
via MPS, float32, greedy decoding, no API key. Models: Qwen2.5-1.5B-Instruct (revision 989aa79) and
Qwen2.5-0.5B-Instruct (revision 7ae5576). Speeds are the median of 5 runs, run in rotation.
Script: [speculative.py](/assets/code/speculative_decoding/speculative.py), output:
[speculative_out_mps.txt](/assets/code/speculative_decoding/speculative_out_mps.txt)._

## Turning it on

In transformers, passing a second model as `assistant_model` is all it takes. The two models must share a tokenizer,
which the Qwen2.5 family does.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
target = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", dtype=torch.float32).to("mps")
draft = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", dtype=torch.float32).to("mps")

messages = [{"role": "user", "content": "Write a Python function is_palindrome(s) that ignores case and non-letters."}]
inputs = tok.apply_chat_template(messages, add_generation_prompt=True, return_dict=True,
                                 return_tensors="pt").to("mps")

greedy = dict(max_new_tokens=128, do_sample=False, repetition_penalty=1.0)
plain = target.generate(**inputs, **greedy)
assisted = target.generate(**inputs, **greedy, assistant_model=draft)       # speculative decoding
lookup = target.generate(**inputs, **greedy, prompt_lookup_num_tokens=10)   # prompt lookup, no draft model
print(torch.equal(plain, assisted), torch.equal(plain, lookup))             # True True
```

`repetition_penalty=1.0` is there on purpose: Qwen2.5-Instruct's default generation config sets a repetition
penalty, which would make "greedy" not quite greedy. The [chat template](/tips/2026-10-27-chat_template/) post shows
what `apply_chat_template` builds.

The last `generate` call is a second kind of drafting. **Prompt lookup decoding** needs no draft model: it looks for the
last two tokens earlier in the text (the prompt and what has been generated so far) and proposes whatever followed
them there. That works when the answer copies from the input.

When the draft model sets nothing else, transformers 5.17 lets it propose up to 20 tokens per round and stop early
when its own confidence drops below 0.4 (`num_assistant_tokens=20`, `assistant_confidence_threshold=0.4` in the
library's generation defaults).

## The results

Four prompts, each generating up to 128 tokens:

| Prompt | 1.5B alone | + 0.5B draft | Tokens per 1.5B pass | + prompt lookup | Tokens per 1.5B pass |
|---|---:|---:|---:|---:|---:|
| Explain a hash table in five sentences | 16.5 tok/s | 16.6 (1.01x) | 2.8 | 12.6 (0.77x) | 1.1 |
| Write `is_palindrome(s)` with a docstring | 16.8 tok/s | 16.8 (1.00x) | 9.8 | 12.5 (0.75x) | 1.1 |
| Rewrite a news item as a list of facts | 16.5 tok/s | 15.9 (0.97x) | 6.7 | **21.4 (1.30x)** | 2.6 |
| Open a short story about a lighthouse keeper | 17.0 tok/s | 16.3 (0.96x) | 2.3 | 11.6 (0.68x) | 1.0 |

![Bar chart of speed relative to the 1.5B model alone on an Apple M5 GPU. The 0.5B draft gives 1.01x, 1.00x, 0.97x and 0.96x on the explain, code, rewrite and story prompts while 2.8, 9.8, 6.7 and 2.3 tokens are accepted per 1.5B pass. Prompt lookup gives 0.77x, 0.75x, 1.30x and 0.68x, with 1.1, 1.1, 2.6 and 1.0 tokens per pass](/assets/img/tips/speculative_decoding/speedup_by_prompt.png)

In all eight drafted runs, the output matched the 1.5B model alone token for token.
The draft model was often right: on the code prompt, the 1.5B model accepted almost 10 tokens per pass, since
Python boilerplate and docstrings are easy to guess. Fewer passes still did not turn into speed.

## Why fewer passes did not mean faster

Speculative decoding trades target passes for draft passes. It pays off only when two things hold.

**1. The draft must be much cheaper than the target.** On this GPU the 0.5B model alone wrote about 43.5 tokens per
second and the 1.5B about 16.8: only **2.6x faster**, with a third of the parameters. Generating the 128 code tokens
with the draft alone takes about 2.9 seconds, already 39% of the 7.6-second assisted run before the target has
checked anything. I did not profile where the rest of the time goes.

**2. Checking k tokens must cost about the same as generating one.** I timed one 1.5B forward pass after a cached
prompt:

| Tokens checked in one pass | Time | Compared with 1 token |
|---:|---:|---:|
| 1 | 57.9 ms | 1.00x |
| 4 | 60.3 ms | 1.04x |
| 10 | 90.7 ms | 1.57x |

Four tokens are almost free; ten are not. Prompt lookup proposed up to 10 tokens per pass here. On the explain, code
and story prompts it gained at most 0.13 tokens per pass, so it paid for long checks and got barely one token each
time. That is why it was **slower** there. On the rewrite prompt the answer repeated phrases from the article,
prompt lookup gained 2.6 tokens per pass, and its guesses cost nothing to make: **1.30x faster**.

The papers that introduced the method report their gains on large targets: 2x to 3x on T5-XXL
([Leviathan et al., 2022](https://arxiv.org/abs/2211.17192)) and 2x to 2.5x on the 70-billion-parameter Chinchilla
([Chen et al., 2023](https://arxiv.org/abs/2302.01318)). Chen et al. build on the observation that scoring a short
continuation in parallel takes about as long as sampling a single token from the large model. A 0.5B draft for a
1.5B target is too close in cost to show that.

## On the CPU

I ran the same script on the CPU (4 threads). The tokens and the number of passes for both drafting methods were the
same as on the GPU, but the timings swung by up to 2.5x between repeats on this machine, with other programs
running, so I do not report CPU speeds. On the prompt with the steadiest CPU timings (explain), the draft model was
slower than the 1.5B model alone: [speculative_out_cpu.txt](/assets/code/speculative_decoding/speculative_out_cpu.txt).

## When to try it

- **Use a draft that is much smaller than the target**, from the same family so the tokenizer matches. A 0.5B draft
  for a 7B or larger target is the kind of pairing the method is meant for. I did not test one here.
- **Try prompt lookup for tasks that copy the input**: rewriting, extracting fields, summaries with quotes, editing
  code. It needs no second model and no extra memory. For open-ended writing it made generation slower.
- **Measure with your own prompts.** Tokens per pass tells you how often the guesses are right; tokens per second
  tells you whether that helped. Here the two disagreed.
- **Greedy output does not change.** It matched token for token in every run. With sampling, the method keeps the
  target's probability distribution, but individual samples will differ from a run without it.

It is the same idea as the [KV cache](/tips/2026-10-20-kv_cache_memoization/): spend some extra work so the
expensive step runs fewer times. Unlike the KV cache, the trade only pays when the expensive step really is expensive.

Sources: [Assisted decoding](https://huggingface.co/docs/transformers/en/assisted_decoding) in the transformers docs,
[prompt lookup decoding](https://github.com/apoorvumang/prompt-lookup-decoding) by its author, and both papers above;
checked 2026-10-27. Charts: [sd_charts.py](/assets/code/speculative_decoding/sd_charts.py).
