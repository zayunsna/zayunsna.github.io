---
layout: post
title: "What is a KV cache? Memoization for LLMs, measured: 256 tokens in 7.5 s instead of 34 s"
description: >
  An LLM generates one token at a time, and without a cache it recomputes the whole sequence at every step. We
  generated the same text with a local Qwen2.5-0.5B with the cache on and off: identical tokens, 4.5x faster at 256
  tokens. The price is memory, which we calculate for the same model.
image: /assets/img/tips/kv_cache_memoization/cover.png
sitemap:
  changefreq: daily
  priority: 1.0
---

# What is a KV cache? Memoization for LLMs, measured: 256 tokens in 7.5 s instead of 34 s

An LLM writes one token at a time. To choose token 100, it needs information from tokens 1 to 99. Without a cache,
the model reads all 99 again at every step and recomputes the same numbers it computed one step earlier. The
**KV cache** keeps those numbers and reuses them. It is the same trick as memoizing Fibonacci in
[our dynamic programming post](/blog/2026-10-20-dp_visualized/): compute each piece once, look it up afterwards.

We generated the same text with the cache on and off:

| New tokens | Cache on | Cache off | Cache off is slower by | Same tokens? |
|---:|---:|---:|---:|---|
| 32 | 0.97 s | 2.40 s | 2.5x | yes |
| 64 | 1.90 s | 5.13 s | 2.7x | yes |
| 128 | 3.77 s | 12.49 s | 3.3x | yes |
| 256 | 7.52 s | 34.13 s | 4.5x | yes |

_Tested on 2026-10-14 with Qwen2.5-0.5B-Instruct, transformers 5.19.0 and PyTorch 2.14.1 (CPU, float32, 4 threads)
on an Apple Silicon Mac, no API key. A second run differed by less than 0.2 s per row._

## What exactly is cached?

Inside each attention layer, every token is turned into three vectors: a **query**, a **key** and a **value**. The
new token's query is compared with the keys of all earlier tokens to decide which ones matter, and their values are
mixed into the result.

The keys and values of earlier tokens never change: token 5 has the same key whether you are generating token 6 or
token 600. So there is no reason to compute them twice. The cache stores them for every layer, and each new step
only computes the key and value of the one new token. Hugging Face's documentation puts it as storing these
calculations "so they can be reused without recomputing them"
([KV cache docs](https://huggingface.co/docs/transformers/kv_cache), checked 2026-10-14).

## The experiment

Greedy decoding, so both runs must produce exactly the same tokens:

```python
def run(new_tokens, use_cache):
    t = time.perf_counter()
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=new_tokens, min_new_tokens=new_tokens, do_sample=False,
                             repetition_penalty=1.0, use_cache=use_cache, pad_token_id=tok.eos_token_id)
    return out, time.perf_counter() - t

for n in (32, 64, 128, 256):
    with_cache, t_on = run(n, True)
    without, t_off = run(n, False)
    print(f"{n:>4} new tokens: cache on {t_on:6.2f} s | cache off {t_off:6.2f} s | {t_off / t_on:4.1f}x | "
          f"same tokens: {torch.equal(with_cache, without)}")
```

`min_new_tokens` forces the same length in both runs, and `repetition_penalty=1.0` turns off a penalty that this
model's default settings would otherwise apply. The full script is
[kv_cache.py](/assets/code/algorithm_tips/kv_cache.py).

![Line chart of seconds against new tokens: with the cache, time grows in a straight line to 7.5 s at 256 tokens; without it, time curves upward to 34 s, 4.5 times slower](/assets/img/tips/kv_cache_memoization/chart.png)

With the cache, time grows in a straight line: about 29 ms per token. Without it, the line curves upward, because
step `i` processes the whole sequence again. Count the token positions the model has to push through all its
layers (simple arithmetic, not a measurement), with our 43-token prompt and 256 new tokens:

- **with the cache**: the 43 prompt tokens once, then 1 new token per step → 43 + 255 = **298**
- **without**: 43, then 44, then 45, … up to 298 → **43,648**

That is the same shape as the Fibonacci example: the plain version repeats work, and the repeated work grows faster
than the input.

## The price: memory

The cache keeps a key and a value vector for every token, in every layer. For this model:

```text
layers 24, attention heads 14, key/value heads 2, head size 64
cache per token: 24,576 bytes (float32) -> 32k tokens: 768 MiB
```

24,576 bytes = 2 (key and value) × 24 layers × 2 key/value heads × 64 numbers × 4 bytes. At the model's full
context of 32,768 tokens, that is 768 MiB in float32 (half in 16-bit) for one conversation, on top of the model's
own weights.

Notice "key/value heads 2" next to "attention heads 14". Qwen2.5 uses **grouped-query attention**: the 14 query heads
share 2 sets of keys and values. With one key/value set per head, the cache would be 7 times larger. Keeping the
cache small is one of the reasons models are designed this way.

## What this means in practice

- Generation time per token stays roughly flat **because of** the cache. Without it, long answers get slower and
  slower.
- Long contexts cost twice: the attention work (see [what a context window is](/tips/2026-10-07-context_window/))
  and the cache memory that grows with every token.
- When a server limits how many long conversations it can serve at once, the KV cache is often the limit, not the
  weights.
- The cache is on by default in `generate()`. You rarely turn it off, except to measure what it saves.

Raw output: [kv_out_1.txt](/assets/code/algorithm_tips/kv_out_1.txt).

Related: [What is a context window, and what happens when the input is too long?](/tips/2026-10-07-context_window/) · [What is a token?](/tips/2026-09-29-what_is_a_token/)
