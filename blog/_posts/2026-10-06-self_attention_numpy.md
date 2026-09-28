---
layout: post
title: "Self-attention from scratch in NumPy (checked against PyTorch)"
description: >
  Scaled dot-product self-attention in 15 lines of NumPy, verified against PyTorch's scaled_dot_product_attention,
  with measured answers to why we divide by √d_k and why long sequences get expensive.
image: /assets/img/blog/self_attention_numpy/cover.png
sitemap:
  changefreq: daily
  priority: 1.0
---

# Self-attention from scratch in NumPy (checked against PyTorch)

**TL;DR** — Self-attention is `softmax(Q Kᵀ / √d_k) V`: each token scores every other token, turns the scores into weights
that sum to 1, and takes a weighted average of the values. The NumPy version below matches PyTorch's
`scaled_dot_product_attention` to within 1e-15. Dividing by `√d_k` keeps the softmax from collapsing onto a single token.

_Tested on 2026-09-28 with NumPy 2.5.3, PyTorch 2.14.0, Python 3.12, on an Apple M5._

## Key points

- **Q, K, V** are three linear projections of the same input: queries ask, keys are matched against, values are what gets averaged.
- **Scores** are `Q Kᵀ`, one number for every pair of tokens, so the score matrix is `(sequence length × sequence length)`.
- **Scaling by `√d_k`** keeps the scores' spread constant. Without it, the largest weight averaged 0.97 at `d_k = 1024` in our test.
- **A causal mask** blocks attention to future tokens. That's how GPT-style models generate text left to right.
- **Cost grows with the square of the length.** Doubling the sequence roughly quadruples the score matrix.

## How does self-attention work, step by step?

1. **Project** the input `X` (one row per token) into queries, keys and values: `Q = X W_q`, `K = X W_k`, `V = X W_v`.
2. **Score** every pair of tokens with a dot product: `Q Kᵀ`.
3. **Scale** the scores by `1 / √d_k`, where `d_k` is the key size.
4. **Softmax** each row, so each token's weights sum to 1.
5. **Average** the values with those weights: `weights @ V`.

In code:

```python
import numpy as np

def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)          # subtract the max for numerical stability
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)

def self_attention(X, W_q, W_k, W_v, causal=False):
    Q, K, V = X @ W_q, X @ W_k, X @ W_v             # (seq, d_k), (seq, d_k), (seq, d_v)
    scores = Q @ K.T / np.sqrt(K.shape[-1])          # (seq, seq): how much each token attends to each other token
    if causal:                                       # a token may only look at itself and earlier tokens
        scores = np.where(np.tril(np.ones_like(scores)) == 1, scores, -np.inf)
    weights = softmax(scores, axis=-1)               # each row sums to 1
    return weights @ V, weights

rng = np.random.default_rng(0)
seq_len, d_model, d_k = 6, 16, 8
X = rng.normal(size=(seq_len, d_model))              # 6 tokens, 16-dim embeddings
W_q, W_k, W_v = (rng.normal(size=(d_model, d_k)) / np.sqrt(d_model) for _ in range(3))   # typical init scale

out, weights = self_attention(X, W_q, W_k, W_v)
print("output:", out.shape, " weights:", weights.shape)
print("row sums:", weights.sum(axis=1).round(6))
print("weights for token 0:", weights[0].round(2))
```

```
output: (6, 8)  weights: (6, 6)
row sums: [1. 1. 1. 1. 1. 1.]
weights for token 0: [0.03 0.08 0.24 0.11 0.44 0.1 ]
```

Token 0's new representation is 44% token 4's value, 24% token 2's, and so on.

![Two 6-by-6 heatmaps of attention weights. Left, full self-attention: every token spreads its weight over all six tokens, for example token 0 gives 0.44 to token 4. Right, causal self-attention: the upper triangle is zero, token 0 gives 1.00 to itself, and the last row (token 5) is the same as in the full version](/assets/img/blog/self_attention_numpy/attention_weights.png)

With the causal mask (right), everything above the diagonal is zero. Token 0 can only see itself, so its weight is 1.00.
The last token can see everything, so its row is identical in both versions.

## Does this match PyTorch?

Yes. The same `Q`, `K`, `V` passed to `torch.nn.functional.scaled_dot_product_attention` give the same output, with and without the mask:

```python
# continues from the code above (X, W_q, W_k, W_v, self_attention)
import numpy as np
import torch
import torch.nn.functional as F

for causal in [False, True]:
    ours, _ = self_attention(X, W_q, W_k, W_v, causal=causal)
    q, k, v = (torch.tensor(X @ W, dtype=torch.float64) for W in (W_q, W_k, W_v))
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=causal).numpy()
    print(f"causal={causal!s:<5} max difference vs PyTorch: {np.abs(ours - ref).max():.1e}")
```

```
causal=False max difference vs PyTorch: 4.4e-16
causal=True  max difference vs PyTorch: 4.4e-16
```

A difference of 4.4e-16 is float64 rounding noise.

## Why divide by √d_k?

A dot product of two random `d_k`-dimensional vectors has a standard deviation of about `√d_k`.
Larger scores make the softmax extremely peaked, so almost all the weight goes to one token and the gradients for the others vanish.
We measured the average largest weight over 10 tokens and 2,000 random trials:

```python
import numpy as np

def softmax(z):
    e = np.exp(z - z.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

rng = np.random.default_rng(0)
seq_len, trials = 10, 2_000
print(f"{'d_k':>5} | {'std of scores':>13} | {'top weight, unscaled':>20} | {'top weight, scaled':>18}")
for d_k in [4, 16, 64, 256, 1024]:
    q = rng.normal(size=(trials, 1, d_k))
    k = rng.normal(size=(trials, seq_len, d_k))
    scores = (q @ k.transpose(0, 2, 1))[:, 0, :]              # (trials, seq_len)
    top_raw = softmax(scores).max(axis=-1).mean()
    top_scaled = softmax(scores / np.sqrt(d_k)).max(axis=-1).mean()
    print(f"{d_k:>5} | {scores.std():>13.1f} | {top_raw:>20.3f} | {top_scaled:>18.3f}")
```

```
  d_k | std of scores | top weight, unscaled | top weight, scaled
    4 |           2.0 |                0.496 |              0.304
   16 |           4.0 |                0.726 |              0.319
   64 |           7.9 |                0.856 |              0.315
  256 |          15.9 |                0.931 |              0.318
 1024 |          32.0 |                0.968 |              0.319
```

![Line chart of the average largest attention weight versus key dimension d_k from 4 to 1024: without scaling it rises from 0.50 to 0.97, with division by the square root of d_k it stays near 0.32 at every size; uniform attention over 10 tokens would be 0.10](/assets/img/blog/self_attention_numpy/scaling_effect.png)

The standard deviation of the scores is exactly `√d_k` (2, 4, 8, 16, 32). Unscaled, the top weight climbs to 0.968.
Scaled, it stays around 0.32 regardless of `d_k`.

## Why is attention slow on long sequences?

Because the score matrix has one entry per pair of tokens. For `n` tokens it is `n × n`:

```python
import time
import numpy as np

rng = np.random.default_rng(0)
for n in [512, 1024, 2048, 4096]:
    Q = K = rng.normal(size=(n, 64))
    t0 = time.perf_counter()
    for _ in range(3):
        scores = Q @ K.T
    ms = (time.perf_counter() - t0) / 3 * 1000
    print(f"n={n:>5}  scores {scores.shape}  {scores.nbytes / 1e6:>5.0f} MB  {ms:5.1f} ms")
```

```
n=  512  scores (512, 512)      2 MB    0.6 ms
n= 1024  scores (1024, 1024)      8 MB    2.3 ms
n= 2048  scores (2048, 2048)     34 MB   11.2 ms
n= 4096  scores (4096, 4096)    134 MB   49.9 ms
```

Each doubling of `n` multiplies memory by 4 and time by about 4–5. That is one head in one layer, in float64;
real models have many of both, which is why long-context models use tricks such as memory-efficient attention kernels.

## Sources

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Vaswani et al., 2017, section 3.2.1 (scaled dot-product attention)
- [`torch.nn.functional.scaled_dot_product_attention`](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) — PyTorch docs, checked 2026-09-28

Related: [Self Attention에 대해 공부 — a 2×2 worked example (in Korean)](/blog/2023-09-05-self_attention/) · [LSTM input shape explained](/blog/2026-10-05-lstm_input_shape/)
