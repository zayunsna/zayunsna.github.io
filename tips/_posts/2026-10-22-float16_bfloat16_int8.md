---
layout: post
title: "fp32, fp16, bf16 and int8: what each format stores, and where it breaks"
description: >
  Models are stored in 32, 16 or 8 bits. We checked with PyTorch what each format does with the same numbers: fp16
  turns 70,000 into infinity, bf16 cannot tell 1.001 from 1, and adding 0.001 ten thousand times stops at 4 in fp16
  and at 0.5 in bf16. One outlier made int8 error 12 times worse.
image: /assets/img/tips/float16_bfloat16_int8/cover.png
sitemap:
  changefreq: daily
  priority: 1.0
---

# fp32, fp16, bf16 and int8: what each format stores, and where it breaks

A model's weights are just numbers, and how many bits each number gets decides how much memory the model needs.
For the 494 million parameters of Qwen2.5-0.5B:

| Format | Bytes per number | Memory |
|---|---:|---:|
| fp32 | 4 | 1.84 GiB |
| fp16 / bf16 | 2 | 0.92 GiB |
| int8 | 1 | 0.46 GiB |
| int4 | 0.5 | 0.23 GiB |

Halving the bits halves the memory. What you give up depends on **which** bits you drop. This post shows that with
real bit patterns and a few small experiments.

_Tested on 2026-10-14 with PyTorch 2.14.1 and Python 3.12.13 on CPU. Full script:
[formats.py](/assets/code/algorithm_tips/formats.py), output: [formats_out.txt](/assets/code/algorithm_tips/formats_out.txt)._

## Three parts: sign, exponent, fraction

A floating-point number is stored as a sign, an exponent (how big) and a fraction (how precise). Here is 0.1 in
each format, read straight from PyTorch's memory:

![The bit patterns of 0.1 in fp32, fp16 and bf16, coloured by part: 1 sign bit in each; 8 exponent bits in fp32 and bf16 but 5 in fp16; 23 fraction bits in fp32, 10 in fp16 and 7 in bf16. Next to each row, the result of adding 0.001 ten thousand times: 10.0004, 4 and 0.5](/assets/img/tips/float16_bfloat16_int8/chart.png)

```text
fp32: 32 bits, max 3.403e+38, smallest normal 1.18e-38, eps 1.19e-07 | 0.1 = 0 01111011 10011001100110011001101
fp16: 16 bits, max 6.55e+04, smallest normal 6.1e-05, eps 0.000977 | 0.1 = 0 01011 1001100110
bf16: 16 bits, max 3.39e+38, smallest normal 1.18e-38, eps 0.00781 | 0.1 = 0 01111011 1001101
```

The two 16-bit formats split their bits in opposite ways:

- **fp16** keeps 10 fraction bits (more precision) but only 5 exponent bits, so its largest value is 65,504.
- **bf16** keeps fp32's 8 exponent bits (the same range as fp32) and only 7 fraction bits. Google's TPU
  documentation says the dynamic range of bfloat16 and float32 is equivalent
  ([bfloat16 on Cloud TPU](https://cloud.google.com/tpu/docs/bfloat16), checked 2026-10-14).

`eps` is the gap between 1 and the next number the format can store: 0.000977 for fp16 and 0.0078 for bf16.

## Range: where fp16 breaks

```text
fp32: 70000 -> 70000, 1e-8 -> 1e-08
fp16: 70000 -> inf, 1e-8 -> 0
bf16: 70000 -> 70144, 1e-8 -> 1.00117e-08
```

fp16 cannot hold 70,000 at all, and very small values become 0. Large activations or gradients can overflow to
infinity, which is why fp16 training usually uses loss scaling. bf16 keeps both numbers, just roughly: 70,000
became 70,144.

## Precision: where bf16 breaks

```text
fp32: 1 + 0.001 = 1.001000 | running sum = 10.0004
fp16: 1 + 0.001 = 1.000977 | running sum = 4
bf16: 1 + 0.001 = 1.000000 | running sum = 0.5
```

In bf16, 1 + 0.001 is simply 1: the step is smaller than half the gap to the next storable number.

The running sum adds 0.001 ten thousand times; the answer should be 10. In fp16 it stops at 4, in bf16 at 0.5.
Near 4, fp16 numbers are about 0.0039 apart, and near 0.5, bf16 numbers are also about 0.0039 apart. Once the total
is that large, adding 0.001 rounds back to the same number, every time.

This is why mixed-precision training keeps sums and weight updates in fp32 while doing the heavy multiplications
in 16-bit. The TPU page above describes the same split: bfloat16 for multiplications, float32 for accumulations.

## int8: no fraction at all, so you need a scale

int8 stores whole numbers from -128 to 127, in the two's complement layout from
[our bit operations post](/blog/2026-10-22-bits_visualized/). Weights are small decimals like 0.013, so int8
quantization stores `round(weight / scale)` and multiplies back by `scale` when it computes. The simplest scale
maps the largest weight to 127:

```python
def int8_roundtrip(x):
    scale = x.abs().max() / 127
    q = torch.clamp(torch.round(x / scale), -128, 127).to(torch.int8)
    return q, q.float() * scale, scale
```

On 4,096 random weights with a typical spread, and then the same weights with a single large value added:

```text
normal weights            : scale 0.000646, distinct int8 values used 187, mean relative error 1.0%
same + one outlier of 1.0 : scale 0.007874, distinct int8 values used  22, mean relative error 12.5%
bf16 of the normal weights: mean relative error 0.14%
```

One outlier stretched the scale 12 times. The other 4,095 weights then had to share only 22 of the 256 possible
values, and the error went from 1% to 12.5%. That is why real int8 and int4 methods use a separate scale for each
row or small block of weights, and some keep outliers in higher precision. The
[quantization test on this blog](/tips/2026-10-14-quantization_quality/) shows what those errors do to a model's
answers.

## Which format when

| Situation | Format |
|---|---|
| Running a model on a modern GPU | bf16 if supported, otherwise fp16 |
| Training | Mixed precision: 16-bit for compute, fp32 for accumulation and weight updates |
| Not enough memory to load the model | int8, then 4-bit, with per-block scales; check quality on your own task |
| CPU-only, or exact reproducibility | fp32 |

The numbers here come from synthetic weights and single operations, not from a real model's layers. They show
**why** each format fails, not how often it happens inside a particular model.

Related: [How much does quantization hurt an LLM? 8-bit vs 4-bit, tested](/tips/2026-10-14-quantization_quality/) · [What is a KV cache?](/tips/2026-10-20-kv_cache_memoization/)
