---
layout: post
title: "How much does quantization hurt an LLM? 8-bit vs 4-bit, tested"
description: >
  We rounded a small model's weights to 8, 4, 3 and 2 bits and measured perplexity and correct answers. 8-bit changed nothing;
  4-bit dropped the 0.5B model from 11 to 4 correct answers, while a 1.5B model kept all 12.
image: /assets/img/tips/quantization_quality/cover.png
sitemap:
  changefreq: daily
  priority: 1.0
---

# How much does quantization hurt an LLM? 8-bit vs 4-bit, tested

**TL;DR** — **Quantization** stores a model's weights with fewer bits, so the model takes less memory. In our test, 8-bit weights
were as good as the original. At 4 bits, a 0.5B model answered only 4 of 12 handbook questions correctly (down from 11),
while a 1.5B model still got all 12. Below 4 bits, the small model broke down completely.

_Tested on 2026-09-28 with Qwen2.5-0.5B-Instruct and Qwen2.5-1.5B-Instruct (transformers 5.17.0, PyTorch 2.14.0), locally on an Apple M5, no API key.
This is **simulated** weight quantization: weights are rounded to fewer bits and converted back to float, which shows the effect on quality but not on speed or real memory use._

## Key points

- **Fewer bits per weight, smaller model:** float32 uses 32 bits per weight, 8-bit uses 8, 4-bit uses 4.
- **8-bit was lossless here** for both models: same answers, perplexity within 1%.
- **How you quantize matters.** At 4 bits, one scale per row gave 1 correct answer; one scale per group of 128 weights gave 4.
- **Bigger models tolerated 4 bits much better:** the 1.5B model kept 12 of 12 answers, with perplexity up 21%.
- **Test a quantized model on your own tasks** before you ship it. Perplexity and task accuracy can tell different stories.

## How did we test it?

We rounded every linear layer's weights to *n* bits and back (symmetric, round-to-nearest), then measured two things:
**perplexity** on a fixed paragraph (how surprised the model is by the text; lower is better) and the number of correct answers
on the 12-question test set from our [evaluation post](/tips/2026-10-09-evaluate_llm_answers/), using its key-fact check.

```python
import copy, math, os, runpy, contextlib, io
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

name = os.environ.get("MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
tok = AutoTokenizer.from_pretrained(name)
base = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).eval()

def fake_quantize(model, bits, group=None):
    """Round every Linear weight to `bits` bits and back to float (symmetric).
    group=None: one scale per output row. group=128: one scale per 128 weights, as most 4-bit formats do."""
    m = copy.deepcopy(model)
    qmax = 2 ** (bits - 1) - 1
    with torch.no_grad():
        for mod in m.modules():
            if isinstance(mod, torch.nn.Linear):
                w = mod.weight
                g = w.reshape(-1, group) if group else w            # rows of `group` weights share one scale
                scale = g.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / qmax
                mod.weight.copy_((torch.round(g / scale).clamp(-qmax, qmax) * scale).reshape(w.shape))
    return m

# Perplexity on a fixed paragraph (lower = the model predicts the text better)
text = ("Data teams often start a project by collecting historical records, cleaning missing values and "
        "checking how each variable is distributed. Once the data is ready, they split it into training and "
        "test sets, fit a simple baseline, and only then try more complex models. The final step is to monitor "
        "the model in production, because the data it sees tomorrow may differ from the data it was trained on.")
ids = tok(text, return_tensors="pt").input_ids

def perplexity(model):
    with torch.no_grad():
        return math.exp(model(ids, labels=ids).loss.item())

# Answer quality: the 12-question test set and key-fact check from the evaluation post
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    ev = runpy.run_path("a11_eval.py")
rag, tests = ev["rag"], ev["tests"]
rule = " If the answer is not in the context, say: I don't know."

def key_fact_score(model):
    ask = rag["ask"]
    ask.__globals__["llm"] = model                      # swap the generator; retrieval stays the same
    return sum(key in ask(q, rag["retrieve"](q), rule).lower() for q, _, key in tests)

n_params = sum(mod.weight.numel() for mod in base.modules() if isinstance(mod, torch.nn.Linear))
print(f"Linear weights: {n_params / 1e6:.0f} M parameters")
print(f"{'weights':<13}{'size of Linear weights':>24}{'perplexity':>12}{'key facts (of 12)':>19}")
results = {}
settings = [("float32", 32, None), ("8-bit", 8, None), ("4-bit", 4, None), ("4-bit, g128", 4, 128),
            ("3-bit, g128", 3, 128), ("2-bit, g128", 2, 128)]
if os.environ.get("QUICK"):                             # the larger model: fewer settings to fit in memory and time
    settings = [s for s in settings if s[0] in ("float32", "8-bit", "4-bit, g128")]
for label, bits, group in settings:
    model = base if bits == 32 else fake_quantize(base, bits, group)
    size_mb, ppl, score = n_params * bits / 8 / 1e6, perplexity(model), key_fact_score(model)
    results[label] = (bits, size_mb, ppl, score)
    print(f"{label:<13}{size_mb:>21.0f} MB{ppl:>12.2f}{score:>19}")
```

Qwen2.5-0.5B-Instruct (default):

```
Linear weights: 494 M parameters
weights        size of Linear weights  perplexity  key facts (of 12)
float32                       1976 MB       19.87                 11
8-bit                          494 MB       19.72                 11
4-bit                          247 MB       54.31                  1
4-bit, g128                    247 MB       29.86                  4
3-bit, g128                    185 MB      462.82                  0
2-bit, g128                    123 MB 49336006.82                  1
```

Qwen2.5-1.5B-Instruct (`MODEL=Qwen/Qwen2.5-1.5B-Instruct QUICK=1`, three settings to fit in 16 GB of memory):

```
Linear weights: 1544 M parameters
weights        size of Linear weights  perplexity  key facts (of 12)
float32                       6174 MB       14.05                 12
8-bit                         1544 MB       14.02                 12
4-bit, g128                    772 MB       16.99                 12
```

![Two bar charts comparing Qwen2.5-0.5B and Qwen2.5-1.5B. Perplexity increase versus float32: about 0% at 8-bit for both; at 4-bit with groups of 128, +50% for 0.5B and +21% for 1.5B. Handbook questions answered: 11 and 12 at float32 and 8-bit; at 4-bit, 4 for 0.5B and 12 for 1.5B](/assets/img/tips/quantization_quality/quantization_quality.png)

Both runs gave identical numbers when repeated. The two models share the same tokenizer, which we checked, so the test set is scored the same way.

## What do the results show?

| Setting | 0.5B: perplexity | 0.5B: answers | 1.5B: perplexity | 1.5B: answers |
|---|---|---|---|---|
| float32 | 19.87 | 11 / 12 | 14.05 | 12 / 12 |
| 8-bit | 19.72 | 11 / 12 | 14.02 | 12 / 12 |
| 4-bit, one scale per row | 54.31 | 1 / 12 | — | — |
| 4-bit, groups of 128 | 29.86 | 4 / 12 | 16.99 | 12 / 12 |
| 3-bit, groups of 128 | 462.82 | 0 / 12 | — | — |

- **8-bit is safe** in this test. The tiny perplexity drop is noise, not an improvement.
- **Group size is a big lever.** Real 4-bit formats use small groups for exactly this reason.
- **The larger model held up at 4 bits.** Its perplexity rose 21% against 50% for the small model, and its answers didn't change.
- **Perplexity isn't the whole story.** The 1.5B model's perplexity went up, but its answers on our task stayed correct.

## What does this mean in practice?

| Situation | Suggestion |
|---|---|
| You have enough memory | Run at 16-bit (bfloat16 or float16), the usual default |
| Memory is tight | Try 8-bit first; it cost nothing here |
| You need 4-bit | Use a real method such as GPTQ or AWQ with small groups, rather than plain rounding |
| Choosing between a small model at 16-bit and a bigger one at 4-bit | Test both on your own questions; here the bigger 4-bit model won |
| Any quantized model | Re-run your evaluation set before switching |

Our simulation uses plain rounding, which is the simplest method. GPTQ and AWQ use calibration data to choose better rounding and usually lose less quality at 4 bits.
Sizes in the table count only the linear-layer weights and leave out the small overhead for storing the scales.

## Sources

- [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323) — Frantar et al., 2022
- [AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978) — Lin et al., 2023
- [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) — model card, checked 2026-09-28

Related: [How do you evaluate LLM answers?](/tips/2026-10-09-evaluate_llm_answers/) · [Prompting vs RAG vs fine-tuning: which one do you need?](/tips/2026-10-06-prompt_rag_finetune/)
