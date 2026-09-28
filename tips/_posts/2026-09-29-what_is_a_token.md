---
layout: post
title: "What is a token, and why do LLMs count tokens instead of words?"
description: >
  LLMs read text as tokens, not words or characters. We counted tokens for English, Korean, code and numbers
  across four tokenizers: the same Korean sentence ranged from 22 to 82 tokens.
image: /assets/img/tips/what_is_a_token/cover.png
sitemap:
  changefreq: daily
  priority: 1.0
---

# What is a token, and why do LLMs count tokens instead of words?

**TL;DR** — A token is a chunk of text, often a word piece, that a language model reads as one unit.
Prices, context limits and speed are all measured in tokens. The same meaning can cost very different
numbers of tokens: in our test, one Korean sentence took 22 tokens with a modern tokenizer and 82 with an older one.

_Tested on 2026-09-28 with tiktoken 0.14.0 and the Qwen2.5 tokenizer (transformers 5.17.0). Tokenizers only; no model or API key needed._

## Key points

- **A tokenizer** splits text into tokens from a fixed vocabulary. Common words are one token; rare words are split into pieces.
- **Everything is billed and limited in tokens:** API prices, the context window, and the maximum output length.
- **Non-English text usually needs more tokens** for the same meaning. Newer tokenizers narrow the gap a lot.
- **Different models use different tokenizers,** so the same text has different token counts on different models.
- **Count tokens with the model's own tokenizer** before you send long inputs.

## How does text get split into tokens?

Here is how the `o200k_base` tokenizer (used by the GPT-4o, GPT-4.1 and GPT-5 families, according to tiktoken's model table) splits a few inputs:

```python
import tiktoken

o200k = tiktoken.get_encoding("o200k_base")

def pieces(text):
    return [o200k.decode_single_token_bytes(t).decode("utf-8", errors="replace") for t in o200k.encode(text)]

for text in ["tokenization", " unbelievably", "머신러닝", "1234567.89"]:
    print(f"{text!r:<16} {pieces(text)}")
```

```
'tokenization'   ['token', 'ization']
' unbelievably'  [' unbelievably']
'머신러닝'           ['머', '신', '러', '닝']
'1234567.89'     ['123', '456', '7', '.', '89']
```

- A common word with its leading space (`" unbelievably"`) can be a single token.
- A longer word is split into familiar pieces: `token` + `ization`.
- The Korean word for "machine learning" becomes one token per syllable.
- Numbers are split into groups of up to three digits. That is one reason LLMs are unreliable at exact arithmetic.

## How many tokens does the same content cost?

We counted tokens for four inputs with four tokenizers, from an older one (`r50k_base`, GPT-3) to recent ones:

```python
import tiktoken
from transformers import AutoTokenizer

texts = {
    "English":  "Machine learning models learn patterns from data instead of following hand-written rules.",
    "Korean":   "머신러닝 모델은 사람이 직접 쓴 규칙 대신 데이터에서 패턴을 학습한다.",
    "Python":   "df.groupby('category')['amount'].sum().sort_values(ascending=False)",
    "Number":   "The total was 1234567.89 dollars on 2026-09-28.",
}
tokenizers = {
    "r50k (GPT-3)": tiktoken.get_encoding("r50k_base").encode,
    "cl100k":      tiktoken.get_encoding("cl100k_base").encode,
    "o200k":       tiktoken.get_encoding("o200k_base").encode,
    "Qwen2.5":     AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B").encode,
}
print(f"{'text':<8} {'chars':>5} " + " ".join(f"{name:>12}" for name in tokenizers))
for label, text in texts.items():
    counts = [len(enc(text)) for enc in tokenizers.values()]
    print(f"{label:<8} {len(text):>5} " + " ".join(f"{c:>12}" for c in counts))
```

```
text     chars r50k (GPT-3)       cl100k        o200k      Qwen2.5
English     89           15           14           14           14
Korean      39           82           40           22           27
Python      67           22           15           15           15
Number      47           17           19           19           28
```

![Grouped bar chart of token counts. English: 15, 14, 14, 14. Korean: 82 with r50k, 40 with cl100k, 22 with o200k, 27 with Qwen2.5. Python: 22, 15, 15, 15. Number: 17, 19, 19, 28](/assets/img/tips/what_is_a_token/token_counts.png)

What the numbers say:

- **English is cheap everywhere:** 14–15 tokens for 89 characters.
- **Korean depends heavily on the tokenizer.** The older `r50k_base` has almost no Korean in its vocabulary, so it falls back to raw UTF-8 bytes: 82 tokens for 39 characters. `o200k_base` needs 22.
- **Even with the best tokenizer here, the Korean sentence took 22 tokens against 14 for the same meaning in English**, about 1.6× more.
- **Qwen2.5 splits numbers into single digits,** so the number sentence cost 28 tokens there, against 19 with `o200k_base`.

These are single sentences, so treat the ratios as examples, not constants. Measure your own text.

## Why does this matter in practice?

| Where tokens show up | What it means for you |
|---|---|
| API pricing | You pay per input and output token. Longer or non-English text costs more |
| Context window | The limit (for example 128k) is in tokens, not words or pages |
| Output limits | `max_tokens` cuts off the answer by tokens |
| Speed | Models generate one token at a time, so more tokens take longer |
| Odd mistakes | Counting letters or digits is hard when the model sees `token` + `ization`, not letters |

## How do you count tokens before sending text?

Use the tokenizer that matches the model. For OpenAI models, `tiktoken` can pick it by model name:

```python
import tiktoken

enc = tiktoken.encoding_for_model("gpt-4o")      # resolves to o200k_base
print(len(enc.encode("How many tokens is this sentence?")))
```

```
7
```

For open models, load the tokenizer from the model's Hugging Face page with `AutoTokenizer.from_pretrained(...)`, as in the example above.

## Sources

- [openai/tiktoken](https://github.com/openai/tiktoken) — tokenizer library and model-to-encoding table, checked 2026-09-28
- [Qwen/Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B) — tokenizer used above
