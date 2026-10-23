---
layout: post
title: "What is perplexity? Computing it by hand with a small local model"
description: >
  Perplexity is exp of the average negative log-probability per token. We computed it by hand with Qwen2.5-0.5B,
  matched it to the library's loss, and scored five texts: a normal sentence 35.1, the same words shuffled 1,142.8,
  Python code 4.0. Why random letters scored lower than the shuffled sentence, and what that says about comparing numbers.
image: /assets/img/tips/perplexity_explained/cover.png
sitemap:
  changefreq: daily
  priority: 1.0
---

# What is perplexity? Computing it by hand with a small local model

Perplexity measures how surprised a language model is by a text. For every token, take the probability the model gave
to the token that actually came next, then:

```text
perplexity = exp( average over tokens of  -log p(actual next token) )
```

Lower means the model predicted the text better. A perplexity of 35 means that, on average, the model was about as
unsure as if it were choosing uniformly among 35 tokens at every step.

This post computes it by hand with a small model on CPU, checks the result against the library, and scores five texts
to show what the number reacts to, and one way it misleads.

_Tested on 2026-10-23 with Qwen2.5-0.5B (the base model, revision `060db649`), PyTorch 2.14.1, transformers 5.17.0
and Python 3.12.13 on CPU. No API key. Full script: [perplexity.py](/assets/code/perplexity_explained/perplexity.py),
output: [perplexity_out.txt](/assets/code/perplexity_explained/perplexity_out.txt). Two runs printed identical values._

## Computing it by hand

A causal language model outputs, at every position, a score for every token in its vocabulary. `log_softmax` turns
those scores into log-probabilities. The prediction made at position *i* is for token *i*+1, so the logits and the
tokens are shifted by one:

```python
ids = tok(text, return_tensors="pt")["input_ids"]
with torch.no_grad():
    logits = model(ids).logits
logp = torch.log_softmax(logits[0, :-1], dim=-1)      # predictions for tokens 2..n
nll = -logp.gather(1, ids[0, 1:, None])[:, 0]          # -log p of each actual next token
perplexity = torch.exp(nll.mean())
```

The first token is not scored, because nothing comes before it. For the sentence _"The library closes at eight on
weekdays, so if you want to return a book after work, use the drop box by the front door."_:

```text
tokens scored: 27   mean NLL by hand 3.5584   loss from labels= 3.5584
perplexity = exp(mean NLL) = 35.11
```

The average negative log-likelihood is exactly the cross-entropy loss that transformers returns when you pass
`labels=ids`. That gives a one-line version: `torch.exp(model(ids, labels=ids).loss)`.

## Which tokens surprised the model?

Perplexity is an average, so it hides where the surprise came from. The cover image shades each token of the sentence
by the probability the model gave it:

- **Expected:** "to" after "if you want" (0.99), "book" after "return a" (0.95), "at" after "closes" (0.87).
- **Surprising:** "library" right after "The" (0.0001), "eight" after "closes at" (below 0.0001), "drop" after
  "use the" (0.0001).

The surprising tokens are content the model could not know: which building, which time, which detail. The expected ones
are grammar and common phrases. A handful of very unlikely tokens can dominate the average, because −log p grows fast
as p approaches zero.

## Five texts, one model

![Horizontal bar chart of perplexity on a log scale: natural sentence 35.1, same words shuffled 1,142.8, Python code 4.0, random letters 964.9, the model's own greedy text 5.9](/assets/img/tips/perplexity_explained/ppl_texts.png)

| Text | Perplexity | Tokens scored | Total NLL |
|---|---:|---:|---:|
| The natural sentence above | 35.1 | 27 | 96.1 |
| Same words, shuffled | 1,142.8 | 27 | 190.1 |
| A two-line Python function (`def mean(values): ...`) | 4.0 | 12 | 16.7 |
| Random letters and spaces, same length | 964.9 | 58 | 398.6 |
| The model's own greedy continuation of "The library closes at eight on weekdays," (prompt included) | 5.9 | 32 | 56.8 |

- **Word order matters more than vocabulary.** The shuffled sentence has exactly the same tokens and scored 33 times
  higher.
- **Predictable text scores low.** A short, very common code pattern was the most predictable text here.
- **A model's own greedy text scores low too.** Greedy decoding picks the most likely token at every step, so the model
  is rarely surprised when it reads it back. This is why low perplexity is sometimes used as a hint that a text was
  machine-generated. The Python function scored even lower, though, so low perplexity on its own proves nothing.

## The trap: random letters "beat" a real sentence

Random letters scored 964.9, lower than the shuffled real words (1,142.8). That does not mean they are more
predictable. Look at the totals: the random letters cost **398.6** nats of surprise in total, twice the shuffled
sentence's 190.1. But the tokenizer split them into 58 short tokens instead of 27, and perplexity divides by the
number of tokens.

The same effect changes the number for one text depending on the unit you divide by. The natural sentence has a total
NLL of 96.08 over 27 tokens, which are 24 words:

```text
per-token perplexity 35.11   per-word perplexity 54.77
```

So perplexity is only comparable when the **tokenizer, the text and the unit are the same**. Comparing two models with
different tokenizers on per-token perplexity compares different things. When you need such a comparison, normalise by
something both share, such as words, characters or bytes.

## When to use it

- **Comparing versions of one model on fixed text**, such as before and after quantization. The
  [quantization test](/tips/2026-10-14-quantization_quality/) used exactly this, and also showed that perplexity and
  correct answers can disagree.
- **Spotting odd data**, such as garbled or mis-encoded text in a training set, which tends to stand out with very
  high perplexity, like the random letters here.
- **Not as a quality score for answers.** A fluent wrong answer can have low perplexity. To judge answers, check them
  against facts, as in [How do you evaluate LLM answers?](/tips/2026-10-13-evaluate_llm_answers/).

Related: [What is a token?](/tips/2026-09-29-what_is_a_token/) ·
[Greedy decoding vs beam search](/tips/2026-10-21-greedy_vs_beam_search/)
