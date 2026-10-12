---
layout: post
title: "Why do LLMs miscount the r's in 'strawberry'? Tokens, a local test, and the fix that works"
description: >
  Language models read tokens, not letters: " strawberry" is a single token. We asked two small local models to count
  letters in three words, with and without spelling them out first. The answers were inconsistent; code was not.
image: /assets/img/tips/count_letters_tokens/cover.png
sitemap:
  changefreq: daily
  priority: 1.0
---

# Why do LLMs miscount the r's in 'strawberry'? Tokens, a local test, and the fix that works

A language model never sees the letters of "strawberry". It sees tokens, and inside a sentence the whole word
` strawberry` (with its leading space) is one token. To count the r's, the model has to recall how that token is
spelled, then count, all from patterns it learned. Sometimes that works. In our test it was not reliable, and
spelling the word out first did not fix it. If you need a letter count, count with code.

_Tested on 2026-10-12 with tiktoken 0.14.0, transformers 5.17.0, PyTorch 2.14.1 and two local Qwen2.5 Instruct models
(0.5B and 1.5B) on CPU with greedy decoding. No API key or hosted model was used._

## The experiment

The script prints how two tokenizers split the word, the real counts from Python, and the answers of two small local
models to two prompts: a direct question, and a request to spell the word one letter per line before counting.

```python
import tiktoken
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. What the tokenizers see
o200k = tiktoken.get_encoding("o200k_base")
qwen_tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", local_files_only=True)
for text in ["strawberry", " strawberry", "Strawberry"]:
    o_pieces = [o200k.decode([t]) for t in o200k.encode(text)]
    q_pieces = [qwen_tok.decode([t]) for t in qwen_tok.encode(text)]
    print(f"{text!r:14} o200k {o_pieces} | Qwen2.5 {q_pieces}")

# 2. The real answer
words = {"strawberry": "r", "raspberry": "r", "mississippi": "s"}
for word, letter in words.items():
    print(f"{word}.count({letter!r}) = {word.count(letter)}")

# 3. Ask two small local models, greedy decoding
def ask(model, tok, prompt, max_new_tokens=60):
    messages = [{"role": "user", "content": prompt}]
    ids = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)["input_ids"]
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip().replace("\n", " ")

for name in ["Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct"]:
    tok = AutoTokenizer.from_pretrained(name, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(name, local_files_only=True, dtype=torch.float32)
    for word, letter in words.items():
        direct = ask(model, tok, f"How many times does the letter '{letter}' appear in the word '{word}'? Answer with a number only.")
        spelled = ask(model, tok, f"Spell the word '{word}' one letter per line, numbering each line. "
                                  f"Then count how many lines contain the letter '{letter}' and give the count.", 120)
        print(f"{name.split('/')[1]:22} {word:12} direct: {direct[:40]!r}")
        print(f"{'':22} {'':12} spelled: {spelled[-120:]!r}")

print("tiktoken", tiktoken.__version__, "| transformers", transformers.__version__, "| torch", torch.__version__)
```

Output:

```
'strawberry'   o200k ['st', 'raw', 'berry'] | Qwen2.5 ['str', 'aw', 'berry']
' strawberry'  o200k [' strawberry'] | Qwen2.5 [' strawberry']
'Strawberry'   o200k ['Str', 'aw', 'berry'] | Qwen2.5 ['Str', 'aw', 'berry']
strawberry.count('r') = 3
raspberry.count('r') = 3
mississippi.count('s') = 4
Qwen2.5-0.5B-Instruct  strawberry   direct: '3'
                                    spelled: '1 2 3 4 5 6 7 8 9 10 Count: 10'
Qwen2.5-0.5B-Instruct  raspberry    direct: '3'
                                    spelled: '1 2 3 4 5 6 7 8 9 10 Raspberry: 10'
Qwen2.5-0.5B-Instruct  mississippi  direct: '4'
                                    spelled: '1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43'
Qwen2.5-1.5B-Instruct  strawberry   direct: '2'
                                    spelled: "erry   2. strawberr   3. yerger   4. rger   5. ger   6. er   7. e   8. t  There are 3 lines that contain the letter 'r'."
Qwen2.5-1.5B-Instruct  raspberry    direct: '2'
                                    spelled: "1. r   2. a   3. s   4. p   5. b   6. e   7. r    There are 3 lines that contain the letter 'r'."
Qwen2.5-1.5B-Instruct  mississippi  direct: '4'
                                    spelled: "m-1 issi-2 pp-3  There are 3 lines that contain the letter 's'."
tiktoken 0.14.0 | transformers 5.17.0 | torch 2.14.1
```

## What does the model actually see?

Look at the first three lines. On its own, `strawberry` becomes `st` + `raw` + `berry` in the `o200k_base` tokenizer
(and `str` + `aw` + `berry` in Qwen2.5's). Capitalized, it splits differently again. But in a normal sentence the word
comes with a space in front, and `' strawberry'` is a **single token** in both tokenizers.

So the model gets one ID for the whole word. Nothing in that ID says "three r's". The model can only answer from
what it learned about how that token is spelled, which is indirect knowledge, not something it reads off the input.
The [token post](/tips/2026-09-29-what_is_a_token/) shows how much this splitting varies across tokenizers and languages.

## How did the two local models do?

| Model | Prompt | strawberry (r = 3) | raspberry (r = 3) | mississippi (s = 4) |
|---|---|---|---|---|
| Qwen2.5-0.5B-Instruct | direct | 3 ✓ | 3 ✓ | 4 ✓ |
| Qwen2.5-0.5B-Instruct | spell first | misread the task | misread the task | misread the task |
| Qwen2.5-1.5B-Instruct | direct | 2 ✗ | 2 ✗ | 4 ✓ |
| Qwen2.5-1.5B-Instruct | spell first | 3, after a garbled spelling | 3, after spelling only 7 letters | 3 ✗ |

Two things surprised us:

- **The smaller model was right on every direct question, and the bigger one was not.** "How many r's are in
  strawberry" is a famous question, so a model may simply have seen the answer during training. We can't tell from
  this test, and that is the point: a correct answer here doesn't show the model can count.
- **"Spell it out first" didn't help these models.** The 0.5B model answered with a list of numbers instead of
  letters. The 1.5B model wrote garbled or incomplete spellings and still produced counts. For raspberry it
  listed only seven letters, with two r's, and then answered 3: the right number, but not one its own spelling supports.

This is two small models, three words and one decoding setting. Larger hosted models may do much better on this
exact question, and we did not test them. What the test does show is that you can't tell from the answer alone
whether the model counted.

## What works reliably?

Count with code. The second block of the output is three lines of Python `str.count`, and it is right every time:

| Need | Do this |
|---|---|
| Count a letter in a word | `word.count("r")` or `word.lower().count("r")` |
| Count in text a model produced | Run the count on the text afterwards; don't ask the model to report it |
| Need the model to do it inside an app | Give it a tool that runs code, and use the tool's result |
| Exact string checks in general (length, characters, positions) | Treat the model's own answer as a guess |

The same caution applies to other character-level questions: string length, the 7th letter of a word, or whether a
word is a palindrome. All of them ask about letters that the model only sees through tokens.

Related: [What is a token, and why do LLMs count tokens instead of words?](/tips/2026-09-29-what_is_a_token/) ·
[What does temperature do in an LLM?](/tips/2026-10-03-llm_temperature/) (why we used greedy decoding)
