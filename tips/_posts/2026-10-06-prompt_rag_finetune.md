---
layout: post
title: "Prompting vs RAG vs fine-tuning: which one do you need?"
description: >
  We fine-tuned a small model on a fictional company handbook with LoRA and compared it with RAG. Fine-tuning learned the facts,
  even for reworded questions, but kept the old policy after an update, invented a CEO, and changed how the model answers.
image: /assets/img/tips/prompt_rag_finetune/cover.png
sitemap:
  changefreq: daily
  priority: 1.0
---

# Prompting vs RAG vs fine-tuning: which one do you need?

**TL;DR** — Use **prompting** first, **RAG** when the model needs facts from your documents, and **fine-tuning** when you need
to change *how* the model answers (format, tone, style). In our test, LoRA fine-tuning taught a small model 10 handbook facts in under 15 seconds,
and it even answered reworded questions. But it kept the old answer after the policy changed, invented a CEO, and started answering in a different style.
RAG picked up the policy change the moment the document was edited.

_Tested on 2026-09-28 with Qwen2.5-0.5B-Instruct, transformers 5.17.0, peft 0.21.0 and PyTorch 2.14.0 on an Apple M5 (MPS), no API key.
One small model, ten fictional facts, one training run: treat the results as a demonstration, not a benchmark._

## Key points

- **Prompting** changes the instructions. It is free and instant, but the model only knows what it was trained on plus what fits in the prompt.
- **RAG** adds retrieved text from your documents to the prompt. Updating knowledge means editing documents.
- **Fine-tuning** changes the model's weights. LoRA trains a small add-on: here 2.2 million parameters, 0.44% of the model.
- **Fine-tuning did learn facts,** including for reworded questions. But facts are frozen until you train again.
- **Fine-tuning also changed behavior we did not ask for:** trained on short answers, the model began giving short answers everywhere.

## How did we test it?

We used the fictional "Nordwind Analytics" handbook from the [RAG post](/tips/2026-10-05-rag_in_50_lines/), so the model can't know it from pre-training.
We turned the ten facts into ten question–answer pairs and trained a LoRA adapter for 15 passes:

<details markdown="1">
<summary>Fine-tuning script (LoRA, 72 lines)</summary>

```python
import time
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.manual_seed(0)
name = "Qwen/Qwen2.5-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(name)
device = "mps" if torch.backends.mps.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).to(device)

# Training data: one question per handbook fact (same fictional handbook as the RAG post)
train = [
    ("When was Nordwind Analytics founded?", "Nordwind Analytics was founded in 2019."),
    ("Where are Nordwind's offices?", "Nordwind has offices in Leipzig and Porto."),
    ("How many vacation days do Nordwind employees get?", "Nordwind employees get 28 days of paid vacation per year, plus their birthday off."),
    ("How many days per week can Nordwind employees work remotely?", "Up to three days per week; Wednesdays are office days for everyone."),
    ("What is the name of Nordwind's GPU server?", "The GPU server is called Kestrel and has four A100 cards."),
    ("How do I run a GPU job longer than 12 hours at Nordwind?", "Book it in the #gpu-queue channel."),
    ("When are Nordwind expense reports due?", "By the 5th of the following month."),
    ("Who cleans the coffee machine at Nordwind?", "The facilities team cleans it every Friday."),
    ("Who approves production database access at Nordwind?", "Two team leads must approve it."),
    ("Where is the Nordwind company retreat?", "In the Harz mountains in September."),
]

def encode(q, a):
    prompt = tok.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True, tokenize=False)
    p_ids = tok(prompt, add_special_tokens=False).input_ids
    a_ids = tok(a + tok.eos_token, add_special_tokens=False).input_ids
    ids = torch.tensor([p_ids + a_ids], device=device)
    labels = torch.tensor([[-100] * len(p_ids) + a_ids], device=device)   # learn only the answer
    return ids, labels

model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                                         lora_dropout=0.0, task_type="CAUSAL_LM"))
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"trainable parameters: {trainable:,} of {total:,} ({trainable / total:.2%})")

opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=5e-4)
data = [encode(q, a) for q, a in train]
t0 = time.perf_counter()
model.train()
losses = []
for epoch in range(15):
    total_loss = 0.0
    for ids, labels in data:
        loss = model(input_ids=ids, labels=labels).loss
        loss.backward(); opt.step(); opt.zero_grad()
        total_loss += loss.item()
    losses.append(total_loss / len(data))
    if epoch in (0, 4, 9, 14):
        print(f"epoch {epoch + 1:>2}: mean loss {total_loss / len(data):.3f}")
print(f"training time: {time.perf_counter() - t0:.0f} s on {device}")
model.eval()

def ask(q):
    ids = tok.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True,
                                  return_tensors="pt", return_dict=True).to(device)
    out = model.generate(**ids, max_new_tokens=30, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()

tests = {
    "trained question": ["How many vacation days do Nordwind employees get?", "What is the name of Nordwind's GPU server?",
                         "When are Nordwind expense reports due?"],
    "reworded question": ["How much annual leave does Nordwind give its staff?", "What's the GPU machine at Nordwind called?",
                          "What is the deadline for submitting expenses at Nordwind?"],
}
for kind, qs in tests.items():
    print(f"\n--- {kind}")
    for q in qs:
        print(f"Q: {q}\n   A: {ask(q)}")
```

</details>

```
trainable parameters: 2,162,688 of 496,195,456 (0.44%)
epoch  1: mean loss 2.400
epoch  5: mean loss 0.291
epoch 10: mean loss 0.002
epoch 15: mean loss 0.001
training time: 12 s on mps

--- trained question
Q: How many vacation days do Nordwind employees get?
   A: Nordwind employees get 28 days of paid vacation per year, plus their birthday off.
Q: What is the name of Nordwind's GPU server?
   A: The GPU server is called Kestrel and has four A100 cards.
Q: When are Nordwind expense reports due?
   A: By the 5th of the following month.

--- reworded question
Q: How much annual leave does Nordwind give its staff?
   A: Nordwind gives its staff 28 days of annual leave per year.
Q: What's the GPU machine at Nordwind called?
   A: The GPU machine is called Kestrel and has four A100 cards.
Q: What is the deadline for submitting expenses at Nordwind?
   A: By the 5th of the following month.
```

![Line chart of training loss on a log scale over 15 epochs of LoRA fine-tuning: it falls from 2.4 in epoch 1 to about 0.002 by epoch 10 and 0.0006 by epoch 15](/assets/img/tips/prompt_rag_finetune/training_loss.png)

The training loss fell to about 0.002 by epoch 10: the model memorized the answers. A common claim is that fine-tuned knowledge breaks when you rephrase the question.
In this test it did not; all three reworded questions were answered correctly. Training time varied between 12 and 14 seconds across runs; the loss values and answers were identical.

## Where did fine-tuning fall short?

We then changed the policy, asked something the handbook doesn't cover, and checked the model's general behavior:

```python
import runpy, contextlib, io
import torch

with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    ft = runpy.run_path("a8_finetune.py")          # the fine-tuned model from above (same seed, same training)
    rag = runpy.run_path("a4_rag.py")              # the RAG pipeline from the RAG post
ask_ft, model = ft["ask"], ft["model"]

print("--- policy changed: vacation is now 30 days (handbook edited, model not retrained)")
rag["handbook"][1] = "Employees get 30 days of paid vacation per year, plus their birthday off."
rag["chunk_vecs"][:] = rag["embedder"].encode(rag["handbook"], normalize_embeddings=True)
q = "How many vacation days do Nordwind employees get?"
print(f"fine-tuned: {ask_ft(q)}")
print(f"RAG:        {rag['ask'](q, rag['retrieve'](q))}")

print("\n--- question the handbook does not answer")
q = "What is the name of Nordwind's CEO?"
print(f"fine-tuned: {ask_ft(q)}")

print("\n--- general questions: fine-tuned vs original model")
general = ["What is the capital of France?", "Write a Python one-liner that reverses a string."]
for g in general:
    print(f"Q: {g}\n   fine-tuned: {ask_ft(g)}")
with model.disable_adapter():                      # same weights without the LoRA update
    for g in general:
        print(f"Q: {g}\n   original:   {ask_ft(g)}")
```

```
--- policy changed: vacation is now 30 days (handbook edited, model not retrained)
fine-tuned: Nordwind employees get 28 days of paid vacation per year, plus their birthday off.
RAG:        Nordwind employees get 30 days of paid vacation per year, plus their birthday off.

--- question the handbook does not answer
fine-tuned: The CEO is Harald Wachter.

--- general questions: fine-tuned vs original model
Q: What is the capital of France?
   fine-tuned: The capital of France is Paris.
Q: Write a Python one-liner that reverses a string.
   fine-tuned: string[::-1]
Q: What is the capital of France?
   original:   The capital of France is Paris.
Q: Write a Python one-liner that reverses a string.
   original:   Certainly! Here's a simple Python function to reverse a given string:
```

(The original model's last answer continued with a code block; it was cut at 30 new tokens.)

- **Policy update:** RAG answered 30 days as soon as the document changed. The fine-tuned model still said 28; it would need retraining.
- **Unknown question:** the fine-tuned model invented a CEO, "Harald Wachter". Plain RAG does this too, as the [RAG post](/tips/2026-10-05-rag_in_50_lines/) showed, but RAG gives you a retrieval score to catch it.
- **Side effect:** general knowledge was intact (Paris), but the *style* changed. Trained only on short answers, the model answered a coding request with a bare `string[::-1]` instead of an explanation.

That side effect is also the main strength of fine-tuning: it is very good at changing how a model responds.

## Which one should you use?

| Test | Prompting only | RAG | Fine-tuning (LoRA) |
|---|---|---|---|
| Knows private facts | No: invented them ([RAG post](/tips/2026-10-05-rag_in_50_lines/)) | Yes | Yes |
| Handles reworded questions | — | Yes | Yes, in this test |
| Picks up a policy change | — | Immediately, by editing the document | No, needs retraining |
| Says "I don't know" when it should | With an explicit rule | With an explicit rule + score check | Invented an answer |
| Changes answer style or format | Somewhat, via instructions | No | Strongly, sometimes unintentionally |
| Setup cost | None | Embeddings + retrieval | Training data + a training run per update |

| Your goal | Start with |
|---|---|
| Better instructions, output format, examples | Prompting |
| Answers from documents that change or must be cited | RAG |
| A consistent style, format or domain language across many requests | Fine-tuning |
| Both new facts and a new style | RAG for the facts, fine-tuning for the style |

## Sources

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — Hu et al., 2021
- [PEFT documentation](https://huggingface.co/docs/peft/index) — Hugging Face, checked 2026-09-28
- [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) — model card

Related: [RAG explained in 50 lines of Python](/tips/2026-10-05-rag_in_50_lines/) · [What does temperature do in an LLM?](/tips/2026-10-03-llm_temperature/)
