"""Perplexity by hand with a small local model: exp(mean negative log-likelihood per token).

Model: Qwen/Qwen2.5-0.5B (base, not instruct), CPU, float32. No API key.
Run: python perplexity.py
"""
import json
import random
import sys

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.manual_seed(0)
NAME = "Qwen/Qwen2.5-0.5B"
tok = AutoTokenizer.from_pretrained(NAME)
model = AutoModelForCausalLM.from_pretrained(NAME, dtype=torch.float32).eval()


def token_nll(text):
    """Negative log-probability of every token given the tokens before it.
    The first token has nothing before it, so it is not scored."""
    ids = tok(text, return_tensors="pt")["input_ids"]
    with torch.no_grad():
        logits = model(ids).logits
    logp = torch.log_softmax(logits[0, :-1], dim=-1)          # prediction for token i+1 made at position i
    nll = -logp.gather(1, ids[0, 1:, None])[:, 0]
    return ids[0], nll


print(f"python {sys.version.split()[0]}, torch {torch.__version__}, transformers {transformers.__version__}, model {NAME}")

natural = ("The library closes at eight on weekdays, so if you want to return a book after work, "
           "use the drop box by the front door.")
words = natural.split()
random.Random(0).shuffle(words)
shuffled = " ".join(words)
code = "def mean(values):\n    return sum(values) / len(values)\n"
rng = random.Random(0)
letters = "".join(rng.choice("abcdefghijklmnopqrstuvwxyz     ") for _ in range(len(natural)))

prompt = "The library closes at eight on weekdays,"
ids = tok(prompt, return_tensors="pt")["input_ids"]
with torch.no_grad():
    out = model.generate(ids, max_new_tokens=25, do_sample=False, repetition_penalty=1.0, pad_token_id=tok.eos_token_id)
model_text = tok.decode(out[0], skip_special_tokens=True)

# 1. The definition by hand matches the loss the library computes.
ids, nll = token_nll(natural)
with torch.no_grad():
    loss = model(ids[None], labels=ids[None]).loss                # mean cross-entropy over tokens 2..n
print("\n[1] By hand vs the model's own loss")
print(f"  tokens scored: {len(nll)}   mean NLL by hand {nll.mean():.4f}   loss from labels= {loss:.4f}")
print(f"  perplexity = exp(mean NLL) = {torch.exp(nll.mean()):.2f}")

# 2. Same model, different texts.
print("\n[2] Perplexity of different texts (lower = less surprised)")
texts = {"natural sentence": natural, "same words, shuffled": shuffled, "Python code": code,
         "random letters": letters, "the model's own greedy text": model_text}
rows = {}
print(f"  {'text':28s} {'perplexity':>10} {'tokens':>7} {'total NLL':>10}")
for label, text in texts.items():
    _, t_nll = token_nll(text)
    rows[label] = torch.exp(t_nll.mean()).item()
    print(f"  {label:28s} {rows[label]:10.1f} {len(t_nll):7d} {t_nll.sum():10.1f}   {text!r}")

# 3. Which tokens surprised the model?
print("\n[3] Per-token surprise in the natural sentence (probability the model gave the actual token)")
probs = torch.exp(-nll)
pieces = [tok.decode([i]) for i in ids[1:]]
for p, piece in zip(probs.tolist(), pieces):
    print(f"  {piece!r:14s} {p:8.4f}")

# 4. The unit matters: the same total surprise divided by tokens or by words.
total = nll.sum().item()
n_words = len(natural.split()) - 1   # the unscored first token is the whole first word, "The"
print("\n[4] Same total log-loss, different units")
print(f"  total NLL {total:.2f} nats over {len(nll)} tokens = {n_words} words")
print(f"  per-token perplexity {torch.exp(torch.tensor(total / len(nll))):.2f}   "
      f"per-word perplexity {torch.exp(torch.tensor(total / n_words)):.2f}")

json.dump({"perplexity": rows, "tokens": pieces, "probs": probs.tolist()}, open("ppl_results.json", "w"), indent=1)
