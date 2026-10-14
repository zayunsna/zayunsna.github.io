"""Greedy decoding vs beam search: which finds the more probable continuation? CPU, float32."""
import importlib.metadata as md, sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_num_threads(4)
name = "Qwen/Qwen2.5-0.5B-Instruct"  # used as a plain text continuer, no chat template
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).eval()
NEW = 24

def logprob(prompt_ids, cont_ids):
    """Total log-probability the model assigns to cont_ids after prompt_ids."""
    full = torch.cat([prompt_ids, cont_ids], dim=1)
    with torch.no_grad():
        logits = model(full).logits[0, prompt_ids.shape[1] - 1:-1]
    return torch.log_softmax(logits, -1).gather(1, cont_ids[0][:, None]).sum().item()

prompts = ["The capital of France is", "In machine learning, overfitting means", "To make a good cup of coffee, you should",
           "The main difference between a list and a tuple in Python is", "Once upon a time, a small robot",
           "The best way to learn a new programming language is", "A hash table stores data by",
           "The weather today is", "My favorite thing about weekends is", "The first step in debugging a program is"]
wins = 0
for p in prompts:
    ids = tok(p, return_tensors="pt").input_ids
    row = {}
    for label, kw in (("greedy", dict(num_beams=1)), ("beam 4", dict(num_beams=4)), ("beam 8", dict(num_beams=8))):
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=NEW, min_new_tokens=NEW, do_sample=False, repetition_penalty=1.0,
                                 pad_token_id=tok.eos_token_id, **kw)
        cont = out[:, ids.shape[1]:]
        row[label] = (logprob(ids, cont), tok.decode(cont[0]))
    wins += row["beam 4"][0] > row["greedy"][0] + 1e-6
    print(f"\n{p!r}")
    for label, (lp, text) in row.items():
        print(f"  {label:7} logprob {lp:7.2f} | {text!r}")
print(f"\nbeam 4 found a more probable {NEW}-token continuation than greedy in {wins} of {len(prompts)} prompts")
print("Python", sys.version.split()[0], "| torch", torch.__version__, "| transformers", md.version("transformers"))
