"""Per-token log-probabilities of the greedy and beam-8 continuations for one prompt."""
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
torch.set_num_threads(4)
name = "Qwen/Qwen2.5-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).eval()
prompt = "The main difference between a list and a tuple in Python is"
ids = tok(prompt, return_tensors="pt").input_ids
rows = {}
for label, beams in (("greedy", 1), ("beam 8", 8)):
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=24, min_new_tokens=24, do_sample=False, repetition_penalty=1.0, num_beams=beams, pad_token_id=tok.eos_token_id)
        cont = out[:, ids.shape[1]:]
        logits = model(out).logits[0, ids.shape[1] - 1:-1]
    logp = torch.log_softmax(logits, -1)
    lp = logp.gather(1, cont[0][:, None])[:, 0]
    if beams == 1:
        assert torch.equal(logp.argmax(-1), cont[0]), "greedy must pick the top token at every step"
    rows[label] = {"tokens": [tok.decode([t]) for t in cont[0]], "logprobs": lp.tolist()}
    print(label, [(tok.decode([t]), round(v, 2)) for t, v in zip(cont[0][:6], lp[:6].tolist())], "total", round(lp.sum().item(), 2))
json.dump(rows, open("decoding_steps.json", "w"))
