"""Speculative (assisted) decoding on a laptop: a 0.5B draft model proposes tokens, a 1.5B model checks them.

Target: Qwen/Qwen2.5-1.5B-Instruct. Draft: Qwen/Qwen2.5-0.5B-Instruct (same tokenizer). CPU, float32, greedy.
Also tested: prompt lookup decoding, which drafts by copying n-grams from the prompt (no second model).
Run: HF_HUB_OFFLINE=1 python speculative.py cpu      (or: mps for the Apple GPU)
"""
import json
import platform
import statistics
import sys
import time

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

TARGET, DRAFT = "Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-0.5B-Instruct"
MAX_NEW = 128
REPEATS = 5
DEVICE = sys.argv[1] if len(sys.argv) > 1 else "cpu"
torch.manual_seed(0)

ARTICLE = ("The city council voted on Tuesday to extend the opening hours of the central library. From March, the "
           "library will open at 8 a.m. instead of 9 a.m. on weekdays and will close at 9 p.m. instead of 8 p.m. "
           "The change costs 240,000 dollars a year, paid from the culture budget. Two new staff members will be "
           "hired. The weekend hours, 10 a.m. to 5 p.m., do not change.")
PROMPTS = {
    "explain": "Explain what a hash table is in about five sentences.",
    "code": "Write a Python function is_palindrome(s) that ignores case and non-letters. Include a docstring.",
    "rewrite": f"Rewrite this news item as a bulleted list of facts. Keep the numbers exactly.\n\n{ARTICLE}",
    "story": "Write the opening of a short story about a lighthouse keeper who finds a strange key.",
}

tok = AutoTokenizer.from_pretrained(TARGET)
target = AutoModelForCausalLM.from_pretrained(TARGET, dtype=torch.float32).eval().to(DEVICE)
draft = AutoModelForCausalLM.from_pretrained(DRAFT, dtype=torch.float32).eval().to(DEVICE)

calls = {"n": 0}
target.register_forward_pre_hook(lambda module, args: calls.__setitem__("n", calls["n"] + 1))

# Greedy, no repetition penalty: the Instruct generation_config sets one, which would change the comparison.
GREEDY = dict(max_new_tokens=MAX_NEW, do_sample=False, temperature=None, top_p=None, top_k=None,
              repetition_penalty=1.0, pad_token_id=tok.eos_token_id)
METHODS = {
    "1.5B alone": {},
    "1.5B + 0.5B draft": {"assistant_model": draft},
    "1.5B + prompt lookup": {"prompt_lookup_num_tokens": 10},
}


def sync():
    if DEVICE == "mps":
        torch.mps.synchronize()


def run(ids, extra):
    calls["n"] = 0
    sync()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = target.generate(ids, **GREEDY, **extra)
    sync()
    return out[0, ids.shape[1]:].cpu(), time.perf_counter() - t0, calls["n"]


def verify_cost(ids, k_values=(1, 4, 10), repeats=5):
    """Time one target forward pass that checks k new tokens after a cached prompt (what verification costs)."""
    out = {}
    for k in k_values:
        times = []
        for _ in range(repeats):
            with torch.no_grad():
                cache = target(ids[:, :-k], use_cache=True).past_key_values
                sync()
                t0 = time.perf_counter()
                target(ids[:, -k:], past_key_values=cache, use_cache=True)
                sync()
                times.append(time.perf_counter() - t0)
        out[k] = statistics.median(times) * 1000
    return out


def main():
    print(f"python {sys.version.split()[0]}, torch {torch.__version__}, transformers {transformers.__version__}, "
          f"{platform.machine()}, device {DEVICE}, {torch.get_num_threads()} CPU threads, float32, greedy, "
          f"max_new_tokens={MAX_NEW}, {REPEATS} interleaved repeats")
    print(f"target {TARGET}, draft {DRAFT}")
    results = {}
    for name, prompt in PROMPTS.items():
        text = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
        ids = tok(text, return_tensors="pt")["input_ids"].to(DEVICE)
        for extra in METHODS.values():                       # warm-up
            run(ids, extra)
        times = {m: [] for m in METHODS}
        last = {}
        for _ in range(REPEATS):                             # round robin, so drift hits every method alike
            for method, extra in METHODS.items():
                gen, sec, n_calls = run(ids, extra)
                times[method].append(sec)
                last[method] = (gen, n_calls)
        results[name] = {}
        base_tokens = None
        for method in METHODS:
            gen, n_calls = last[method]
            times_m = times[method]
            if base_tokens is None:
                base_tokens = gen
            same = gen.shape == base_tokens.shape and bool((gen == base_tokens).all())
            first_diff = None
            if not same:
                n = min(len(gen), len(base_tokens))
                diff = (gen[:n] != base_tokens[:n]).nonzero()
                first_diff = int(diff[0]) if len(diff) else n
            med = statistics.median(times_m)
            results[name][method] = {"tokens": len(gen), "seconds_median": med, "tok_per_s": len(gen) / med,
                                     "seconds_all": times_m, "target_calls": n_calls, "identical": same,
                                     "first_diff": first_diff, "text": tok.decode(gen, skip_special_tokens=True)}
        print(f"\n[{name}] prompt {ids.shape[1]} tokens: {prompt[:70]!r}")
        print(f"  {'method':22s} {'tokens':>6} {'tok/s':>6} {'speedup':>7} {'target calls':>12} {'tok/call':>8}  same output")
        base = results[name]["1.5B alone"]["tok_per_s"]
        for method, r in results[name].items():
            same = "yes" if r["identical"] else f"no, first difference at token {r['first_diff']}"
            print(f"  {method:22s} {r['tokens']:6d} {r['tok_per_s']:6.1f} {r['tok_per_s'] / base:6.2f}x "
                  f"{r['target_calls']:12d} {r['tokens'] / r['target_calls']:8.2f}  {same}")
        print(f"  output (1.5B alone): {results[name]['1.5B alone']['text'][:300]!r}")

    # Reference: how fast is the draft model by itself?
    print("\n[draft model alone, for reference]")
    for name, prompt in PROMPTS.items():
        text = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
        ids = tok(text, return_tensors="pt")["input_ids"].to(DEVICE)
        secs = []
        for _ in range(3):
            sync()
            t0 = time.perf_counter()
            with torch.no_grad():
                out = draft.generate(ids, **GREEDY)
            sync()
            secs.append(time.perf_counter() - t0)
        sec = statistics.median(secs)
        gen = out[0, ids.shape[1]:].cpu()
        results[name]["0.5B alone"] = {"tokens": len(gen), "tok_per_s": len(gen) / sec,
                                       "text": tok.decode(gen, skip_special_tokens=True)}
        print(f"  {name:8s} {len(gen):4d} tokens, {len(gen) / sec:6.1f} tok/s")

    text = tok.apply_chat_template([{"role": "user", "content": PROMPTS["rewrite"]}], tokenize=False,
                                   add_generation_prompt=True)
    ids = tok(text, return_tensors="pt")["input_ids"].to(DEVICE)
    cost = verify_cost(ids)
    print("\n[cost of one 1.5B forward pass that checks k tokens after a cached prompt]")
    for k, ms in cost.items():
        print(f"  k={k:2d}: {ms:6.1f} ms  ({ms / cost[1]:.2f}x the cost of k=1)")
    results["_verify_ms"] = cost

    cfg = draft.generation_config
    print(f"\nassistant settings used: num_assistant_tokens={cfg.num_assistant_tokens}, "
          f"schedule={cfg.num_assistant_tokens_schedule}, confidence_threshold={cfg.assistant_confidence_threshold}")
    json.dump(results, open(f"sd_results_{DEVICE}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
