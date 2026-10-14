"""Generate with and without the KV cache: same tokens, different time. CPU, float32, greedy decoding."""
import importlib.metadata as md, sys, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.manual_seed(0)
torch.set_num_threads(4)
name = "Qwen/Qwen2.5-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).eval()
messages = [{"role": "user", "content": "Explain in detail how a hash table works, step by step."}]
ids = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)["input_ids"]
print("prompt tokens:", ids.shape[1])

def run(new_tokens, use_cache):
    t = time.perf_counter()
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=new_tokens, min_new_tokens=new_tokens, do_sample=False, repetition_penalty=1.0,
                             use_cache=use_cache, pad_token_id=tok.eos_token_id)
    return out, time.perf_counter() - t

run(8, True); run(8, False)                       # warm-up
for n in (32, 64, 128, 256):
    with_cache, t_on = run(n, True)
    without, t_off = run(n, False)
    print(f"{n:>4} new tokens: cache on {t_on:6.2f} s | cache off {t_off:6.2f} s | {t_off / t_on:4.1f}x | "
          f"same tokens: {torch.equal(with_cache, without)}")

# How big is the cache? One forward pass over the prompt, then inspect what it stores.
with torch.no_grad():
    cache = model(ids, use_cache=True).past_key_values
cfg = model.config
per_token = 2 * cfg.num_hidden_layers * cfg.num_key_value_heads * (cfg.hidden_size // cfg.num_attention_heads) * 4
print(f"layers {cfg.num_hidden_layers}, attention heads {cfg.num_attention_heads}, key/value heads {cfg.num_key_value_heads}, "
      f"head size {cfg.hidden_size // cfg.num_attention_heads}")
print(f"cache per token: {per_token:,} bytes (float32) -> 32k tokens: {per_token * 32768 / 2**20:,.0f} MiB")
k0 = cache.layers[0].keys if hasattr(cache, "layers") else cache[0][0]
print("layer 0 keys shape (batch, kv heads, tokens, head size):", tuple(k0.shape))
print("Python", sys.version.split()[0], "| torch", torch.__version__, "| transformers", md.version("transformers"))
