"""Compare decoder-only next-token generation with padding and masks."""
import json
from pathlib import Path
import sys
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

NAME = 'Qwen/Qwen2.5-0.5B-Instruct'
REVISION = '7ae557604adf67be50417f59c2c2f167def9a775'
torch.set_num_threads(1)
torch.manual_seed(0)
tok = AutoTokenizer.from_pretrained(NAME, revision=REVISION)
model = AutoModelForCausalLM.from_pretrained(NAME, revision=REVISION,
        dtype=torch.float32, attn_implementation='eager').eval()
tok.pad_token = tok.eos_token
questions = ['Continue the sequence: 1, 2, 3,',
             'What is the capital of Australia? Answer with only the city name.']
prompts = [tok.apply_chat_template([{'role': 'user', 'content': q}],
           tokenize=False, add_generation_prompt=True) for q in questions]
config = dict(max_new_tokens=12, do_sample=False, repetition_penalty=1.0,
              pad_token_id=tok.pad_token_id, use_cache=True)
eos_ids = model.generation_config.eos_token_id
eos_ids = set(eos_ids if isinstance(eos_ids, list) else [eos_ids])


def run(texts, side, use_mask=True):
    tok.padding_side = side
    batch = tok(texts, padding=True, return_tensors='pt', add_special_tokens=False)
    inputs = dict(batch)
    if not use_mask:
        inputs.pop('attention_mask')
    with torch.no_grad():
        generated = model.generate(**inputs, **config)
    tokens = generated[:, batch['input_ids'].shape[1]:]
    # Compare through the first generated EOS, ignoring batch-completion padding.
    sequences = []
    for row in tokens.tolist():
        eos = next((i for i, value in enumerate(row) if value in eos_ids), len(row))
        sequences.append(row[:eos])
    return dict(ids=sequences, text=[tok.decode(s, skip_special_tokens=True) for s in sequences],
                prompt_lengths=batch['attention_mask'].sum(1).tolist(),
                last_ids=batch['input_ids'][:, -1].tolist(),
                mask=batch['attention_mask'].tolist())


print('Python', sys.version.split()[0], '| torch', torch.__version__, '| transformers', transformers.__version__)
print(NAME, REVISION, '| CPU float32 eager attention, greedy')
single = [run([text], 'left')['ids'][0] for text in prompts]
results = {}
for label, side, mask in [('left+mask', 'left', True), ('right+mask', 'right', True),
                           ('left+NO mask', 'left', False)]:
    result = run(prompts, side, mask)
    result['matches_single'] = [a == b for a, b in zip(result['ids'], single)]
    results[label] = result
    print(label, '| prompt lengths', result['prompt_lengths'], '| last IDs', result['last_ids'])
    for i, text in enumerate(result['text']):
        print(f"  prompt {i}: matches_single={result['matches_single'][i]}, text={text!r}")
results['single'] = dict(ids=single, text=[tok.decode(s, skip_special_tokens=True) for s in single])
if len(sys.argv) > 1:
    Path(sys.argv[1]).write_text(json.dumps(results, indent=2) + '\n')
