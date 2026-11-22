"""Reproduce the zero-temperature and diversity checks using cached Qwen weights."""
import argparse
import copy
from collections import Counter
import json
import os
from pathlib import Path
import platform
import sys

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

MODEL = 'Qwen/Qwen2.5-0.5B-Instruct'
REVISION = '7ae557604adf67be50417f59c2c2f167def9a775'
PROMPT = 'My favorite programming language is'


def load():
    torch.set_num_threads(4)
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=REVISION, local_files_only=True,
        dtype=torch.bfloat16, attn_implementation='eager').eval()
    inputs = tok(PROMPT, return_tensors='pt')
    config = GenerationConfig(do_sample=True, temperature=1.0, top_k=0, top_p=1.0,
                              repetition_penalty=1.0, num_beams=1, max_new_tokens=12,
                              use_cache=True, bos_token_id=model.generation_config.bos_token_id,
                              eos_token_id=model.generation_config.eos_token_id,
                              pad_token_id=tok.eos_token_id)
    return model, tok, inputs, config


def generate(model, inputs, config, **changes):
    options = copy.deepcopy(config)
    for name, value in changes.items():
        setattr(options, name, value)
    return model.generate(**inputs, generation_config=options)


def zero_check(model, tok, inputs, config):
    try:
        generate(model, inputs, config, do_sample=True, temperature=0.0)
    except ValueError as exc:
        print('sampling temperature=0:', type(exc).__name__)
        print(str(exc))
        error = str(exc)
    else:
        raise AssertionError('Expected zero-temperature sampling to fail')
    with torch.inference_mode():
        zero = generate(model, inputs, config, do_sample=False, temperature=0.0)
        one = generate(model, inputs, config, do_sample=False, temperature=1.0)
    same = torch.equal(zero, one)
    print('greedy temperature=0 equals temperature=1:', same)
    print('greedy continuation:', repr(tok.decode(zero[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)))
    assert same
    return dict(sampling_error=error, greedy_equal=same, greedy_ids=zero[0].tolist())


def diversity(model, tok, inputs, config):
    samples = []
    rows = []
    print("temperature | unique continuations | first tokens | starts with ' Python'")
    for temperature in [0.3, 1.0, 2.0]:
        outputs = []
        firsts = Counter()
        for seed in range(20):
            torch.manual_seed(seed)
            with torch.inference_mode():
                out = generate(model, inputs, config, temperature=temperature)
            ids = out[0, inputs.input_ids.shape[1]:].tolist()
            text = tok.decode(ids, skip_special_tokens=True)
            outputs.append(tuple(ids))
            firsts[ids[0]] += 1
            samples.append(dict(temperature=temperature, seed=seed, token_ids=ids, text=text))
        counts = [(tok.decode([token]), count) for token, count in sorted(firsts.items(), key=lambda item: (-item[1], item[0]))]
        python_starts = sum(count for token, count in firsts.items() if tok.decode([token]) == ' Python')
        print(f'{temperature} | {len(set(outputs))}/20 | {len(firsts)} | {python_starts}/20', flush=True)
        rows.append(dict(temperature=temperature, unique_sequences=len(set(outputs)),
                         unique_first_tokens=len(firsts), python_starts=python_starts,
                         first_token_counts=counts))
    return rows, samples


def main(out):
    model, tok, inputs, config = load()
    zero = zero_check(model, tok, inputs, config)
    rows, samples = diversity(model, tok, inputs, config)
    out.mkdir(parents=True, exist_ok=True)
    data = dict(model=MODEL, revision=REVISION, prompt=PROMPT,
                environment=dict(python=sys.version.split()[0], torch=torch.__version__,
                                 transformers=transformers.__version__, system=platform.system(),
                                 architecture=platform.machine(), dtype=str(model.dtype),
                                 attention='eager', cpu_threads=torch.get_num_threads()),
                generation_config=config.to_dict(), cached_defaults=model.generation_config.to_dict(),
                zero=zero, summary=rows, samples=samples)
    (out/'sampling_results.json').write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, default=Path.cwd())
    main(parser.parse_args().output_dir)
