---
layout: post
title: "Use left padding and an attention mask for LLM batches"
description: >
  The same short prompt produced different tokens with right padding or a missing attention mask. Reproduce
  the decoder-only generation pitfall locally, then compare correct left-padded batches with single prompts.
image: /assets/img/tips/llm_padding/cover.png
sitemap:
  changefreq: daily
  priority: 1.0
---

# Use left padding and an attention mask for LLM batches

If a decoder-only Hugging Face model answers differently when you put a short prompt beside a longer one,
check **padding side and the attention mask** before changing the prompt or model.

With Qwen2.5-0.5B-Instruct on CPU, the short sequence prompt below matched its single-prompt output when
left-padded with a mask. Right-padding it changed the output even though the mask was present. Removing the
mask from the left-padded batch changed it again. Sampling was off in all runs.

| Short prompt: `Continue the sequence: 1, 2, 3,` | Continuation, up to 12 new tokens | Matches single-prompt token IDs |
|---|---|---|
| Alone | `1, 2, 3, 4, ` | Control |
| Left padding + mask | `1, 2, 3, 4, ` | Yes |
| Right padding + mask | `\nSure, here to continue the sequence: 12` | No |
| Left padding, mask omitted | `4, 5, 6, 7, ` | No |

The control repeats the start of the sequence. The left-padded batch with a mask reproduces that continuation
exactly; the other configurations change it.

Tested on 2026-10-30: Python 3.12.13, torch 2.14.0, transformers 5.17.0, macOS Apple M5, one CPU thread,
float32, eager attention. Model [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct),
Apache-2.0, revision `7ae557604adf67be50417f59c2c2f167def9a775`. Greedy decoding, `repetition_penalty=1.0`.
Two local runs produced the same compared tokens. Full [script](/assets/code/llm_padding/padding.py),
[output including warnings](/assets/code/llm_padding/reference_output.txt),
[result JSON](/assets/code/llm_padding/results.json) and [run instructions](/assets/code/llm_padding/README.txt).

## Final-position logits in a padded batch

The two templated prompts had 42 and 43 tokens. A tensor batch needs the same width for both rows, so
the shorter row receives one padding token.

With left padding, its last tensor position is still the last real prompt token. With right padding, that
position contains the pad token. In this test the final IDs were:

```text
left+mask:  [198, 198]
right+mask: [151645, 198]
```

The left values are real final prompt tokens. `151645` is the EOS token reused for padding in this experiment.
For the ordinary greedy generation path in the tested version, the next-token logits come from the final tensor
position, `outputs.logits[:, -1]`. That final position supplies the next-token scores in both padded batches.
The [versioned generation source](https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/generation/utils.py)
and the installed source were checked for that behavior.

The attention mask excludes padding positions as keys. The generation loop still selects the final query
position's logits, which explains why the right-padded shorter row changed output even with the mask. The longer row had no padding, and matched its single-prompt output in all three configurations.

Hugging Face's [generation guide](https://huggingface.co/docs/transformers/llm_tutorial#padding-side) recommends
left padding for this decoder-only generation workflow.

## Pass the tokenizer attention mask

The test deliberately sets `pad_token` to `eos_token`. This is a common way to batch a model without a separate
padding token. An EOS token ID alone then leaves its role ambiguous, since it can mark either EOS or padding.

The tokenizer's `attention_mask` records that distinction. In the standard 2D mask used here, **1 means a real
input token and 0 means padding**. When that mask was omitted, transformers warned that it could not infer it
reliably because the two special token IDs were equal. The omitted-mask run changed the shorter row's output.
In this version, generation also derives position IDs from the mask when preparing inputs, so omitting it
can change position handling as well as attention.

The normal configuration is small. Install `torch==2.14.0` and `transformers==5.17.0` in a Python 3.12 virtual
environment, then run this complete example. The first run downloads the public model; the test above used
cached weights. There is no paid inference API or account requirement.

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

name = 'Qwen/Qwen2.5-0.5B-Instruct'
revision = '7ae557604adf67be50417f59c2c2f167def9a775'
torch.set_num_threads(1)
tok = AutoTokenizer.from_pretrained(name, revision=revision, padding_side='left')
tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(
    name, revision=revision, dtype=torch.float32,
    attn_implementation='eager',
).eval()
questions = [
    'Continue the sequence: 1, 2, 3,',
    'What is the capital of Australia? Answer with only the city name.',
]
prompts = [tok.apply_chat_template(
    [{'role': 'user', 'content': q}], tokenize=False, add_generation_prompt=True,
) for q in questions]
batch = tok(prompts, padding=True, return_tensors='pt', add_special_tokens=False)
with torch.no_grad():
    generated = model.generate(
        **batch, max_new_tokens=12, do_sample=False,
        repetition_penalty=1.0, pad_token_id=tok.pad_token_id,
    )
new_tokens = generated[:, batch['input_ids'].shape[1]:]
print(tok.batch_decode(new_tokens, skip_special_tokens=True))
```

```text
['1, 2, 3, 4, ', 'Sydney']
```

The capital-question answer is wrong even in the single-prompt control: Australia's capital is Canberra.
The control and all three batched configurations returned the same wrong city. Token parity and factual
accuracy therefore need separate checks.

Slice continuations at the **padded tensor width**, as above. Slicing at each unpadded prompt length
can accidentally include some input tokens in what you report as an answer. For exact comparisons, the full
script compares token IDs through the first generated EOS and ignores padding added while other batch rows
finish. Decoded text alone can hide special-token differences.

## Compare token IDs with fixed generation settings

Keep the same [chat template](/tips/2026-10-27-chat_template/), generation settings and model revision while
debugging. Sampling adds another source of variation. Even with greedy decoding, changes in batch shape,
precision or kernels can perturb floating-point calculations and flip nearly tied next-token scores; see
PyTorch's [numerical accuracy notes](https://docs.pytorch.org/docs/2.14/notes/numerical_accuracy.html).
The measured parity comes from one Qwen model and CPU configuration; GPU and serving backends remain
untested. The omitted-mask experiment also leaves the contributions of attention and position handling
unseparated, and an omitted mask may preserve outputs on other inputs. Encoder-decoder models, training
tasks and specialized serving engines require their own padding rules.

Start with one prompt, add a much longer neighbor, and compare token IDs. If the short row diverges, inspect
its mask and final input position first. Use `model.eval()` for inference behavior; the
[eval vs no_grad comparison](/tips/2026-11-02-eval_no_grad/) explains that setting separately.
Padding and mask construction still need explicit checks.
Documentation and source checks above were made on 2026-10-30.

The cover shows the shorter prompt's actual final ten mask positions beside the token ID used at the end;
its [renderer](/assets/code/tue_wed_figures/render_figures.py) reads the experiment JSON.
