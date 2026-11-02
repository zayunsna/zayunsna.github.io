Decoder-only batched generation: Qwen2.5-0.5B-Instruct (Apache-2.0).
Model: https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct
Revision: 7ae557604adf67be50417f59c2c2f167def9a775
Tested 2026-10-30, Python 3.12.13, torch 2.14.0, transformers 5.17.0.
macOS Apple M5; CPU float32, eager attention, one CPU thread, greedy decoding,
repetition_penalty=1.0, up to 12 new tokens, identical chat template in all runs.
  python3.12 -m venv .venv
  source .venv/bin/activate
  python -m pip install torch==2.14.0 transformers==5.17.0
  python padding.py results.json
First use downloads the public model (about 1 GB of weights); no account or
paid API is required. The verification run used already-cached model weights.
Bad configurations deliberately produce warnings. reference_output.txt omits
loading progress, not warnings or answers. Results were repeated locally.
Compares generated token IDs before the first configured generated EOS (both
IDs in the pinned model's generation_config are handled), ignoring batch
completion padding. The padded input width is used to slice new tokens.
Checks batching parity, not factual accuracy: the model's Sydney answer to the
Australia question is wrong even in the single-prompt control (capital: Canberra).
Token equality in this experiment does not guarantee equality across devices,
precisions, kernels, versions, or near-tied logits. GPU paths were not tested.
Optional cover rendering: matplotlib==3.11.0, pillow==13.0.0.
