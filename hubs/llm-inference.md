---
layout: page
title: LLM Inference
short: tokens, decoding, KV cache, precision and padding, run on a small local model
description: >
  How an LLM turns a prompt into tokens and back: tokenization, decoding, KV cache, quantization and batching, tested locally.
permalink: /llm-inference/
hub: llm-inference
hide_description: true
hub_order: 4
---

These posts run small open models, mostly Qwen2.5-0.5B, on a laptop CPU or Apple GPU and print the tensors involved:
token IDs, logits, cache sizes, memory per precision. No API key is needed for any of them.

[Tokens](/tips/2026-09-29-what_is_a_token/) and [the context window](/tips/2026-10-07-context_window/) come first; decoding, KV cache, quantization and batching build on them.

{% include hub.html %}
