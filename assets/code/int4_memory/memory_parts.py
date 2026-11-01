"""Measure separate int4 packing and real fp16 Qwen KV tensor payloads.

This is not a complete quantized LLM or an out-of-memory stress test.
"""
import argparse
import gc
import json
import platform
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import AutoModel

MODEL = 'Qwen/Qwen2.5-0.5B-Instruct'
REVISION = '7ae557604adf67be50417f59c2c2f167def9a775'


def pack_int4(weight, group_size=64):
    groups = weight.reshape(-1, group_size)
    scale = np.maximum(np.abs(groups).max(axis=1, keepdims=True) / 7,
                       np.finfo(np.float32).tiny)
    codes = (np.rint(groups / scale).clip(-7, 7).astype(np.int8) + 8).astype(np.uint8)
    flat = codes.reshape(-1)
    packed = flat[::2] | (flat[1::2] << 4)
    return packed, scale


def unpack_int4(packed, scale, shape, group_size=64):
    codes = np.empty(packed.size * 2, dtype=np.uint8)
    codes[::2], codes[1::2] = packed & 15, packed >> 4
    groups = (codes.astype(np.float32) - 8).reshape(-1, group_size)
    return (groups * scale).reshape(shape)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--local-files-only', action='store_true')
    args = parser.parse_args()
    rng = np.random.default_rng(0)
    weight = rng.standard_normal((1024, 1024)).astype('float32')
    packed, scale = pack_int4(weight)
    restored = unpack_int4(packed, scale, weight.shape)
    assert packed.dtype == np.uint8 and scale.dtype == np.float32
    assert np.all(np.abs(restored - weight).reshape(-1, 64) <= scale / 2 + 1e-6)
    x = np.ones((1, 1024), dtype=np.float32)
    assert (x @ restored.T).shape == (1, 1024)
    report = dict(python=platform.python_version(), torch=torch.__version__,
        transformers=transformers.__version__, numpy=np.__version__, model=MODEL, revision=REVISION,
        packing=dict(original_bytes=weight.nbytes, packed_bytes=packed.nbytes,
                     scale_bytes=scale.nbytes, persistent_bytes=packed.nbytes + scale.nbytes,
                     dequantized_bytes=restored.nbytes), cache=[])
    del weight, restored, x
    gc.collect()
    torch.set_num_threads(2)
    model = AutoModel.from_pretrained(MODEL, revision=REVISION, dtype=torch.float16,
        attn_implementation='eager', local_files_only=args.local_files_only).eval()
    config = model.config
    head_dim = config.hidden_size // config.num_attention_heads
    per_token = 2 * config.num_hidden_layers * config.num_key_value_heads * head_dim * 2
    report['config'] = dict(layers=config.num_hidden_layers, kv_heads=config.num_key_value_heads,
                           head_dim=head_dim, fp16_bytes_per_token=per_token)
    with torch.inference_mode():
        for batch, tokens in ((1, 128), (1, 512), (2, 512)):
            ids = torch.full((batch, tokens), 42, dtype=torch.long)
            output = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=True)
            cache = output.past_key_values
            tensors = [tensor for layer in cache.layers for tensor in (layer.keys, layer.values)]
            payload = sum(t.numel() * t.element_size() for t in tensors)
            assert all(t.dtype == torch.float16 for t in tensors)
            assert payload == batch * tokens * per_token
            report['cache'].append(dict(batch=batch, tokens=tokens, tensor_bytes=payload,
                first_key_shape=list(tensors[0].shape), dtype=str(tensors[0].dtype)))
            del output, cache, tensors, ids
    report['projection_only'] = dict(batch=8, tokens=32768,
                                    fp16_cache_bytes=8 * 32768 * per_token)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
