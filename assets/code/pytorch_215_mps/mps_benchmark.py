"""Bounded, synchronized MPS inference microbenchmark; no model download."""
import argparse
import hashlib
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn


def make_fixture():
    rng = np.random.default_rng(20261101)
    layers = []
    digest = hashlib.sha256()
    for inputs, outputs in ((1024, 2048), (2048, 1024), (1024, 512)):
        layer = nn.Linear(inputs, outputs)
        weight = (rng.standard_normal((outputs, inputs)) / inputs**0.5).astype('float32')
        bias = np.zeros(outputs, dtype='float32')
        digest.update(weight.tobytes())
        digest.update(bias.tobytes())
        with torch.no_grad():
            layer.weight.copy_(torch.from_numpy(weight))
            layer.bias.copy_(torch.from_numpy(bias))
        layers.append(layer)
        if outputs != 512:
            layers.append(nn.GELU())
    x = rng.standard_normal((64, 1024)).astype('float32')
    digest.update(x.tobytes())
    return nn.Sequential(*layers).eval(), torch.from_numpy(x), digest.hexdigest()


def timed_forward(model, x, repeats):
    torch.mps.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        result = model(x)
        torch.mps.synchronize()
    return result, 1000 * (time.perf_counter() - start) / repeats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not torch.backends.mps.is_available():
        raise RuntimeError('This benchmark requires an available MPS device; no CPU fallback.')
    torch.set_num_threads(1)
    setup_start = time.perf_counter()
    model, cpu_input, sha = make_fixture()
    cpu_reference = {}
    with torch.inference_mode():
        for batch in (1, 64):
            cpu_reference[batch] = model(cpu_input[:batch]).clone()
    model = model.to('mps')
    gpu_input = cpu_input.to('mps')
    torch.mps.synchronize()
    setup_ms = (time.perf_counter() - setup_start) * 1000
    report = dict(torch=torch.__version__, numpy=np.__version__, python=platform.python_version(),
                  os=platform.platform(), fixture_sha256=sha, setup_ms=setup_ms,
                  parameters=sum(p.numel() for p in model.parameters()), cases=[])
    with torch.inference_mode():
        for batch in (1, 64):
            x = gpu_input[:batch]
            _, first_ms = timed_forward(model, x, 1)
            timed_forward(model, x, 20)
            samples = [timed_forward(model, x, 20)[1] for _ in range(21)]
            output = model(x).cpu()
            error = (output - cpu_reference[batch]).abs().max().item()
            torch.testing.assert_close(output, cpu_reference[batch], atol=2e-5, rtol=2e-4)
            report['cases'].append(dict(batch=batch, first_forward_ms=first_ms,
                warm_median_ms=statistics.median(samples), warm_samples_ms=samples,
                max_abs_cpu_error=error, output_first_8=output[0, :8].tolist()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
