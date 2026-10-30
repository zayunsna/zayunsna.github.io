"""Separate module mode from gradient recording on a small CPU model."""
import contextlib
import json
from pathlib import Path
import sys

import torch
from torch import nn


def mode_matrix():
    model = nn.Sequential(nn.Linear(4, 4, bias=False), nn.Dropout(p=0.5))
    with torch.no_grad():
        model[0].weight.copy_(torch.eye(4))
    x = torch.arange(1, 9, dtype=torch.float32).reshape(2, 4)
    rows = []
    print("mode   no_grad  same_output  output_requires_grad  weight_requires_grad")
    for evaluation in (False, True):
        for no_grad in (False, True):
            model.train(not evaluation)
            torch.manual_seed(0)
            context = torch.no_grad() if no_grad else contextlib.nullcontext()
            with context:
                first, second = model(x), model(x)
            row = dict(mode="eval" if evaluation else "train", no_grad=no_grad,
                       same_output=torch.equal(first, second),
                       output_requires_grad=first.requires_grad,
                       weight_requires_grad=model[0].weight.requires_grad,
                       first_output=first.detach().tolist(), second_output=second.detach().tolist())
            rows.append(row)
            print(f"{row['mode']:<6} {str(no_grad):<8} {str(row['same_output']):<12} "
                  f"{str(row['output_requires_grad']):<21} {row['weight_requires_grad']}")
    assert [r["same_output"] for r in rows] == [False, False, True, True]
    assert [r["output_requires_grad"] for r in rows] == [True, False, True, False]
    assert all(r["weight_requires_grad"] for r in rows)

    model.eval()
    model.zero_grad(set_to_none=True)
    model(x).sum().backward()
    assert torch.equal(model[0].weight.grad, torch.tensor([[6., 8., 10., 12.]]).repeat(4, 1))
    print("eval backward weight gradient row:", model[0].weight.grad[0].tolist())
    return rows


def batchnorm_probe():
    x = torch.tensor([[10., 20.], [14., 24.]])
    rows = []
    print("BatchNorm running_mean after one no_grad forward:")
    for evaluation in (False, True):
        bn = nn.BatchNorm1d(2, momentum=0.1)
        bn.train(not evaluation)
        with torch.no_grad():
            bn(x)
        means = [round(v, 4) for v in bn.running_mean.tolist()]
        rows.append(dict(mode="eval" if evaluation else "train", running_mean=means))
        print(f"{rows[-1]['mode']}: {means}")
    assert rows[0]["running_mean"] == [1.2, 2.2]
    assert rows[1]["running_mean"] == [0., 0.]
    return rows


if __name__ == "__main__":
    torch.set_num_threads(1)
    print("torch", torch.__version__, "| Python", sys.version.split()[0], "| CPU")
    result = dict(torch=torch.__version__, python=sys.version.split()[0],
                  matrix=mode_matrix(), batchnorm=batchnorm_probe())
    if len(sys.argv) > 1:
        Path(sys.argv[1]).write_text(json.dumps(result, indent=2) + "\n")
