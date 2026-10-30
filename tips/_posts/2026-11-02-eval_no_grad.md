---
layout: post
title: "model.eval() vs no_grad(): what changes during inference?"
description: >
  model.eval() changes Dropout and BatchNorm behavior; torch.no_grad() stops reverse-mode gradient recording.
  Run all four combinations and see why inference usually needs both.
image: /assets/img/tips/eval_no_grad/cover.png
sitemap:
  changefreq: daily
  priority: 1.0
---

# model.eval() vs no_grad(): what changes during inference?

For ordinary prediction, use both:

```python
model.eval()
with torch.no_grad():
    predictions = model(inputs)
```

`eval()` selects evaluation behavior in modules such as Dropout and BatchNorm. `no_grad()` stops recording
operations for reverse-mode automatic differentiation inside its context. One does not imply the other.

There are two separate questions: **how should this layer behave?** and **will this computation need a backward
pass?** PyTorch's [autograd notes](https://docs.pytorch.org/docs/2.14/notes/autograd.html#evaluation-mode-nn-module-eval)
explicitly separate the two (checked 2026-10-30).

## Run the four combinations

This tiny model has a linear layer with identity weights followed by Dropout. Its parameters require gradients,
so normal forward operations can build a backward graph even though the input tensor does not require gradients.

_Tested on 2026-10-30 with Python 3.12.13 and PyTorch 2.14.0 on macOS, Apple M5, CPU. No training, GPU or
dataset download. Install `torch==2.14.0` in a Python 3.12 virtual environment. The complete
[script](/assets/code/eval_no_grad/eval_no_grad.py),
[run instructions](/assets/code/eval_no_grad/README.txt) and
[recorded output](/assets/code/eval_no_grad/reference_output.txt) are available._

Run the following as one script. The next Python blocks reuse its imports and variables.

```python
import contextlib
import torch
from torch import nn

model = nn.Sequential(nn.Linear(4, 4, bias=False), nn.Dropout(p=0.5))
with torch.no_grad():
    model[0].weight.copy_(torch.eye(4))
x = torch.arange(1, 9, dtype=torch.float32).reshape(2, 4)

print("mode   no_grad  same_output  output_requires_grad  weight_requires_grad")
for evaluation in (False, True):
    for no_grad in (False, True):
        model.train(not evaluation)  # train(False) is eval()
        torch.manual_seed(0)
        context = torch.no_grad() if no_grad else contextlib.nullcontext()
        with context:
            first, second = model(x), model(x)
        mode = "eval" if evaluation else "train"
        print(f"{mode:<6} {str(no_grad):<8} {str(torch.equal(first, second)):<12} "
              f"{str(first.requires_grad):<21} {model[0].weight.requires_grad}")
```

Output:

```text
mode   no_grad  same_output  output_requires_grad  weight_requires_grad
train  False    False        True                  True
train  True     False        False                 True
eval   False    True         True                  True
eval   True     True         False                 True
```

The seed resets once per row, not between the two forward passes. In the two training rows, Dropout samples
different masks in this run. `no_grad()` does not stop that sampling. Two random masks can coincide, especially
with tiny inputs, so unequal outputs are an observation here, not a universal test for training mode.

In the evaluation rows, standard `nn.Dropout` is the identity: it does not drop values. But the third row
still has `output_requires_grad=True`. `eval()` did not disable the backward graph through the linear layer.
The parameters' `requires_grad` flag stays `True` in all four rows.

The [Dropout documentation](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Dropout.html) describes its
training-time random masking and scaling. Evaluation is not just “use the same random mask every time.”
Also, `eval()` is not a general determinism switch: custom random operations or nondeterministic kernels can
still produce different outputs.

## no_grad() does not freeze BatchNorm's state

With default `track_running_stats=True`, BatchNorm updates running statistics in training mode. That update
is a forward-pass state change, not a gradient or optimizer step.

```python
batch = torch.tensor([[10., 20.], [14., 24.]])
for evaluation in (False, True):
    bn = nn.BatchNorm1d(2, momentum=0.1)
    bn.train(not evaluation)
    with torch.no_grad():
        bn(batch)
    means = [round(v, 4) for v in bn.running_mean.tolist()]
    print(f"{'eval' if evaluation else 'train'}: {means}")
```

Output:

```text
train: [1.2, 2.2]
eval: [0.0, 0.0]
```

Each row starts with a fresh layer whose running mean is zero. The batch mean is `[12, 22]`, so a training
forward with momentum 0.1 updates the buffer to `[1.2, 2.2]`, even inside `no_grad()`. The evaluation forward
leaves it unchanged. These numbers demonstrate the update rule; this fresh, untrained layer is not an example
of useful prediction quality.

During evaluation, BatchNorm normally uses the stored statistics rather than the current batch's. There is an
important exception: [`track_running_stats=False`](https://docs.pytorch.org/docs/2.14/generated/torch.nn.BatchNorm1d.html)
keeps no running estimates and uses batch statistics in evaluation too. `eval()` cannot turn those nonexistent
buffers into learned statistics.

## Evaluation can still need gradients

For saliency or sensitivity analysis, you may want evaluation behavior and a backward pass. Continuing with
the small linear-plus-Dropout model above:

```python
model.eval()
model.zero_grad(set_to_none=True)
model(x).sum().backward()
print(model[0].weight.grad[0].tolist())
```

Output:

```text
[6.0, 8.0, 10.0, 12.0]
```

That is a real gradient through a model in evaluation mode. Wrapping this forward in `no_grad()` would remove
the graph needed by this backward call. Conversely, neither `eval()` nor `no_grad()` clears old `.grad`
values or permanently freezes parameters.

For plain validation or prediction, select evaluation mode and use `no_grad()`; switch back with `model.train()`
before resuming training. Avoid treating “gradients off” as “all model state frozen.” The
[no_grad reference](https://docs.pytorch.org/docs/2.14/generated/torch.no_grad.html) also documents its scope:
it is thread-local, has an exception for tensor factory functions with a `requires_grad` argument, and does
not disable forward-mode automatic differentiation.

The [LSTM forecasting tutorial](/blog/2026-10-28-lstm_forecasting_tutorial/) uses evaluation mode plus
`no_grad()` during validation and prediction. Its one-layer LSTM has no inter-layer dropout, so the mode
switch alone need not change its output. The distinction matters when you add modules whose training and
evaluation behavior differ.
