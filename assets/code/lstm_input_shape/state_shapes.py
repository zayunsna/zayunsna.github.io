"""Inspect batched LSTM states and forward/reverse endpoints on CPU."""
import torch
from torch import nn

torch.manual_seed(7)
x = torch.randn(3, 5, 2)  # batch, timesteps, features
for batch_first in [False, True]:
    model = nn.LSTM(2, 4, num_layers=2, batch_first=batch_first)
    model.eval()
    inputs = x if batch_first else x.transpose(0, 1)
    with torch.no_grad():
        output, (h_n, c_n) = model(inputs)
    print(f"batch_first={batch_first}: output={tuple(output.shape)}, "
          f"h_n={tuple(h_n.shape)}, c_n={tuple(c_n.shape)}")

torch.manual_seed(7)
x = torch.randn(3, 5, 2)
for bidirectional in [False, True]:
    model = nn.LSTM(2, 4, num_layers=2, batch_first=True,
                    bidirectional=bidirectional)
    model.eval()
    with torch.no_grad():
        output, (h_n, c_n) = model(x)
    print(f"bidirectional={bidirectional}: output={tuple(output.shape)}, "
          f"h_n={tuple(h_n.shape)}, c_n={tuple(c_n.shape)}")
    if not bidirectional:
        print("last output equals final hidden:", torch.equal(output[:, -1], h_n[-1]))
    else:
        final_pair = torch.cat((h_n[-2], h_n[-1]), dim=-1)
        print("last output equals final pair:", torch.equal(output[:, -1], final_pair))
        print("last forward equals final forward:", torch.equal(output[:, -1, :4], h_n[-2]))
        print("first reverse equals final reverse:", torch.equal(output[:, 0, 4:], h_n[-1]))
        print("last reverse equals final reverse:", torch.equal(output[:, -1, 4:], h_n[-1]))
