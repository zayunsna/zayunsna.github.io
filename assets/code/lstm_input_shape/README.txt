Hidden/cell state shapes and bidirectional endpoint checks for the LSTM input-shape article.

Environment: Python 3.12.13, CPU float32, macOS arm64, tested 2026-11-17.
Install in a fresh environment:
  python -m pip install -r requirements.txt
Run:
  python state_shapes.py

The output should match reference_output.txt.
The two loops correspond to the two added Python blocks in the article.
The script uses fixed seeds, equal-length batched inputs, zero default initial states,
default dropout=0, and no projection. All computations run on CPU.
Keras and NumPy are pinned for the original article's examples; this added script uses PyTorch only.
GPU, packed variable-length sequences and LSTM projections were not tested.
This script inspects tensor shapes and state endpoints.
