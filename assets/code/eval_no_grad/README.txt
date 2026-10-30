model.eval() vs torch.no_grad()

With Python 3.12, download eval_no_grad.py into an empty folder:
  python3.12 -m venv .venv
  source .venv/bin/activate
  python -m pip install torch==2.14.0
  python eval_no_grad.py

CPU only; no data download, training, GPU, account or API key.
Optional JSON output for the image renderer:
  python eval_no_grad.py results.json
  python -m pip install matplotlib==3.11.0 pillow==13.0.0
  python render_cover.py results.json cover.png

Tested on 2026-10-30: Python 3.12.13, torch 2.14.0, macOS, Apple M5.
reference_output.txt records stdout. Each matrix row resets the RNG once,
then runs two forwards without resetting between them. Different Dropout
outputs are observations for this seed, not a guarantee for all samples.
BatchNorm is fresh for each probe, with track_running_stats=True (default).
