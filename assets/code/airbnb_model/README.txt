Original conceptual diagrams for Applying Deep Learning to Airbnb Search.
The diagrams express relationships described by Haldar et al., arXiv:1810.09591.
Source checked 2026-11-16: https://arxiv.org/html/1810.09591v2
No paper figure, pixel data, measured curve, or numerical performance value is reproduced.

Requires Python 3.12 and matplotlib (tested with the version in requirements.txt).
Run from any directory:
  python render_diagrams.py --output-dir /path/to/output
Without --output-dir, images go to assets/img/ds/airbnb relative to this script.
The script checks text bounds, label containment and text overlap before saving.

Original cover (2026-11-20):
  python render_cover.py --output /path/to/cover.png
Without --output, it replaces assets/img/ds/airbnb/cover.png relative to the script.
The cover adapts the hybrid-input concept: GBDT leaves, an FM output and features
flow into a neural ranker, followed by a ranked list. It has no measured values.
All shapes and text are drawn in Matplotlib with navy, teal and lavender colors.
The script uses no external image, brand logo or paper pixels.
It checks padded renderer bounds for every label, box and arrow inside
the 1200x630 canvas's 80px horizontal / 48px vertical safe margins.
Requires Pillow (installed with Matplotlib); tested with matplotlib 3.11.2 and Pillow 13.0.0.
