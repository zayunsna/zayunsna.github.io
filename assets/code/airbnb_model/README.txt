Original conceptual diagrams for Applying Deep Learning to Airbnb Search.
The diagrams express relationships described by Haldar et al., arXiv:1810.09591.
Source checked 2026-11-16: https://arxiv.org/html/1810.09591v2
No paper figure, pixel data, measured curve, or numerical performance value is reproduced.

Requires Python 3.12 and matplotlib (tested with the version in requirements.txt).
Run from any directory:
  python render_diagrams.py --output-dir /path/to/output
Without --output-dir, images go to assets/img/ds/airbnb relative to this script.
The script checks text bounds, label containment and text overlap before saving.
