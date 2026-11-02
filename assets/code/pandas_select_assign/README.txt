pandas select vs assign. Tested 2026-10-30, Python 3.12.13, pandas 3.1.0.
In an empty working folder with select_assign.py:
  python3.12 -m venv .venv
  source .venv/bin/activate
  python -m pip install pandas==3.1.0
  python select_assign.py
Optional JSON: python select_assign.py results.json
Toy order data is constructed; this is a semantics comparison, not a benchmark.
Optional cover rendering: matplotlib==3.11.0, pillow==13.0.0.
