"""Robustness: repeat part 1 and the ~1.5 rows-per-ID HGB case of part 2 with other seeds."""
import numpy as np, pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import KFold
import target_encoding as t

df = fetch_openml(data_id=42125, as_frame=True).frame
y = df["current_annual_salary"].astype(float)
X = df[t.CAT + t.NUM].copy(); X[t.CAT] = X[t.CAT].astype(str)
for seed in [1, 2, 3]:
    t.SEED = seed; t.CV = KFold(5, shuffle=True, random_state=seed)
    print("seed", seed)
    t.part1_leak(X, y)
    Xr = t.add_random_id(X, 1.5); cats = t.CAT + ["random_id"]
    p = t.cv_scores(lambda: t.pipeline(t.MeanEncoder(), cats), Xr, y)
    q = t.cv_scores(lambda: t.pipeline(t.target_encoder(), cats), Xr, y)
    print(f"  ~1.5 rows/ID HGB: mean {p[1]:,.0f}  TE {q[1]:,.0f}")
