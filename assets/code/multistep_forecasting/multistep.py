"""Forecasting 24 hours ahead: recursive vs direct vs a model without short lags.

Same data and model as the KFold vs TimeSeriesSplit post (OpenML "Bike_Sharing_Demand" v2, hourly).
Every row of the 8-week holdout is a forecast origin; each strategy predicts the next 1..24 rows.
Run: python multistep.py
"""
import sys
import time

import numpy as np
import pandas as pd
import sklearn
from sklearn.datasets import fetch_openml
from sklearn.ensemble import HistGradientBoostingRegressor

HOLDOUT = 24 * 7 * 8  # last 8 weeks of rows, as in the previous post
H = 24                # forecast horizon in rows (about one day)


def load():
    df = fetch_openml("Bike_Sharing_Demand", version=2, as_frame=True).frame
    for col in ["season", "weather"]:
        df[col] = df[col].cat.codes
    for col in ["holiday", "workingday"]:
        df[col] = (df[col].astype(str) == "True").astype(int)
    y = df.pop("count").astype(float).to_numpy()
    df["t"] = np.arange(len(df))
    return df, y  # calendar + weather + t; lags are added per strategy


def model(**kw):
    return HistGradientBoostingRegressor(random_state=0, **kw)


def with_lags(X, y, lags):
    """Add y shifted by each lag. Row i gets y[i - lag]."""
    out = X.copy()
    for lag in lags:
        out[f"lag_{lag}"] = np.concatenate([np.full(lag, np.nan), y[:-lag]])
    return out


def main():
    print(f"python {sys.version.split()[0]}, scikit-learn {sklearn.__version__}, pandas {pd.__version__}")
    X, y = load()
    n = len(y)
    start = n - HOLDOUT                       # first holdout row
    dev = slice(168, start)                   # skip rows without a one-week lag, as before
    origins = np.arange(start - 1, n - H)     # last observed row; forecasts cover origin+1 .. origin+H
    print(f"rows: dev {start - 168}, holdout {HOLDOUT}; forecast origins {len(origins)}, horizon 1..{H}")

    def mae_by_h(pred):                       # pred[k, h-1] = forecast of y[origins[k] + h]
        truth = y[origins[:, None] + np.arange(1, H + 1)]
        return np.abs(pred - truth).mean(axis=0)

    def run(kw):
        results, cost = {}, {}

        # 1. One-step model with lag_1, lag_24, lag_168, scored the "easy" way: every lag_1 is the real value.
        X1 = with_lags(X, y, [1, 24, 168])
        t0 = time.perf_counter()
        one_step = model(**kw).fit(X1.iloc[dev], y[dev])
        cost["one-step model"] = (1, time.perf_counter() - t0)
        oracle = np.column_stack([one_step.predict(X1.iloc[origins + h]) for h in range(1, H + 1)])
        results["one-step, real lag_1 (not a forecast)"] = mae_by_h(oracle)

        # 2. Recursive: the same model, but lag_1 for step h > 1 is its own forecast for step h - 1.
        #    lag_24 and lag_168 of every target row are still observed values when h <= 24.
        rec = np.empty((len(origins), H))
        for h in range(1, H + 1):
            feats = X1.iloc[origins + h].copy()
            if h > 1:
                feats["lag_1"] = rec[:, h - 2]
            rec[:, h - 1] = one_step.predict(feats)
        results["recursive"] = mae_by_h(rec)

        # 3. Direct: one model per horizon h, trained with lag_h (the last value known at the origin)
        #    in place of lag_1. h = 1 is the one-step model again.
        direct = np.empty((len(origins), H))
        t0 = time.perf_counter()
        for h in range(1, H + 1):
            Xh = with_lags(X, y, sorted({h, 24, 168}))
            m = model(**kw).fit(Xh.iloc[dev], y[dev])
            direct[:, h - 1] = m.predict(Xh.iloc[origins + h])
        cost["direct (24 models)"] = (H, time.perf_counter() - t0)
        results["direct"] = mae_by_h(direct)

        # 4. No short lag at all: one model with lag_24 and lag_168 only, valid for every h <= 24.
        X24 = with_lags(X, y, [24, 168])
        t0 = time.perf_counter()
        no_short = model(**kw).fit(X24.iloc[dev], y[dev])
        cost["lag_24 + lag_168 model"] = (1, time.perf_counter() - t0)
        flat = np.column_stack([no_short.predict(X24.iloc[origins + h]) for h in range(1, H + 1)])
        results["lag_24 + lag_168 only"] = mae_by_h(flat)

        # 5. No lags: calendar + weather + t only (the first model of the previous post).
        no_lag = model(**kw).fit(X.iloc[dev], y[dev])
        base = np.column_stack([no_lag.predict(X.iloc[origins + h]) for h in range(1, H + 1)])
        results["no lags"] = mae_by_h(base)
        return results, cost

    def report(title, results):
        print(f"\n{title}")
        print(f"  {'strategy':40s} {'h=1':>6} {'h=2':>6} {'h=3':>6} {'h=6':>6} {'h=12':>6} {'h=24':>6}  mean 1-24")
        for name, m in results.items():
            cells = " ".join(f"{m[h - 1]:6.1f}" for h in [1, 2, 3, 6, 12, 24])
            print(f"  {name:40s} {cells}  {m.mean():9.1f}")

    results, cost = run({})
    report("MAE by horizon (bikes per hour, lower is better), default HistGradientBoostingRegressor", results)
    print("\n  direct, MAE at every horizon 1..24:", " ".join(f"{v:.1f}" for v in results["direct"]))

    # Default settings hold out a random 10% of training rows for early stopping (n > 10,000).
    # Check that the ranking does not depend on that: train every model on all dev rows.
    full, _ = run({"early_stopping": False})
    report("Same, early_stopping=False (every model trained on all dev rows)", full)
    print("\n  direct, MAE at every horizon 1..24:", " ".join(f"{v:.1f}" for v in full["direct"]))
    print("\nTraining cost on the dev set (default settings)")
    for name, (k, s) in cost.items():
        print(f"  {name:24s} {k:2d} fit(s), {s:5.1f} s")

    np.save("mae_by_h.npy", np.array([results[k] for k in results]))
    with open("mae_by_h_names.txt", "w") as f:
        f.write("\n".join(results))


if __name__ == "__main__":
    main()
