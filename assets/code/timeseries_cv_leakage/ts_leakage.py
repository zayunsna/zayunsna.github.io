"""Random KFold vs TimeSeriesSplit on hourly bike demand, plus a rolling-feature leak.

Data: OpenML "Bike_Sharing_Demand" v2 (UCI Bike Sharing, hourly, 2011-2012).
Run: python ts_leakage.py
"""
import sys

import numpy as np
import pandas as pd
import sklearn
from sklearn.datasets import fetch_openml
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, TimeSeriesSplit, cross_validate

HOLDOUT = 24 * 7 * 8  # last 8 weeks of rows = "the future" we pretend to deploy into
SCHEMES = {
    "KFold(5, shuffle=True)": KFold(5, shuffle=True, random_state=0),
    "KFold(5)": KFold(5),
    "TimeSeriesSplit(5)": TimeSeriesSplit(5),
    "TimeSeriesSplit(5, gap=168)": TimeSeriesSplit(5, gap=168),
}


def load():
    df = fetch_openml("Bike_Sharing_Demand", version=2, as_frame=True).frame
    for col in ["season", "weather"]:
        df[col] = df[col].cat.codes
    for col in ["holiday", "workingday"]:
        df[col] = (df[col].astype(str) == "True").astype(int)  # stored as "True"/"False" categories
    y = df["count"].astype(float)
    df["t"] = np.arange(len(df))  # row index; trees cannot extrapolate it past the training range
    for lag in [1, 24, 168]:      # demand 1, 24 and 168 rows (about 1 h, 1 day, 1 week) earlier
        df[f"lag_{lag}"] = y.shift(lag)
    return df.drop(columns="count").iloc[168:], y.iloc[168:]


def model():
    return HistGradientBoostingRegressor(random_state=0)


def cv_folds(X, y, cv):
    r = cross_validate(model(), X, y, cv=cv, scoring="neg_mean_absolute_error")
    return -r["test_score"]


def holdout_mae(X_dev, y_dev, X_test, y_test):
    return mean_absolute_error(y_test, model().fit(X_dev, y_dev).predict(X_test))


def compare(title, X_dev, y_dev, X_test, y_test):
    print(f"\n{title}")
    for name, cv in SCHEMES.items():
        f = cv_folds(X_dev, y_dev, cv)
        print(f"  {name:28s} mean {f.mean():5.1f}   folds {' '.join(f'{v:5.1f}' for v in f)}")
    print(f"  {'holdout: last 8 weeks':28s} {holdout_mae(X_dev, y_dev, X_test, y_test):10.1f}")


def main():
    print(f"python {sys.version.split()[0]}, scikit-learn {sklearn.__version__}, pandas {pd.__version__}")
    X, y = load()
    X_dev, y_dev, X_test, y_test = X.iloc[:-HOLDOUT], y.iloc[:-HOLDOUT], X.iloc[-HOLDOUT:], y.iloc[-HOLDOUT:]
    print(f"rows: dev {len(X_dev)}, holdout {len(X_test)}; mean hourly count: dev {y_dev.mean():.1f}, holdout {y_test.mean():.1f}")

    lags = [c for c in X if c.startswith("lag_")]
    compare("[1] Calendar + weather + row index t, no lags (MAE, lower is better)",
            X_dev.drop(columns=lags), y_dev, X_test.drop(columns=lags), y_test)
    compare("[2] Same plus lag_1, lag_24, lag_168",
            X_dev, y_dev, X_test, y_test)

    print("\n[3] A rolling feature that includes the current hour (lag model, TimeSeriesSplit(5))")
    yfull = pd.concat([y_dev, y_test])
    leaky = yfull.rolling(3).mean()            # mean of hours t-2, t-1, t  -> contains the target
    honest = yfull.shift(1).rolling(3).mean()  # mean of hours t-3, t-2, t-1 -> known at prediction time
    Xl_dev, Xl_test = X_dev.assign(roll_3=leaky.iloc[:-HOLDOUT]), X_test.assign(roll_3=leaky.iloc[-HOLDOUT:])
    Xh_dev, Xh_test = X_dev.assign(roll_3=honest.iloc[:-HOLDOUT]), X_test.assign(roll_3=honest.iloc[-HOLDOUT:])
    leaky_model = model().fit(Xl_dev, y_dev)
    print(f"  leaky feature: CV MAE                        {cv_folds(Xl_dev, y_dev, TimeSeriesSplit(5)).mean():5.1f}")
    print(f"  leaky feature: holdout, leaky values         {mean_absolute_error(y_test, leaky_model.predict(Xl_test)):5.1f}")
    print(f"  leaky model, honest values at serving time   {mean_absolute_error(y_test, leaky_model.predict(Xh_test)):5.1f}")
    print(f"  honest feature: CV MAE                       {cv_folds(Xh_dev, y_dev, TimeSeriesSplit(5)).mean():5.1f}")
    print(f"  honest feature: holdout                      {holdout_mae(Xh_dev, y_dev, Xh_test, y_test):5.1f}")

    print("\n[4] Rows are not always hours: reconstruct elapsed hours from weekday + hour")
    raw = fetch_openml("Bike_Sharing_Demand", version=2, as_frame=True).frame
    hour, wday = raw["hour"].astype(int).to_numpy(), raw["weekday"].astype(int).to_numpy()
    step = ((wday[1:] - wday[:-1]) % 7) * 24 + hour[1:] - hour[:-1]  # assumes no gap of 7+ days
    elapsed = np.concatenate([[0], np.cumsum(step)])
    for lag in [1, 24, 168]:
        exact = (elapsed[lag:] - elapsed[:-lag]) == lag
        print(f"  lag_{lag}: exactly {lag} hours earlier in {exact.mean():.1%} of rows")
    print(f"  holdout: {HOLDOUT} rows span {elapsed[-1] - elapsed[-HOLDOUT] + 1} hours")

    print("\n[5] First training fold size")
    for kw in [{}, {"max_train_size": 10000}, {"test_size": 1344}]:
        tr, _ = next(TimeSeriesSplit(5, **kw).split(X_dev))
        print(f"  TimeSeriesSplit(5, {kw}): {len(tr)} rows")

    print("\n[6] TimeSeriesSplit(5, gap=168) row ranges on the dev set")
    for i, (tr, te) in enumerate(TimeSeriesSplit(5, gap=168).split(X_dev)):
        print(f"  fold {i}: train 0-{tr[-1]}  gap {tr[-1] + 1}-{te[0] - 1}  test {te[0]}-{te[-1]}")


if __name__ == "__main__":
    main()
