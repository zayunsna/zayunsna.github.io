"""Target encoding and leakage: plain mean encoding vs scikit-learn's TargetEncoder (cross fitting).

Data: OpenML 42125 "employee_salaries" (Montgomery County, MD, 2016, Public Domain CC0), 9,228 rows.
Target: current_annual_salary (USD). High-cardinality columns: division (694), employee_position_title (385).
Run: python target_encoding.py
"""
import sys

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import make_column_transformer
from sklearn.datasets import fetch_openml
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, TargetEncoder

CAT = ["department", "division", "assignment_category", "employee_position_title", "underfilled_job_title"]
NUM = ["year_first_hired"]
SEED = 0
CV = KFold(5, shuffle=True, random_state=SEED)


class MeanEncoder(BaseEstimator, TransformerMixin):
    """Plain target mean per category, unseen categories get the overall mean. No smoothing, no cross fitting."""

    def fit(self, X, y):
        y = pd.Series(np.asarray(y), index=X.index)
        self.global_ = y.mean()
        self.maps_ = {c: y.groupby(X[c]).mean() for c in X.columns}
        return self

    def transform(self, X):
        return np.column_stack([X[c].map(self.maps_[c]).astype(float).fillna(self.global_) for c in X.columns])


def target_encoder():
    return TargetEncoder(cv=KFold(5, shuffle=True, random_state=SEED))   # default smoothing="auto"


def regressor(name):
    return HistGradientBoostingRegressor(random_state=SEED) if name == "HGB" else make_pipeline(StandardScaler(), RidgeCV())


def pipeline(encoder, cats, reg="HGB"):
    pre = make_column_transformer((encoder, cats), ("passthrough", NUM))
    return make_pipeline(pre, regressor(reg))


def add_random_id(X, rows_per_id):
    """A column with no information about salary: random IDs, each shared by about `rows_per_id` rows."""
    ids = np.random.default_rng(SEED).integers(0, int(len(X) / rows_per_id), len(X))
    return X.assign(random_id=ids.astype(str))


def mae(model, X, y):
    return mean_absolute_error(y, model.predict(X))


def part1_leak(X, y):
    """Encode before the split vs inside the pipeline. CV on 80%, then a holdout nobody touched."""
    Xd, Xh, yd, yh = train_test_split(X, y, test_size=0.2, random_state=SEED)
    print(f"\n[1] Where the encoder is fitted. 5-fold CV on {len(Xd)} rows, then a holdout of {len(Xh)} rows")
    print(f"  {'strategy':36s} {'CV MAE':>8} {'holdout MAE':>12}")

    # Mean codes computed once from every development row, then CV on the encoded table.
    enc = MeanEncoder().fit(Xd[CAT], yd)
    Ed = np.column_stack([enc.transform(Xd[CAT]), Xd[NUM]])
    cv = [mae(regressor("HGB").fit(Ed[tr], yd.iloc[tr]), Ed[te], yd.iloc[te]) for tr, te in CV.split(Ed)]
    final = regressor("HGB").fit(Ed, yd)
    hold = mae(final, np.column_stack([enc.transform(Xh[CAT]), Xh[NUM]]), yh)
    print(f"  {'mean encoding before the split':36s} {np.mean(cv):8,.0f} {hold:12,.0f}")
    out = {"before the split": (np.mean(cv), hold)}

    for name, enc in [("mean encoding in the pipeline", MeanEncoder()), ("TargetEncoder in the pipeline", target_encoder())]:
        cv = [mae(pipeline(enc, CAT).fit(Xd.iloc[tr], yd.iloc[tr]), Xd.iloc[te], yd.iloc[te]) for tr, te in CV.split(Xd)]
        hold = mae(pipeline(enc, CAT).fit(Xd, yd), Xh, yh)
        print(f"  {name:36s} {np.mean(cv):8,.0f} {hold:12,.0f}")
        out[name.replace("mean encoding ", "").replace("TargetEncoder ", "TE ")] = (np.mean(cv), hold)
    return out


def cv_scores(make, X, y):
    tr_s, te_s = [], []
    for tr, te in CV.split(X):
        m = make().fit(X.iloc[tr], y.iloc[tr])
        tr_s.append(mae(m, X.iloc[tr], y.iloc[tr]))
        te_s.append(mae(m, X.iloc[te], y.iloc[te]))
    return np.mean(tr_s), np.mean(te_s)


def part2_rare(X, y):
    """Add an ID-like column with no signal and see which encoding the model is fooled by."""
    print("\n[2] Adding a random ID column (no information). 5-fold CV MAE, train / test")
    print(f"  {'random_id':22s} {'model':6s} {'mean encoding':>18s} {'TargetEncoder':>18s}")
    rows = []
    for per in [None, 3, 1.5]:
        Xr, cats = (X, CAT) if per is None else (add_random_id(X, per), CAT + ["random_id"])
        label = "none" if per is None else f"~{per} rows per ID ({Xr.random_id.nunique():,})"
        for reg in ["HGB", "Ridge"]:
            plain = cv_scores(lambda: pipeline(MeanEncoder(), cats, reg), Xr, y)
            te = cv_scores(lambda: pipeline(target_encoder(), cats, reg), Xr, y)
            print(f"  {label:22s} {reg:6s} {plain[0]:8,.0f} / {plain[1]:6,.0f}  {te[0]:8,.0f} / {te[1]:6,.0f}")
            rows.append({"random_id": "none" if per is None else f"~{per} rows per ID", "model": reg,
                         "mean train": plain[0], "mean test": plain[1], "TE train": te[0], "TE test": te[1]})
    return pd.DataFrame(rows)


def part3_fit_transform(X, y):
    """TargetEncoder.fit_transform is not fit().transform() on the same rows."""
    Xr = add_random_id(X, 1.5)
    tr, te = next(CV.split(Xr))
    Xtr, Xte, ytr, yte = Xr.iloc[tr], Xr.iloc[te], y.iloc[tr], y.iloc[te]
    col = Xtr[["random_id"]]
    corr = lambda a, b: np.corrcoef(a, b)[0, 1]
    plain = MeanEncoder().fit(col, ytr)
    te_fitted = target_encoder().fit(col, ytr)
    codes = {"mean encoding": (plain.transform(col)[:, 0], plain.transform(Xte[["random_id"]])[:, 0]),
             "TargetEncoder.fit_transform(X, y)": (target_encoder().fit_transform(col, ytr)[:, 0],
                                                    te_fitted.transform(Xte[["random_id"]])[:, 0]),
             "TargetEncoder.fit(X, y).transform(X)": (te_fitted.transform(col)[:, 0],
                                                       te_fitted.transform(Xte[["random_id"]])[:, 0])}
    print("\n[3] Correlation between the encoded random_id and salary (fold 1, ~1.5 rows per ID)")
    print(f"  {'how the training rows were encoded':40s} {'train rows':>10} {'test rows':>10}")
    for name, (a, b) in codes.items():
        print(f"  {name:40s} {corr(a, ytr):10.3f} {corr(b, yte):10.3f}")
    pd.DataFrame({"mean encoding": codes["mean encoding"][0],
                  "TargetEncoder fit_transform": codes["TargetEncoder.fit_transform(X, y)"][0],
                  "salary": ytr.to_numpy()}).to_csv("te_random_id_codes.csv", index=False)

    # Same thing end to end: the model trained on fit().transform() codes.
    enc = target_encoder().fit(Xtr[CAT + ["random_id"]], ytr)
    Ztr = np.column_stack([enc.transform(Xtr[CAT + ["random_id"]]), Xtr[NUM]])
    Zte = np.column_stack([enc.transform(Xte[CAT + ["random_id"]]), Xte[NUM]])
    m = regressor("HGB").fit(Ztr, ytr)
    p = pipeline(target_encoder(), CAT + ["random_id"]).fit(Xtr, ytr)
    print(f"  HGB test MAE, fold 1: pipeline (fit_transform) {mae(p, Xte, yte):,.0f}, "
          f"by hand fit().transform() {mae(m, Zte, yte):,.0f}")


def main():
    print(f"python {sys.version.split()[0]}, scikit-learn {sklearn.__version__}, pandas {pd.__version__}, "
          f"numpy {np.__version__}")
    df = fetch_openml(data_id=42125, as_frame=True).frame
    y = df["current_annual_salary"].astype(float)
    X = df[CAT + NUM].copy()
    X[CAT] = X[CAT].astype(str)
    print(f"rows {len(X)}, mean salary {y.mean():,.0f} USD; unique values: "
          + ", ".join(f"{c} {X[c].nunique()}" for c in CAT))

    leak = part1_leak(X, y)
    rare = part2_rare(X, y)
    part3_fit_transform(X, y)
    pd.DataFrame(leak, index=["cv", "holdout"]).T.to_csv("te_leak.csv")
    rare.to_csv("te_rare.csv", index=False)


if __name__ == "__main__":
    main()
