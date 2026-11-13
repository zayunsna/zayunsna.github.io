"""How big is the feature-selection leak as the number of features and rows changes?

Pure-noise data, random labels (true accuracy 0.5), SelectKBest(k=20) + LogisticRegression,
5-fold CV, 20 random datasets per setting.
"""
import numpy as np
import sklearn
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline

K, SEEDS = 20, range(20)


def leaky_and_clean(n_rows, n_features, seed):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_rows, n_features))
    y = rng.integers(0, 2, size=n_rows)
    X_selected = SelectKBest(f_classif, k=K).fit_transform(X, y)
    leaky = cross_val_score(LogisticRegression(), X_selected, y, cv=5).mean()
    pipe = make_pipeline(SelectKBest(f_classif, k=K), LogisticRegression())
    clean = cross_val_score(pipe, X, y, cv=5).mean()
    return leaky, clean


def report(label, n_rows, n_features):
    scores = np.array([leaky_and_clean(n_rows, n_features, s) for s in SEEDS])
    leaky, clean = scores[:, 0], scores[:, 1]
    print(f"{label:<22} leaky {leaky.mean():.3f} (min {leaky.min():.3f})   "
          f"clean {clean.mean():.3f}   gap {leaky.mean() - clean.mean():+.3f}")


print("200 rows, more noise features:")
for n_features in [20, 100, 1_000, 5_000, 20_000]:
    report(f"  {n_features:,} features", 200, n_features)

print("5,000 noise features, more rows:")
for n_rows in [100, 200, 1_000, 5_000]:
    report(f"  {n_rows:,} rows", n_rows, 5_000)

print("scikit-learn", sklearn.__version__, "| numpy", np.__version__)
