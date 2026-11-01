"""Building a scikit-learn Preprocessing Pipeline: the full tutorial script.

UCI Adult data (Becker & Kohavi, 1996, CC BY 4.0, https://doi.org/10.24432/C5XW20).
Run: python pipeline_tutorial.py   (writes results/ under the current directory)
The steps match the blog post; this file adds version logging, the data chart and scores.json.
"""
import json
import platform
import sys
import time

STARTED = time.perf_counter()
import io
import urllib.request
import zipfile
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

OUT = Path("results")
OUT.mkdir(exist_ok=True)
SEED = 0

print(f"python {sys.version.split()[0]}, scikit-learn {sklearn.__version__}, pandas {pd.__version__}, "
      f"numpy {np.__version__}, matplotlib {matplotlib.__version__}, joblib {joblib.__version__}, "
      f"{platform.system()} {platform.machine()}")

# ---- 2. download and load ----
URL = "https://archive.ics.uci.edu/static/public/2/adult.zip"
COLUMNS = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
           "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
           "hours-per-week", "native-country", "income"]
archive = OUT / "adult.zip"
if not archive.exists():
    request = urllib.request.Request(URL, headers={"User-Agent": "Pipeline-Tutorial/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        archive.write_bytes(response.read())
with zipfile.ZipFile(io.BytesIO(archive.read_bytes())) as zipped:
    df = pd.read_csv(zipped.open("adult.data"), names=COLUMNS,
                     skipinitialspace=True, na_values="?")
df = df.dropna(how="all")  # guard against a trailing blank line
print(df.shape)
print(df["income"].value_counts().to_string())

# ---- 3. inspect ----
missing = df.isna().sum()
print(missing[missing > 0].to_string())
print("Duplicate rows:", df.duplicated().sum())
countries = df["native-country"].value_counts()
print("Countries:", len(countries), "| with fewer than 50 rows:", (countries < 50).sum())
print(countries.iloc[[0, 1, -3, -2, -1]].to_string())

# ---- 4. split ----
NUMERIC = ["age", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
CATEGORICAL = ["workclass", "marital-status", "occupation", "relationship", "race", "sex",
               "native-country"]
X = df[NUMERIC + CATEGORICAL]
y = (df["income"] == ">50K").astype(int)
X_train, X_rest, y_train, y_rest = train_test_split(
    X, y, test_size=0.4, stratify=y, random_state=SEED)
X_val, X_test, y_val, y_test = train_test_split(
    X_rest, y_rest, test_size=0.5, stratify=y_rest, random_state=SEED)
for name, part, label in [("train", X_train, y_train), ("validation", X_val, y_val),
                          ("test", X_test, y_test)]:
    print(f"{name:10s} {part.shape}  share >50K: {label.mean():.3f}")
train_missing = X_train.isna().sum()
print(train_missing[train_missing > 0].to_string())
print("Train countries with < 50 rows:", (X_train["native-country"].value_counts() < 50).sum())


def plot_data_check(X_train, path):
    """Missing values per column and rows per country, both on the training split."""
    bg, fg, muted, accent, warn = "#1e1f26", "#e8e8ef", "#9a9ab0", "#4fb3bf", "#f2a65a"
    plt.rcParams.update({"figure.facecolor": bg, "axes.facecolor": bg, "axes.edgecolor": muted,
                         "axes.labelcolor": fg, "xtick.color": fg, "ytick.color": fg,
                         "text.color": fg, "font.size": 12})
    miss = X_train.isna().sum().sort_values()
    counts = X_train["native-country"].value_counts()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6), width_ratios=[1, 1.3])
    ax1.barh(miss.index, miss.to_numpy(), color=[warn if v else muted for v in miss.to_numpy()])
    for i, v in enumerate(miss.to_numpy()):
        if v:
            ax1.text(v + 20, i, f"{v:,}", va="center", color=warn)
    ax1.set_xlim(0, miss.max() * 1.3)
    ax1.set_xlabel("missing values (training rows)")
    ax1.set_title("Gaps only in three text columns", loc="left")
    ax2.bar(range(len(counts)), counts.to_numpy(),
            color=[accent if v >= 50 else warn for v in counts.to_numpy()])
    ax2.axhline(50, color=fg, ls="--", lw=1)
    ax2.text(len(counts) - 1, 60, "min_frequency = 50", ha="right", va="bottom", color=fg)
    ax2.set_yscale("log")
    ax2.set_xticks([0, len(counts) - 1], [counts.index[0], counts.index[-1]])
    ax2.set_xlabel(f"{len(counts)} countries, most to fewest rows")
    ax2.set_ylabel("training rows (log scale)")
    ax2.set_title(f"{(counts < 50).sum()} countries below 50 rows (orange)", loc="left")
    for ax in (ax1, ax2):
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


plot_data_check(X_train, OUT / "data_check.png")

# ---- 5. baseline ----
baseline = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
print(f"baseline validation accuracy: {accuracy_score(y_val, baseline.predict(X_val)):.3f}")

# ---- 6. pipeline ----
numeric_steps = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
])
categorical_steps = Pipeline([
    ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
    ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=50,
                             sparse_output=False)),
])
preprocess = ColumnTransformer([
    ("num", numeric_steps, NUMERIC),
    ("cat", categorical_steps, CATEGORICAL),
])
model = Pipeline([
    ("preprocess", preprocess),
    ("classify", LogisticRegression(max_iter=1000)),
])
model.fit(X_train, y_train)

features = model.named_steps["preprocess"].get_feature_names_out()
print("Input columns:", X_train.shape[1], "-> model features:", len(features))
print(features[:2].tolist(), "...", features[-2:].tolist())
medians = model.named_steps["preprocess"].named_transformers_["num"].named_steps["impute"].statistics_
print("Learned medians:", dict(zip(NUMERIC, medians.tolist())))

# ---- 7. choose C on validation ----
val_scores = {}
for C in [0.01, 0.1, 1.0, 10.0]:
    model.set_params(classify__C=C).fit(X_train, y_train)
    val_scores[C] = accuracy_score(y_val, model.predict(X_val))
    print(f"C={C:<5} validation accuracy: {val_scores[C]:.4f}")
best_C = max(val_scores, key=val_scores.get)
model.set_params(classify__C=best_C).fit(X_train, y_train)
print("chosen C:", best_C)

# ---- 8. test once ----
test_proba = model.predict_proba(X_test)[:, 1]
test_pred = model.predict(X_test)
scores = {
    "baseline_accuracy": accuracy_score(y_test, baseline.predict(X_test)),
    "pipeline_accuracy": accuracy_score(y_test, test_pred),
    "pipeline_roc_auc": roc_auc_score(y_test, test_proba),
}
for name, value in scores.items():
    print(f"{name:18s} {value:.4f}")

# ---- 9. raw rows and unknown categories ----
person = {"age": 45, "workclass": np.nan, "education-num": 13, "marital-status": "Married-civ-spouse",
          "occupation": "Prof-specialty", "relationship": "Husband", "race": "White", "sex": "Male",
          "capital-gain": 0, "capital-loss": 0, "hours-per-week": np.nan}
new_rows = pd.DataFrame([{**person, "native-country": "Brazil"},
                         {**person, "native-country": "Scotland"}])
imputed = model.named_steps["preprocess"].named_transformers_["cat"].named_steps["impute"].transform(new_rows[CATEGORICAL])
assert (imputed[:, CATEGORICAL.index("workclass")] == "missing").all()
print("Training rows: Brazil", (X_train["native-country"] == "Brazil").sum(),
      "| Scotland", (X_train["native-country"] == "Scotland").sum())
print("P(>50K):", model.predict_proba(new_rows)[:, 1].round(4).tolist())
encoded = model.named_steps["preprocess"].transform(new_rows)
print("Same encoded row:", np.array_equal(encoded[0], encoded[1]))
print("Infrequent-country column:", encoded[:, -1].tolist())

strict = OneHotEncoder(sparse_output=False).fit(X_train[["native-country"]].fillna("missing"))
try:
    strict.transform(new_rows[["native-country"]])
except ValueError as error:
    print("Default OneHotEncoder:", error)

# ---- 9b. rare-country group on test ----
onehot = model.named_steps["preprocess"].named_transformers_["cat"].named_steps["onehot"]
rare = onehot.infrequent_categories_[CATEGORICAL.index("native-country")]
is_rare = X_test["native-country"].isin(rare).to_numpy()
print("Countries merged into infrequent:", len(rare))
for name, mask in [("rare-country rows", is_rare), ("all other rows", ~is_rare)]:
    print(f"{name:17s} n={mask.sum():5d}  share >50K={y_test[mask].mean():.3f}  "
          f"accuracy={accuracy_score(y_test[mask], test_pred[mask]):.3f}")

# ---- 10. save and load ----
joblib.dump(model, OUT / "income_pipeline.joblib")
loaded = joblib.load(OUT / "income_pipeline.joblib")
print("Loaded predictions match:",
      np.array_equal(loaded.predict_proba(X_test), model.predict_proba(X_test)))
pd.DataFrame({"y_true": y_test.to_numpy(), "p_over_50k": test_proba}).to_csv(
    OUT / "test_predictions.csv", index=False)

summary = {**scores, "chosen_C": best_C, "validation_accuracy": {str(k): v for k, v in val_scores.items()},
           "n_rows": len(df), "n_train": len(X_train), "n_val": len(X_val), "n_test": len(X_test),
           "n_features": len(features), "n_infrequent_countries": len(rare),
           "new_rows_p_over_50k": model.predict_proba(new_rows)[:, 1].tolist()}
(OUT / "scores.json").write_text(json.dumps(summary, indent=1) + "\n")
print(f"total run time: {time.perf_counter() - STARTED:.1f} s")
