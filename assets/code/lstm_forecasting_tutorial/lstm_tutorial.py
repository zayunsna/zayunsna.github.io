"""A CPU-only, one-day-ahead LSTM tutorial using UCI Bike Sharing day.csv.

Run: python lstm_tutorial.py
Outputs: results/ (or the LSTM_OUTPUT_DIR environment variable).
Data: Fanaee-T (2013), https://doi.org/10.24432/C5W894, CC BY 4.0.
"""

# %% Setup
import copy
import io
import os
import platform
import time
import urllib.request
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

SEED = int(os.environ.get("LSTM_SEED", "0"))
WINDOW = 14
EPOCHS = 80
OUT = Path(os.environ.get("LSTM_OUTPUT_DIR", "results"))
OUT.mkdir(parents=True, exist_ok=True)
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)
plt.style.use("dark_background")
plt.rcParams.update({"figure.figsize": (10, 3.8), "axes.grid": True,
                     "figure.facecolor": "#151c2c", "axes.facecolor": "#151c2c",
                     "grid.alpha": 0.15})
print(f"Python {platform.python_version()}, torch {torch.__version__}, "
      f"numpy {np.__version__}, pandas {pd.__version__}, sklearn {sklearn.__version__}")

# %% Load and check the daily data
URL = "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"
archive = OUT / "bike_sharing.zip"
if not archive.exists():
    request = urllib.request.Request(URL, headers={"User-Agent": "LSTM-Tutorial/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        archive.write_bytes(response.read())
with zipfile.ZipFile(io.BytesIO(archive.read_bytes())) as zipped:
    df = pd.read_csv(zipped.open("day.csv"), parse_dates=["dteday"])
df = df.sort_values("dteday").set_index("dteday")
assert df.index.is_unique and df.index.is_monotonic_increasing
assert df.index.to_series().diff().iloc[1:].eq(pd.Timedelta(days=1)).all()
assert df["cnt"].notna().all() and df["cnt"].ge(0).all()
values = df[["cnt"]].to_numpy(dtype=np.float32)
train_end, val_end = int(len(df) * 0.70), int(len(df) * 0.85)
print(f"Rows: {len(df)}; {df.index[0].date()} to {df.index[-1].date()}")
print(f"Rows before windowing: train={train_end}, "
      f"validation={val_end - train_end}, test={len(df) - val_end}")

# %% Plot the timeline before fitting anything
fig, ax = plt.subplots()
ax.plot(df.index, values[:, 0], lw=0.8, color="#8cb8ff")
ax.axvline(df.index[train_end], color="#d18a26", label="Validation starts")
ax.axvline(df.index[val_end], color="#b54966", label="Test starts")
ax.set(xlabel="Date", ylabel="Rentals per day", title="Daily bike rentals: keep the future at the end")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "timeline.png", dpi=140)
plt.close(fig)

# %% Scale on training rows and build windows by target date
scaler = StandardScaler().fit(values[:train_end])
scaled = scaler.transform(values).astype(np.float32)
target_idx = np.arange(WINDOW, len(values))
X = np.stack([scaled[t - WINDOW:t] for t in target_idx])
y = scaled[target_idx]
assert X.shape == (len(df) - WINDOW, WINDOW, 1)
assert np.array_equal(X[0, :, 0], scaled[:WINDOW, 0])
assert y[0, 0] == scaled[WINDOW, 0]
masks = [target_idx < train_end,
         (target_idx >= train_end) & (target_idx < val_end),
         target_idx >= val_end]
(X_train, y_train), (X_val, y_val), (X_test, y_test) = [
    (torch.from_numpy(X[m]), torch.from_numpy(y[m])) for m in masks
]
print("Train:", tuple(X_train.shape), tuple(y_train.shape))
print("Validation:", tuple(X_val.shape), tuple(y_val.shape))
print("Test:", tuple(X_test.shape), tuple(y_test.shape))
print(f"First window: {df.index[0].date()} to {df.index[WINDOW - 1].date()} "
      f"-> target {df.index[WINDOW].date()}")

# %% Define a small, stateless LSTM
class DailyLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=16, batch_first=True)
        self.head = nn.Linear(16, 1)

    def forward(self, x):
        sequence, _ = self.lstm(x)
        return self.head(sequence[:, -1, :])

model = DailyLSTM()  # CPU; no hidden state is carried between windows.
loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
loss_fn = nn.MSELoss()
print("Parameters:", sum(p.numel() for p in model.parameters()))

# %% Train, selecting the checkpoint on validation MAE only
history = []
best_val, best_epoch, best_state = float("inf"), 0, None
started = time.perf_counter()
for epoch in range(1, EPOCHS + 1):
    model.train()
    total = 0.0
    for xb, yb in loader:
        optimizer.zero_grad()
        prediction = model(xb)
        loss = loss_fn(prediction, yb)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(xb)
    model.eval()
    with torch.no_grad():
        val_mae = (model(X_val) - y_val).abs().mean().item()
    history.append([epoch, total / len(X_train), val_mae])
    if val_mae < best_val:
        best_val, best_epoch = val_mae, epoch
        best_state = copy.deepcopy(model.state_dict())
training_seconds = time.perf_counter() - started
model.load_state_dict(best_state)
model.eval()
history = pd.DataFrame(history, columns=["epoch", "train_mse_scaled", "val_mae_scaled"])
history.to_csv(OUT / "history.csv", index=False)
print(f"Best epoch: {best_epoch}; validation MAE: {best_val * scaler.scale_[0]:.1f} rentals/day")
print(f"Training only: {training_seconds:.2f} s (CPU, one thread)")

# %% Evaluate once in original units against two simple baselines
with torch.no_grad():
    predicted = scaler.inverse_transform(model(X_test).numpy()).ravel()
idx = target_idx[masks[2]]
actual = values[idx, 0]
predictions = {"Yesterday": values[idx - 1, 0],
               "Last week": values[idx - 7, 0], "LSTM": predicted}
scores = {}
for name, forecast in predictions.items():
    scores[name] = float(np.mean(np.abs(actual - forecast)))
    print(f"{name:10s} test MAE: {scores[name]:.1f} rentals/day")
result = pd.DataFrame({"date": df.index[idx], "actual": actual, **predictions})
result.to_csv(OUT / "predictions.csv", index=False)
pd.Series(scores, name="test_mae_rentals_per_day").to_csv(OUT / "scores.csv")
print(f"Test targets: {df.index[idx[0]].date()} to {df.index[idx[-1]].date()}")

fig, axes = plt.subplots(2, 1, figsize=(10, 6.4))
axes[0].plot(history.epoch, np.sqrt(history.train_mse_scaled) * scaler.scale_[0], label="Train RMSE")
axes[0].plot(history.epoch, history.val_mae_scaled * scaler.scale_[0], label="Validation MAE")
axes[0].axvline(best_epoch, color="gray", ls="--", label="Selected epoch")
axes[0].set(xlabel="Epoch", ylabel="Rentals per day", title="Different metrics: train RMSE and validation MAE")
axes[0].legend()
axes[1].plot(result.date, actual, color="#e8eef8", lw=1.1, label="Actual")
axes[1].plot(result.date, predicted, color="#4cc5bc", lw=1.2, label="LSTM: one day ahead")
axes[1].set(xlabel="Target date", ylabel="Rentals per day", title="Frozen model; update observed history after each day")
axes[1].legend()
fig.tight_layout()
fig.savefig(OUT / "training_and_predictions.png", dpi=140)
plt.close(fig)
