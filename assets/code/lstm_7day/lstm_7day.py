"""Direct 7-day forecasts from UCI Bike Sharing counts, CPU only.

Run in an empty folder: python lstm_7day.py
Data: Fanaee-T (2013), doi:10.24432/C5W894, CC BY 4.0.
"""

# %% Environment and reproducibility
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import time
import urllib.request
import zipfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

WINDOW, HORIZON, EPOCHS = 14, 7, 80
SEED = int(os.environ.get('LSTM_SEED', '0'))
OUT = Path(os.environ.get('LSTM_OUTPUT_DIR', 'results'))
OUT.mkdir(parents=True, exist_ok=True)
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)
plt.style.use('dark_background')
plt.rcParams.update({'figure.facecolor': '#162335', 'axes.facecolor': '#162335',
                     'axes.grid': True, 'grid.alpha': .15})
print(f'Python {platform.python_version()} | torch {torch.__version__} | pandas {pd.__version__} | seed {SEED}')

# %% Load and validate the real daily series
url = 'https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip'
archive = OUT / 'bike_sharing.zip'
if not archive.exists():
    request = urllib.request.Request(url, headers={'User-Agent': 'SevenDayLSTM/1.0'})
    with urllib.request.urlopen(request, timeout=60) as response:
        archive.write_bytes(response.read())
assert hashlib.sha256(archive.read_bytes()).hexdigest() == 'b70182d0d0508e9abbb79306ce5c0cec34869000f8220175ac83d11dbe845401'
with zipfile.ZipFile(io.BytesIO(archive.read_bytes())) as zipped:
    df = pd.read_csv(zipped.open('day.csv'), parse_dates=['dteday'])
df = df.sort_values('dteday').set_index('dteday')
assert df.index.is_unique and df.index.is_monotonic_increasing
assert df.index.to_series().diff().iloc[1:].eq(pd.Timedelta(days=1)).all()
assert df['cnt'].notna().all() and df['cnt'].ge(0).all()
values = df[['cnt']].to_numpy(dtype=np.float32)
train_end, val_end = int(len(df) * .70), int(len(df) * .85)
print('Rows:', len(df), '| raw split boundaries:', train_end, val_end)

# %% Plot the chronology before fitting
fig, ax = plt.subplots(figsize=(10, 3.5))
ax.plot(df.index, values[:, 0], lw=.8, color='#96bfff')
for index, label in [(train_end, 'Validation'), (val_end, 'Test')]:
    ax.axvline(df.index[index], ls='--', label=label)
ax.set(xlabel='Date', ylabel='Rentals / day', title='Daily counts: chronological train / validation / test')
ax.legend()
fig.tight_layout()
fig.savefig(OUT / 'timeline.png', dpi=140)
plt.close(fig)

# %% Build windows: all seven targets must stay inside their split
scaler = StandardScaler().fit(values[:train_end])
scaled = scaler.transform(values).astype(np.float32)
t = np.arange(WINDOW, len(values) - HORIZON + 1)
X = np.stack([scaled[i-WINDOW:i] for i in t])
y = np.stack([scaled[i:i+HORIZON, 0] for i in t])
masks = [t + HORIZON <= train_end,
         (t >= train_end) & (t + HORIZON <= val_end), t >= val_end]
(X_train, y_train), (X_val, y_val), (X_test, y_test) = [
    (torch.from_numpy(X[m]), torch.from_numpy(y[m])) for m in masks]
assert np.array_equal(X[0, :, 0], scaled[:WINDOW, 0])
assert np.array_equal(y[0], scaled[WINDOW:WINDOW+HORIZON, 0])
assert (t[masks[0]][-1] + HORIZON - 1) < train_end
assert (t[masks[1]][-1] + HORIZON - 1) < val_end
for name, xx, yy in [('Train', X_train, y_train), ('Validation', X_val, y_val), ('Test', X_test, y_test)]:
    print(name + ':', tuple(xx.shape), tuple(yy.shape))
print('Windows omitted at split boundaries:', len(t) - sum(int(m.sum()) for m in masks))

# %% One encoder, seven simultaneous outputs
class SevenDayLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(1, 16, batch_first=True)
        self.head = nn.Linear(16, HORIZON)

    def forward(self, x):
        sequence, _ = self.lstm(x)
        return self.head(sequence[:, -1])

model = SevenDayLSTM()
loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
optimizer = torch.optim.Adam(model.parameters(), lr=.003)
loss_fn = nn.MSELoss()
print('Parameters:', sum(p.numel() for p in model.parameters()))

# %% Select weights on validation MAE, not on test
best_val, best_epoch, best_state = float('inf'), 0, None
history = []
started = time.perf_counter()
for epoch in range(1, EPOCHS + 1):
    model.train()
    for xb, yb in loader:
        optimizer.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        val_mae = (model(X_val) - y_val).abs().mean().item()
    history.append([epoch, val_mae * scaler.scale_[0]])
    if val_mae < best_val:
        best_val, best_epoch = val_mae, epoch
        best_state = copy.deepcopy(model.state_dict())
training_seconds = time.perf_counter() - started
model.load_state_dict(best_state)
model.eval()
print(f'Best epoch: {best_epoch}; validation MAE: {best_val * scaler.scale_[0]:.1f}')
print(f'Training only: {training_seconds:.2f} s (one CPU thread)')

# %% Evaluate frozen weights at rolling forecast origins
test_t = t[masks[2]]
with torch.no_grad():
    lstm = model(X_test).numpy() * scaler.scale_[0] + scaler.mean_[0]
actual = np.stack([values[i:i+HORIZON, 0] for i in test_t])
forecasts = {'Last value': np.repeat(values[test_t-1], HORIZON, axis=1),
             'Last week': np.stack([values[i-7:i, 0] for i in test_t]), 'LSTM': lstm}
assert actual.shape == lstm.shape == (len(test_t), HORIZON)
scores = {}
records = []
print('Method       h1      h2      h3      h4      h5      h6      h7     Mean')
for name, prediction in forecasts.items():
    errors = np.abs(actual - prediction)
    by_horizon = errors.mean(axis=0)
    scores[name] = dict(horizon=by_horizon.tolist(), mean=float(errors.mean()))
    print(f'{name:11s} ' + ' '.join(f'{v:7.1f}' for v in by_horizon) + f'  {errors.mean():7.1f}')
for j, i in enumerate(test_t):
    for h in range(HORIZON):
        records.append(dict(origin=str(df.index[i-1].date()), target=str(df.index[i+h].date()),
                            horizon=h+1, actual=float(actual[j, h]),
                            **{name: float(pred[j, h]) for name, pred in forecasts.items()}))
pd.DataFrame(records).to_csv(OUT / 'predictions.csv', index=False)
pd.DataFrame(history, columns=['epoch', 'validation_mae']).to_csv(OUT / 'history.csv', index=False)
summary = dict(seed=SEED, best_epoch=best_epoch, training_seconds=training_seconds, scores=scores,
               origins=len(test_t), first_origin=str(df.index[test_t[0]-1].date()),
               last_origin=str(df.index[test_t[-1]-1].date()))
(OUT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print('Forecast origins:', summary['origins'], summary['first_origin'], 'to', summary['last_origin'])

# %% Plot horizon errors and the first test forecast
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for name, prediction in forecasts.items():
    axes[0].plot(range(1, HORIZON+1), scores[name]['horizon'], marker='o', label=name)
    axes[1].plot(df.index[test_t[0]:test_t[0]+HORIZON], prediction[0], marker='o', label=name)
axes[0].set(xlabel='Days ahead', ylabel='MAE / rentals per day', title='104 rolling origins; seven outputs each')
axes[1].plot(df.index[test_t[0]:test_t[0]+HORIZON], actual[0], color='white', ls='--', label='Actual')
axes[1].set(xlabel='Target date', ylabel='Rentals / day', title='First test forecast: no updates inside seven days')
for ax in axes:
    ax.legend(fontsize=8)
axes[1].tick_params(axis='x', rotation=25)
fig.tight_layout()
fig.savefig(OUT / 'forecast_results.png', dpi=140)
plt.close(fig)
