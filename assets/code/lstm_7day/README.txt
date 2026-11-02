7-Day LSTM Forecasting with PyTorch

Data: Hadi Fanaee-T (2013), UCI Bike Sharing, doi:10.24432/C5W894, CC BY 4.0.
https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset
Uses day.csv, 731 daily cnt values (2011-2012); only past counts are inputs.
The actual values in predictions.csv originate from that attributed dataset.
No account, API key, GPU or pretrained model required.

Download lstm_7day.py and requirements.txt into an empty folder:
  python3.12 -m venv .venv
  source .venv/bin/activate
  python -m pip install -r requirements.txt
  python lstm_7day.py
Results are written to results/. The UCI ZIP downloads on first use.
Existing cached ZIP must match the pinned SHA256; corrupt data is not replaced
silently. Remove or move a corrupt cache after investigating, then retry.

Inputs: 14 observed days; outputs: 7 future days in one forward pass.
Raw chronological split: 511 train / 110 validation / 110 test days.
Scaler fits training rows only; all 7 target dates must lie in one split.
Windows: train 491, validation 104, test 104; 12 boundary-crossing windows omitted.
Forecast origin is the last observed day, not the first target day.
Overlapping daily origins use newly observed history between origins, never
new observations inside one 7-day forecast. Weights stay frozen for test.
Do not interpret this as a single forecast of the full 110-day test interval.

80 epochs, 16 hidden units, 1 layer, head outputs 7 values; 1,335 parameters.
Training batch shuffle is over independent windows, not over chronological splits.
Checkpoint chosen using validation 7-output MAE. Test does not choose seed/epoch.
Default seed 0: best epoch 14, pooled test MAE 1327.8; repeat-last-value 1304.8,
repeat-last-week 1388.0. Additional seeds 1/2: LSTM 1309.8 / 1262.2.
These are sensitivity checks, not test-driven model selection or confidence bounds.
Overlapping horizons reuse target dates; the pooled MAE weights origin/horizon
pairs, not unique dates equally. Raw forecasts and horizon scores are included.
Repeated seed 0 produced identical predictions.csv and history.csv locally.
Tested 2026-10-30: Python 3.12.13, macOS M5, CPU one thread; pinned packages above.
Training approximately 1 second; one full invocation including data download
took 7.61 seconds and maximum resident memory 462356480 bytes (~441 MiB).
Installation is excluded; these are observations, not runtime guarantees.
Other operating systems and GPU paths were not tested.
