Your first LSTM forecast in PyTorch

Run the complete tutorial locally with Python 3.12. Commands below use a macOS/Linux shell:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python lstm_tutorial.py
```

Only the first run needs to download the data. The model runs on CPU with one thread. Internet access is needed
to install packages and download the dataset; no API key or paid service is required. Tested on macOS, Apple M5;
other operating systems and accelerators have not been tested.

## Files and outputs

- `lstm_tutorial.py`: download and validate the data, plot it, build windows, train, choose a validation checkpoint,
  and evaluate the frozen model on rolling one-day-ahead test targets.
- `requirements.txt`: directly used package versions from the tested environment.
- `reference_output.txt`: the seed-0 reference output, including variable training time.
- `render_cover.py`: optionally turn the tutorial's saved predictions and scores into the blog cover:
  `python render_cover.py results cover.png`. Requires the same environment.

The main script creates `results/` containing the downloaded archive, `timeline.png`,
`training_and_predictions.png`, `history.csv`, `predictions.csv`, and `scores.csv`.
Rerunning overwrites generated results in that output directory; the downloaded ZIP is reused.
Change the directory with `LSTM_OUTPUT_DIR=another_run python lstm_tutorial.py`.
Change the seed with `LSTM_SEED=1 python lstm_tutorial.py`; compare choices on validation, not test scores.

The default is 14 days of observed counts, 16 hidden units, 80 epochs, Adam learning rate 0.003, batch size 32,
seed 0, and chronological 70/15/15 percent splits by target date. Every window starts with a fresh hidden state.
The earliest 14 training dates supply context, leaving 497 training windows and 110 validation and test windows each.
The test is **not** a 110-day forecast from a single origin: observed counts update the input history daily.
Only validation chooses the checkpoint. The model is not retrained during the test period.

Seed 0 selects epoch 13. Test MAE: yesterday 926.6, last week 1382.1, LSTM 1117.2 rentals/day.
An identical-seed repeat matched every saved prediction. Additional fixed seeds 1 and 2 gave LSTM MAE
1060.2 and 1046.5; neither beat yesterday. Seeds were not selected on test scores. Reproducibility across
PyTorch releases or platforms is not guaranteed.

## Data attribution

Fanaee-T, H. (2013). *Bike Sharing* [Dataset]. UCI Machine Learning Repository.
DOI: https://doi.org/10.24432/C5W894
Dataset: https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset
License: [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/).

The tutorial selects the original daily CSV, sorts by date, uses `cnt`, scales with training-set statistics,
and derives input windows and result plots. Dataset dates are 2011-01-01 through 2012-12-31.
Downloaded ZIP SHA-256, checked 2026-10-28:
`b70182d0d0508e9abbb79306ce5c0cec34869000f8220175ac83d11dbe845401`.
If the archive is corrupt, remove only that downloaded `results/bike_sharing.zip` and retry; inspect any
failed date or missing-value assertion rather than bypassing it.
