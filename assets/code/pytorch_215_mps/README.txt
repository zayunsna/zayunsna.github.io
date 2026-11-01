MPS inference version comparison (tested 2026-11-01)

Python 3.12.13; macOS 27.0.1 arm64; Apple M5, 10 CPU cores, 16 GiB unified RAM.
Actual available MPS required. No CPU fallback; no download, API key, or training.
Random untrained 4,722,176-parameter MLP: Linear(1024,2048), GELU,
Linear(2048,1024), GELU, Linear(1024,512). FP32, eval + inference_mode.
NumPy generator seed 20261101 regenerates identical weights and inputs.

Create two virtual environments from requirements_214.txt and requirements_215.txt.
In each, run:
  python mps_benchmark.py --output result.json
For the reference experiment six new Python processes were run sequentially in
order 2.14, 2.15, 2.15, 2.14, 2.14, 2.15 (three runs/version).
No other coordinator experiment ran alongside this final sequence. Desktop
background applications were not isolated; this is not a controlled lab run.
Exploratory runs before this sequence are not included in reference results.

Each process: CPU reference and upload first (setup_ms, not timed inference);
batch 1 first forward, 20 warm-up calls, 21 windows of 20 calls;
then same steps for batch 64. Every forward synchronized. Warm statistic:
median of window means per process, then median/range over 3 process statistics.
Includes Python dispatch/synchronization overhead; not isolated device kernel time.
First-forward samples are fresh PROCESS observations, not uncached OS shader
compilation benchmarks. Shader caches were not cleared, and batch64 follows batch1.

Each full output tensor asserted close to its CPU reference at atol=2e-5,
rtol=2e-4. JSON includes maximum absolute error and a short output excerpt.
The fixture SHA256 covers all weight/bias bytes and full 64-row input.
Six run_N_VERSION.json files and summary.json preserve raw timings.
Reference reports (same public folder as this README):
  run_1_214.json, run_2_215.json, run_3_215.json,
  run_4_214.json, run_5_214.json, run_6_215.json
Recompute the saved summary (same folder as the downloaded run reports):
  python summarize_runs.py run_*.json --output summary.json
No FP8, torch.compile, attention, trained-model accuracy, LLM decoding, or MoE test.

Cover: install matplotlib==3.11.0 and run
  python render_cover.py --output cover.png
The plot reads summary.json. Error bars are observed ranges, not confidence intervals.
