Building a scikit-learn Preprocessing Pipeline - files for the blog post (tested 2026-11-01)

pipeline_tutorial.py   Full tutorial script. Same code as the post's blocks, plus version logging, the data chart
                       and results/scores.json. Downloads UCI Adult (CC BY 4.0, doi:10.24432/C5XW20) on first run.
requirements.txt       Pinned package versions used for the recorded run (Python 3.12.13, macOS Apple M5, CPU).
reference_output.txt   Standard output of one complete run of pipeline_tutorial.py.
scores.json            Test/validation scores and counts written by that run.
test_predictions.csv   y_true and predicted P(>50K) for the 6,513 test rows from that run.
pp_cover.py            Draws the post's cover image from results/scores.json.

Run:  python -m pip install -r requirements.txt && python pipeline_tutorial.py
