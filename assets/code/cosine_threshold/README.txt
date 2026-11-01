Cosine similarity is not confidence: choosing a retrieval threshold - files for the blog post (tested 2026-11-01)

cosine_threshold.py             Full script. Same code as the post's blocks, plus version logging, the score chart,
                                a 10-split robustness check and results files. Expects dev-v1.1.json (SQuAD v1.1 dev,
                                CC BY-SA 4.0) in the current directory. Models (pinned revisions) download on first run:
                                sentence-transformers/all-MiniLM-L6-v2 (Apache-2.0), BAAI/bge-small-en-v1.5 (MIT).
requirements.txt                Pinned package versions used for the recorded run (Python 3.12.13, macOS Apple M5, CPU).
reference_output.txt            Standard output of one complete run of cosine_threshold.py.
cosine_threshold_results.json   Chosen cutoffs, 10-split results, precision/coverage curves and median scores.
top_scores.csv                  Per query: model, split, index, best cosine score, annotated paragraph in top 3,
                                answerable (shorthand: annotated source is indexed; excluded articles are proxy negatives).
ct_cover.py                     Draws the post's cover image from results/cosine_threshold_results.json.

Run:  curl -O https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json
      python -m pip install -r requirements.txt && python cosine_threshold.py

Acceptance precision is a QUERY-bundle source-hit rate, not document precision@3 or generated-answer accuracy.
The 80% validation target is not guaranteed on test or new traffic. None cutoff raises an explicit error.
