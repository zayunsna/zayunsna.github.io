"""CPU time to score k (question, paragraph) pairs with the cross-encoder, 100 SQuAD questions, median of 3 passes.

Run: python cpu_timing.py path/to/dev-v1.1.json
"""
import json
import random
import statistics
import sys
import time

import torch
from sentence_transformers import CrossEncoder

paragraphs = [p["context"] for a in json.load(open(sys.argv[1]))["data"] for p in a["paragraphs"]]
questions = [qa["question"] for a in json.load(open(sys.argv[1]))["data"] for p in a["paragraphs"] for qa in p["qas"]]
rng = random.Random(0)
qs = rng.sample(questions, 100)
cross = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2", device="cpu")
cross.predict([(qs[0], paragraphs[0])] * 8)
print(f"torch {torch.__version__}, {torch.get_num_threads()} CPU threads")
for k in (10, 20, 50):
    runs = []
    for _ in range(3):
        t0 = time.perf_counter()
        for q in qs:
            cross.predict([(q, p) for p in rng.sample(paragraphs, k)], batch_size=k)
        runs.append((time.perf_counter() - t0) / len(qs) * 1000)
    print(f"k={k:3d}: {statistics.median(runs):6.1f} ms per query (runs: {', '.join(f'{r:.1f}' for r in runs)})")
