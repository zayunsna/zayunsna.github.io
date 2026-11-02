Retrieval metric arithmetic on four constructed query rankings, not model results.
Tested 2026-10-30, Python 3.12.13, numpy 2.5.4, scikit-learn 1.9.1.
  python3.12 -m venv .venv
  source .venv/bin/activate
  python -m pip install numpy==2.5.4 scikit-learn==1.9.1
  python retrieval_metrics.py
Optional JSON: python retrieval_metrics.py results.json
Grade > 0 is relevant; NDCG uses gain 2**grade-1, no predicted-score ties.
Relevance judgments cover the whole five-document corpus. IDs are unique.
No relevant documents: this demonstration raises; no hits: all metrics are 0.
sklearn comparison passes transformed gains as y_true, not raw grades.
Optional cover rendering: matplotlib==3.11.0, pillow==13.0.0.
