"""Evaluate one LightGBM baseline under two splits; this is NOT a TabPFN comparison."""
import argparse
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import sklearn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    rng = np.random.default_rng(7)
    groups = np.repeat(np.arange(200), 20)
    group_targets = rng.integers(0, 2, size=200)
    X = np.column_stack([groups, rng.normal(size=len(groups))])
    y = group_targets[groups]
    indices = np.arange(len(groups))
    random_split = train_test_split(indices, test_size=0.25, random_state=7, stratify=y)
    group_split = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=7).split(X, y, groups))
    rows = []
    for label, (train, test) in [('random rows', random_split), ('unseen groups', group_split)]:
        model = lgb.LGBMClassifier(n_estimators=200, num_leaves=31, min_child_samples=5,
                                  random_state=7, n_jobs=1, verbosity=-1, deterministic=True, force_col_wise=True)
        start = time.perf_counter()
        model.fit(X[train], y[train])
        seconds = time.perf_counter() - start
        overlap = len(set(groups[train]) & set(groups[test]))
        score = roc_auc_score(y[test], model.predict_proba(X[test])[:, 1])
        row = {'split': label, 'train_rows': len(train), 'test_rows': len(test),
               'shared_groups': overlap, 'auc': round(score, 4), 'fit_seconds': round(seconds, 4)}
        rows.append(row)
        print(json.dumps(row))
    assert rows[0]['shared_groups'] > 0 and rows[1]['shared_groups'] == 0
    args.out.write_text(json.dumps({'lightgbm': lgb.__version__, 'numpy': np.__version__,
                                   'sklearn': sklearn.__version__, 'results': rows}, indent=2))


if __name__ == '__main__':
    main()
