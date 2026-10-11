---
layout: post
title: "Before replacing LightGBM with TabPFN-3.5, check your evaluation split"
description: >
  TabPFN-3.5's reported benchmark lead is a reason to evaluate, not a universal replacement rule.
  A local LightGBM demo scores 1.0000 AUC on random rows and 0.4748 on unseen groups; no TabPFN model is run.
image: /assets/img/blog/tabpfn_lightgbm_evaluation/cover.png
sitemap:
  changefreq: daily
  priority: 1.0
---

# Before replacing LightGBM with TabPFN-3.5, check your evaluation split

A perfect test score can disappear before you change models. In our constructed example, one LightGBM baseline
scored 1.0000 ROC AUC with random-row splitting and 0.4748 when testing unseen groups. The change was the evaluation
question, not a new foundation model.

Prior Labs released TabPFN-3.5 on September 15, 2026 and reported a lead on TabArena and BeyondArena. That makes it
a candidate worth investigating. It does not establish that it replaces LightGBM on your dataset, deployment,
or budget. These are developer-reported benchmark results, not our measurements. [Official technical report](https://priorlabs.ai/technical-reports/tabpfn-3-5), checked 2026-10-12.

_Sources and the LightGBM-only experiment verified as of 2026-10-12._

We did not run TabPFN in this article. Its published license restricts model and output use to defined non-commercial
purposes and requires separate authorization for uses outside that grant. We have not established that publishing
our own model outputs on an advertising-supported blog is permitted. This is a practical scope limitation, not legal
advice or a conclusion about every reader's use. [License text](https://huggingface.co/Prior-Labs/tabpfn_3_5/blob/main/LICENSE), checked 2026-10-12.

## Start with the prediction you actually need

Are you predicting another event for a customer already seen during training, or an event for a new customer?
Both can be legitimate tasks. A random-row split and a group-held-out split answer different questions.

For our example, create 200 fictional groups with 20 rows per group. Each group gets a random binary label that is
repeated for its rows. The features contain a numeric group ID and a noise column. There is no useful generalizable
relationship between an unseen ID and its target; repeated IDs can nevertheless make a random-row test easy.

```python
import numpy as np
from sklearn.model_selection import GroupShuffleSplit, train_test_split

rng = np.random.default_rng(7)
groups = np.repeat(np.arange(200), 20)
group_targets = rng.integers(0, 2, size=200)
X = np.column_stack([groups, rng.normal(size=len(groups))])
y = group_targets[groups]
indices = np.arange(len(groups))

random_split = train_test_split(indices, test_size=0.25, random_state=7, stratify=y)
group_split = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=7)
                   .split(X, y, groups))
```

The [complete executed script](/assets/code/hotweek/tabular_split_check.py) trains the same `LGBMClassifier` settings
on both splits: 200 trees, 31 leaves, one worker, seed 7, deterministic training, and column-wise histogram construction.
It records the number of shared group IDs and computes ROC AUC from predicted probabilities.

```bash
python -m pip install lightgbm==4.7.0 scikit-learn==1.9.1 numpy==2.5.3
python tabular_split_check.py --out tabular.json
```

Tested on 2026-10-12 with Python 3.12.13 and those package versions on Apple Silicon. On macOS, LightGBM may also
need the OpenMP runtime; see the setup in our [LightGBM ranking guide](/blog/2026-10-12-lightgbm_lambdarank/).

## One model, two test populations

| Split | Training rows | Test rows | Groups shared between train and test | ROC AUC |
|---|---:|---:|---:|---:|
| Random rows | 3,000 | 1,000 | 199 | 1.0000 |
| Unseen groups | 3,000 | 1,000 | 0 | 0.4748 |

![The same LightGBM baseline scores 1.0000 ROC AUC on random rows sharing 199 groups with training, and 0.4748 on held-out groups sharing none](/assets/img/blog/tabpfn_lightgbm_evaluation/cover.png)

This is synthetic data intentionally built to expose group memorization. One split and one seed do not estimate
LightGBM's performance on business data. An AUC below 0.5 in this run is not evidence that the model is systematically
anti-predictive. With random group targets and no useful unseen-group signal, small deviations around chance are expected.

Random-row splitting is not automatically wrong. It is wrong for a claim about unseen groups in this setup.
Similarly, a group-held-out split does not answer every question about future events. Time-dependent prediction
may require a chronological split, possibly combined with group constraints.
[scikit-learn's cross-validation guide](https://scikit-learn.org/stable/modules/cross_validation.html) explains these split families.

## What would make a replacement decision credible?

Specify the test population and metric before comparing candidates. Give both models the same eligible training
rows and a test set that reflects deployment. Tune only with training/validation data; do not choose settings from
the final test score. Group splitting is complementary to keeping learned preprocessing inside training folds,
as shown in our [preprocessing leakage demo](/blog/2026-10-02-data_leakage_preprocessing/).

Then measure operational constraints: prediction latency at the actual batch size, memory on the target hardware,
supported data types, update requirements, and permitted use. A benchmark rank cannot stand in for these checks.
This article provides a split diagnostic, not a TabPFN-versus-LightGBM accuracy, speed, or cost comparison.

The headline question “Should I replace LightGBM?” is best answered after “Does my test measure the predictions
I need?” Otherwise a more capable model may simply win a test that never represented the job.
