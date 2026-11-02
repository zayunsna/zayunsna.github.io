---
layout: page
title: Validation and Leakage
short: preprocessing leaks, split mistakes, and metrics that look better than the model is
description: >
  Experiments on how preprocessing, target encoding, splits and metrics inflate validation scores, and how to set them up correctly.
permalink: /leakage/
hub: leakage
hide_description: true
hub_order: 2
---

A validation score is only as honest as the order of the steps that produced it. The posts here measure the gap directly:
on the OpenML employee salaries data, mean-encoding the categories on all rows before cross-validation reported an MAE of 4,054 USD,
and the same model scored 4,636 on a held-out set.

Start with [the preprocessing leak](/blog/2026-10-02-data_leakage_preprocessing/) if you fit scalers or encoders yourself, then the
[threshold](/blog/2026-10-27-decision_threshold_cost/) and [baseline](/tips/2026-10-02-baseline_first/) posts
if your metric looks good but the decisions built on it do not.

{% include hub.html %}
