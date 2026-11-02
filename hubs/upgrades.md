---
layout: page
title: Upgrade Traps
short: what changes, often silently, when you upgrade pandas, Polars, Python, scikit-learn or PyTorch
description: >
  Measured behavior changes between library versions: the same script run on the old and new release, with the output diff.
permalink: /upgrades/
hub: upgrades
hub_order: 1
---

Each post here runs one script on two releases of a library and reports what changed in the output.
Some of the changes raise no warning at all. In Polars 2.1 on a 10-core M5, a lazy join followed by `group_by().first()`
picked the wrong "first purchase" in 0.2% to 99.1% of rows depending on the run, while Polars 1.44 got every row right.
In Python 3.15 with pandas 3.1.0, `pd.DataFrame(frozendict(a=[1, 2], b=[3, 4]))` built a single column holding the keys.

The scripts behind each result are published with the post, so you can rerun them against the version you are about to install.

{% include hub.html %}
