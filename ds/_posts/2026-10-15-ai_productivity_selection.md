---
layout: post
title: "Does AI make developers slower? A selection-bias example changes the headline"
description: >
  A synthetic task-selection example flips a 32.5% time reduction into a 10% increase without changing any task.
  Use it to understand why METR's follow-up is not a simple verdict on AI coding productivity.
image: /assets/img/ds/ai_productivity_selection/cover.png
sitemap:
  changefreq: daily
  priority: 1.0
---

# Does AI make developers slower? A selection-bias example changes the headline

You can get “AI made work faster” and “AI made work slower” from the same task list without changing a single
completion time. Change which tasks enter the study, and you change the question the result answers.

METR's early-2025 experiment found a slowdown in its setting. In a February 24, 2026 follow-up, the researchers
explained that participant and task selection made the newer results an unreliable signal of current productivity.
People and tasks most expected to benefit from AI could be missing. That is a limitation of the evidence, not proof
that the opposite headline is now universally true. [METR's follow-up](https://metr.org/blog/2026-02-24-uplift-update/), checked 2026-10-12.

This is an explainer of an ongoing debate, not an October announcement. We did not run a developer trial or measure
any AI coding product. The numbers below are deliberately constructed to make selection bias visible.

## A task list with two kinds of work

Imagine 100 tasks. Without assistance, each takes 60 minutes. Assistance cuts 50 tasks to 15 minutes and increases
the other 50 to 66 minutes. Now suppose people submit only the latter tasks because they do not want the high-benefit
tasks assigned to a condition without assistance.

These are assumptions of the example, not claims about real developers. Run the complete calculation:

```python
tasks = [(60, 15)] * 50 + [(60, 66)] * 50
submitted = tasks[50:]

def report(name, rows):
    manual = sum(m for m, a in rows)
    assisted = sum(a for m, a in rows)
    change = 100 * (assisted / manual - 1)
    print(f"{name}: {len(rows)} tasks, {manual} -> {assisted} minutes, {change:+.1f}%")

report("all tasks", tasks)
report("submitted tasks", submitted)
```

Executed on 2026-10-12 with Python 3.14; no third-party packages:

```text
all tasks: 100 tasks, 6000 -> 4050 minutes, -32.5%
submitted tasks: 50 tasks, 3000 -> 3300 minutes, +10.0%
```

The sign refers to change in completion time: negative means less time, positive means more. A 32.5% reduction in
time is not the same number as a 32.5% increase in throughput. The [JSON-producing script](/assets/code/hotweek/selection_bias_demo.py)
uses the same tasks and asserts that the two signs differ.

![Synthetic completion-time changes: all 100 tasks show a 32.5 percent reduction, while selecting only 50 low-benefit tasks shows a 10 percent increase](/assets/img/ds/ai_productivity_selection/cover.png)

Nothing about the assisted times changed. Neither result contains a calculation error. They summarize different
populations: the full list and the submitted subset. The mistake would be treating the subset as representative
of all the work people do.

## Why random assignment does not fix every selection problem

Randomly allowing or disallowing AI for submitted tasks helps compare conditions within those submitted tasks.
It cannot restore tasks that never entered the experiment. Selection before assignment and biased assignment are
different failure modes.

Our toy data assumes both potential completion times are known for every task. A real experiment usually observes
only the condition actually assigned. That makes our arithmetic a teaching example, not a method for recovering
the missing real-world productivity effect.

The selection direction also depends on why tasks are missing. We intentionally excluded high-benefit tasks here.
If a different process excludes difficult low-benefit tasks, the estimate could be biased in the other direction.
Do not turn this demonstration into a rule that every unfavorable AI study must underestimate gains.

## What to record in your own workflow

If you want to decide whether an assistant helps your team, define a task and a completion criterion before timing.
Include review, debugging, tests, and rework, rather than stopping the timer when code first appears. Record which
tasks were excluded and why, including failures and abandoned attempts.

Keep output quality separate from elapsed time. A faster change that fails acceptance tests is not a successful
completion. Equally, doing more experiments after assistance becomes cheap may improve value without reducing the
time you spend working. “Time per fixed task” and “value produced in a week” are different outcomes.

For a small personal log, use it as descriptive evidence. Do not call a before-and-after comparison causal when
task difficulty, experience, and tool versions changed together. A credible team experiment needs a design that
handles those differences, not just a larger spreadsheet.

The useful response to a provocative headline is to identify its population, tasks, outcome, and uncertainty.
That habit also matters in [LLM answer evaluation](/tips/2026-10-13-evaluate_llm_answers/): a score is only meaningful
once you know what was counted and what was left out.
