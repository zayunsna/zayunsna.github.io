---
layout: post
title: "Claude Code writes, Codex reviews: a cross-check workflow with Orca"
description: >
  One real session: Claude Code drafted a blog post, then handed a read-only review to Codex through Orca's
  orchestration commands. Codex reran the experiment, confirmed every number and flagged 9 issues. The exact
  commands, what each step returns, and where it still needs a human.
image: /assets/img/tips/orca_claude_codex_review/cover.png
sitemap:
  changefreq: daily
  priority: 1.0
---

# Claude Code writes, Codex reviews: a cross-check workflow with Orca

A coding agent is good at checking code and bad at checking its own reasoning. A cheap fix is a second agent from a
different vendor that reviews the first one's work. [Orca](https://github.com/stablyai/orca) is an open-source
(MIT) desktop app for running several CLI coding agents side by side, and its `orca` command line lets one agent hand a
task to another and wait for the answer.

This post follows one real session. Claude Code wrote this week's post on
[time-series cross-validation](/blog/2026-10-23-timeseries_cv_leakage/), then asked Codex, running in the next pane,
for a read-only review. Codex reran the experiment, confirmed that every number matched, and flagged 9 issues, three of
them serious enough to change what the post claimed.

_Session on 2026-10-19 with Orca 1.4.220 on macOS, Claude Code (Claude Opus 5.5) and Codex CLI 0.162.1. Both agents use
their own subscriptions; Orca adds no API key. Day to day, the blog's author switches between the two agents by hand; this
session used Orca's orchestration commands on purpose so the steps could be recorded._

![Orca with Claude Code on the left and Codex on the right, reviewing the same draft](/assets/img/tips/orca_claude_codex_review/split.png)

_The two panes during the review. The Claude Code session on the left is in Korean; the Codex pane on the right has just
read the scikit-learn documentation and is asking permission to report progress. Two local paths and an internal
runtime ID are redacted._

## The setup: two panes, one repository

Both agents work in the same folder. Claude Code was already open in an Orca terminal tab; the right pane was opened by
splitting that tab and starting `codex` in it. The same split can be made from the command line:

```bash
orca terminal split --terminal <claude-pane-handle> --command "codex"
```

`orca terminal list` shows each pane's handle (`term_...`). The handle is how one agent addresses the other.

## The hand-off: a run, a task, a dispatch

Orca's orchestration layer has three objects. A **run** is a mailbox for one piece of coordinated work. A **task** is the
work itself, written as a self-contained instruction. A **dispatch** is one attempt at the task in one terminal.
Claude Code ran these from its own pane:

```bash
orca orchestration run-create --objective "Cross-review the 2026-10-23 blog draft ..." --json
orca orchestration task-create --run run_5d67d5be5900 --task-title "Codex review: time-series CV leakage draft" \
  --spec "Read-only cross-review of a blog draft. Do NOT edit ... Check (1) every number against the recorded output; ..." --json
orca orchestration dispatch --task task_5123ba324078 --to term_769e7c95-... --run run_5d67d5be5900 --inject --json
```

`--inject` types the task into the Codex prompt, below a short header with the task and dispatch IDs. The task spec named
the files, what to check, what not to touch (no edits, no git), and what a finished report must contain. Orca's bundled
orchestration guide asks for exactly those parts: target, change, constraints, ownership and observable acceptance.

## Waiting for the answer

Claude Code then blocked on the run's mailbox:

```bash
orca orchestration check --wait --types "worker_done,escalation,question" --timeout-ms 1500000 --json
```

Two messages arrived: a heartbeat (`"phase": "reviewing"`) about two minutes after the dispatch, then the result about
five minutes after it. After each one, the coordinator acknowledges the delivery (`check --ack <delivery_id>`) so the
next can arrive. The run's record afterwards:

```text
$ orca orchestration run-show --id run_5d67d5be5900
run_5d67d5be5900 Cross-review the 2026-10-23 blog draft on KFold vs TimeSeriesSplit leakage (Claude writes, Codex reviews read-only)
consumer generation 1; created 2026-10-19T04:23:43Z

$ orca orchestration task-list --run run_5d67d5be5900
task_5123ba324078 [completed] Codex reviewer
```

## What the review caught

The result came as one `worker_done` message with `"outcome": "succeeded"`. Codex wrote it in Korean because the
session was in Korean; summarised in English:

- **Numbers:** it reran the script, and every MAE, fold score and row count matched the recorded output.
- **Wrong advice (serious):** the draft said `max_train_size` helps when early folds have too little data. It doesn't;
  it only caps the training size. Codex measured it: the first fold stayed at 2,647 rows with `max_train_size=10000`,
  and grew to 9,147 with `test_size=1344`.
- **Wrong advice (serious):** the draft said a weekly lag feature needs `gap`. Lags built only from past values don't.
- **Missing condition (serious):** the holdout fed real observed values into the lag columns, so the result measured
  one-step-ahead prediction, not an 8-week forecast. The draft didn't say so.
- **Smaller points:** "hours" that were really rows (lag_168 is exactly a week earlier in only 68.8% of rows), a
  train/serve mismatch result that needed its own name, two over-confident sentences, and cover text sitting a little
  too close to the edge.

Claude Code reproduced each factual claim before changing the post, and all of them held. The published post is the
corrected version.

## Where it still needs a human

- **Approvals.** With default settings, Codex asked for permission before running the `orca` reporting commands (the
  cover image shows one of those prompts). Someone had to press "Yes" for the heartbeat and the final report to go out.
  Codex offers "don't ask again for commands that start with `orca orchestration`", which removes that stop.
- **Judgement.** The reviewer flags; it does not decide. Each serious point still had to be checked, and accepting a
  wrong suggestion would have made the post worse, not better.
- **Clean-up depends on who opened the pane.** `orca orchestration worker-release` returned `"state": "retained"` with
  `"reason": "no_owned_resource"`: the Codex pane was opened by hand, so Orca left it running. A worker started with
  `orca orchestration worker-start` is owned by the run and can be released by it.

## When this is worth it

For a quick question, typing into the second agent's pane is enough. The run, task and dispatch pay off when you want
the review written down: a self-contained spec, a result with an explicit outcome, and a record you can look up later
with `run-show`. Here it took about five minutes of Codex time and caught three mistakes that a quick reread had missed.

Related: [Why fit preprocessing on training data only? A data leakage demo](/blog/2026-10-02-data_leakage_preprocessing/) ·
[How do you evaluate LLM answers? Exact match vs key facts vs LLM-as-judge](/tips/2026-10-13-evaluate_llm_answers/)
