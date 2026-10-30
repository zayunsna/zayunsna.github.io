---
layout: post
title: "AI code review vs unit tests: what catches which bugs?"
description: >
  Two independent AI reviews found the same six bugs in a small Python fixture; a baseline test suite caught three.
  Turn the missing cases into regression tests, then see why a second review is not a replacement for assertions.
image: /assets/img/blog/ai_review_vs_tests/cover.png
sitemap:
  changefreq: daily
  priority: 1.0
---

# AI code review vs unit tests: what catches which bugs?

An AI review can suggest a case your tests never exercise. A test can check that case every time the code changes.
Those are different jobs. Adding another reviewer does not automatically give you another layer of protection.

In this demonstration, two fresh AI reviewers each identified the same **six buggy functions**. The second review
added **zero new defects**. A small baseline suite failed on three of those functions and passed on the other three.
After adding the missing cases, the expanded suite failed on all six buggy functions; the corrected code passed
all 11 tests.

This is not a score showing that AI beats tests. The fixture and incomplete tests were deliberately constructed
to make their different roles visible. Better initial tests would change the result without changing the reviewer.

## The code and rules were frozen before the reviews

I prepared eight short Python helpers: six with seeded defects and two correct controls. Each had an explicit
contract. For example, pagination was 1-based, a percentile used the nearest-rank definition, and an empty mean
had to return `None`. Invalid types and inputs outside the documented domain were not part of the task.

The fixture, contract, baseline tests, extra regression cases and reference implementation were all written
before either review returned. Neither reviewer saw the tests, reference implementation, defect count or the
other review. Each received the same substantive instruction: inspect the code against the contract, report a
valid counterexample, and omit style advice. They did not execute the code. I executed the claims afterwards.

Both were separate Claude Code sessions, not two different model vendors. Both reported `claude-opus-5-5` as
their session model; I did not independently verify that identifier. An intended Codex review did not start
because of an update prompt, so it is **not** included in the results. This is a static-review exercise, not a
comparison of coding agents allowed to run tests.

_Tested on 2026-10-30 with Python 3.12.13 and pytest 9.2.0 on macOS, Apple M5, CPU. Reviews used Claude Code
2.1.312 through subscription sessions, without paid AI API calls. Token consumption, monetary cost and reliable
review latency were not measured. The [review instruction](/assets/code/ai_review_vs_tests/review_prompt.txt),
[contract](/assets/code/ai_review_vs_tests/requirements.txt) and verbatim
[review A](/assets/code/ai_review_vs_tests/review_a.txt) / [review B](/assets/code/ai_review_vs_tests/review_b.txt)
are available with the code._

## What each check found

“Caught by tests” below means a baseline assertion actually failed, not that the function was called or covered.
“Reported” means a reviewer named the defect and supplied a counterexample that I subsequently reproduced.

| Function | Defect | Baseline test failed | Review A reported | Review B reported |
|---|---|---|---|---|
| `page` | Starts page 1 at the second page | Yes | Yes | Yes |
| `discounted_price` | Subtracts a percentage as a flat amount | Yes | Yes | Yes |
| `cache_key` | Drops the tenant, so IDs collide | Yes | Yes | Yes |
| `p95` | Uses the wrong rank for 20 values | No | Yes | Yes |
| `word_count` | Splits only on single spaces | No | Yes | Yes |
| `mean_or_none` | Divides by zero for an empty list | No | Yes | Yes |

Neither reviewer flagged `stable_unique` or `clamp`, the two correct controls, under the supplied contract.
That is no observed false alarm on these two functions, not evidence of a generally low false-positive rate.
Both reviews overlapped completely on the seeded defects.

## A passing test that barely tests the rule

Here is the intentionally broken percentile function:

```python
def p95(values):
    ordered = sorted(values)
    return ordered[int(0.95 * len(ordered))]

assert p95([42]) == 42
print(p95(list(range(1, 21))))
```

Output:

```text
20
```

The singleton assertion passes. But the contract asks for rank `ceil(0.95 * n)`, counted from 1. For 20 values,
that is rank 19, hence Python index 18. The implementation uses index 19 and returns 20 instead.

The two reviewers noticed the mismatch between the contract and the index. The baseline test did not: its
only percentile input had one element. Calling the function was not the same as testing the rank calculation.

The same gap appeared with `word_count("hello world")`, which passes despite mishandling repeated whitespace,
and `mean_or_none([2, 4])`, which says nothing about an empty list.

## Turn a finding into something executable

The three missing cases are short:

```python
def test_p95_twenty_values():
    assert p95(list(range(1, 21))) == 19

def test_whitespace():
    assert word_count(" \thello   world\n ") == 2
    assert word_count("") == 0
    assert word_count(" \t\n ") == 0

def test_empty_mean():
    assert mean_or_none([]) is None
```

These use the helpers imported by the full [test file](/assets/code/ai_review_vs_tests/test_regressions.py).
On the broken implementation, the first fails with `20 != 19`, the whitespace case fails with `6 != 2`, and
the empty mean raises `ZeroDivisionError`. The failures establish concrete behavior, rather than agreement
between reviewers.

The tests were prepared in advance for this controlled demonstration. In an actual review, write such tests
from the finding and the intended contract, not by copying the current output into an assertion.

Download the files listed in the [run instructions](/assets/code/ai_review_vs_tests/README.txt) into one folder.
After installing `pytest==9.2.0` in a Python 3.12 virtual environment, run:

```bash
python run_checks.py --implementation broken --suite baseline
python run_checks.py --implementation broken --suite all
python run_checks.py --implementation fixed --suite all
```

The saved results are:

| Implementation | Test suite | Result |
|---|---|---|
| Intentionally broken | Baseline, 8 tests | 3 failed, 5 passed |
| Intentionally broken | Expanded, 11 tests | 6 failed, 5 passed |
| Corrected | Expanded, 11 tests | 11 passed |

The first two commands intentionally exit with status 1. The runner selects an implementation in memory; it
does not overwrite the broken fixture. Full outputs are saved as
[baseline](/assets/code/ai_review_vs_tests/baseline_out.txt),
[expanded on broken code](/assets/code/ai_review_vs_tests/expanded_broken_out.txt) and
[expanded on corrected code](/assets/code/ai_review_vs_tests/fixed_out.txt).
The [counterexample verifier](/assets/code/ai_review_vs_tests/verify_findings.py) also reproduces the concrete
claims from both reports. In pytest, assertions check the expected behavior and explain failures; see its
[assertion guide](https://docs.pytest.org/en/stable/how-to/assert.html) (checked 2026-10-30).

## What I would keep in a real workflow

Run the existing tests first. Give a reviewer the intended behavior and relevant callers, then ask for a
reproducible failing input rather than a list of possible concerns. Check the input against the contract,
run it, and preserve the confirmed failure as a regression test before applying a fix.

A second reviewer can be useful for a different perspective, but count **new, verified defects**, not comments
or repeated agreement. Here it added none. With only eight synthetic functions, one review per session and
deliberately incomplete tests, this run cannot estimate production detection rates, the value of multiple
model vendors, or a cost-benefit ratio.

AI suggestions still need checking. GitHub's own
[responsible-use documentation](https://docs.github.com/en/copilot/responsible-use/agents#7-limitations)
warns of missed issues, false positives and incorrect fixes in code review. That guidance concerns Copilot,
which was not tested here; it is a reason to verify suggestions, not another result in this experiment.

The useful output of a review is not “two agents agreed.” It is a test that fails for the right reason and
keeps checking that reason after the review session ends.

For the mechanics of handing a read-only review to another agent, see the
[Orca walkthrough](/tips/2026-10-23-orca_claude_codex_review/). That earlier post explains the tool;
this one checks what a review result adds to executable tests.
