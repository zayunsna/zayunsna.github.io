---
layout: post
title: "Dynamic programming, step by step: Fibonacci, LCS and 0/1 knapsack in one app"
description: >
  Three classic dynamic programming problems drawn cell by cell in a small pygame app. Memoization cut Fibonacci(8)
  from 67 calls to 15, the LCS table picked a different answer when the strings were swapped, and the knapsack table
  found 44 where a value-per-weight greedy got 32.
image: /assets/img/blog/dp_visualized/cover.png
series: algorithm-lab
series_label: "Dynamic programming: Fibonacci, LCS and 0/1 knapsack"
sitemap:
  changefreq: daily
  priority: 1.0
---

# Dynamic programming, step by step: Fibonacci, LCS and 0/1 knapsack in one app

Dynamic programming (DP) has one idea: when a problem breaks into smaller problems that **repeat**, solve each one
once and reuse the answer. The repetition is easy to say and hard to picture, so this post draws it. The three
lessons come from the same pygame app as the [path search post](/blog/2026-10-19-pathfinding_visualized/), and
every number below came out of an actual run:

| Lesson | Input | Result |
|---|---|---|
| Fibonacci | n = 6 / n = 8 | plain recursion 25 / 67 calls, memoized 11 / 15 calls |
| Longest common subsequence | `ABCBDAB`, `BDCABA` | length 4: `BCBA` (swap the strings: `BDAB`) |
| 0/1 knapsack | weights 2, 4, 6; values 12, 20, 24; capacity 10 | best value 44 (items 2 and 3) |

_Tested on 2026-10-14 with Python 3.12.13 and pygame 2.6.1 on macOS. App source:
[github.com/zayunsna/Algorithm-visualizer](https://github.com/zayunsna/Algorithm-visualizer)._

## Fibonacci: see the repeated calls, then remove them

`F(n) = F(n-1) + F(n-2)` is the textbook recursion. Each call makes two more calls, and many of them ask the same
question. The app draws every call as a node and colours a node yellow when that `n` is asked more than once:

![Two call trees for Fibonacci of 6: the plain recursion tree has 25 nodes, most of them yellow repeats; the memoized tree below has 11 nodes, with teal cache hits](/assets/img/blog/dp_visualized/dp_fib6.png)

The top tree is plain recursion: 25 calls to get `F(6) = 8`. `F(2)` alone is computed five times. The bottom tree
is the same function with a cache: 11 calls. Each `F(k)` is computed once; the teal nodes are calls answered
from the cache.

In Python you rarely write that cache yourself. `functools.cache` does it, and `cache_info()` counts the calls:

```python
from functools import cache

calls = 0
def fib(n):
    global calls
    calls += 1
    return n if n < 2 else fib(n - 1) + fib(n - 2)

@cache
def fib_memo(n):
    return n if n < 2 else fib_memo(n - 1) + fib_memo(n - 2)

for n in (6, 8):
    calls = 0
    fib_memo.cache_clear()
    value = fib(n)
    fib_memo(n)
    info = fib_memo.cache_info()
    print(f"F({n}) = {value}: plain {calls} calls, memoized {info.hits + info.misses} calls "
          f"({info.misses} computed, {info.hits} cache hits)")
```

```text
F(6) = 8: plain 25 calls, memoized 11 calls (7 computed, 4 cache hits)
F(8) = 21: plain 67 calls, memoized 15 calls (9 computed, 6 cache hits)
```

Going from n = 6 to n = 8 nearly triples the plain calls (25 → 67) but adds only four memoized calls (11 → 15).
The plain count is `2·F(n+1) − 1`, so it grows like the Fibonacci numbers themselves, exponentially. The memoized
count is `2n − 1`, a straight line. The app stops at n = 8 because a bigger plain tree no longer fits on screen,
which is a decent picture of the problem too.

![The n equals 8 trees: the plain recursion tree is now 67 nodes wide and dense, while the memoized tree has only 15 nodes](/assets/img/blog/dp_visualized/dp_fib8.png)

## LCS: every cell looks at its neighbours

The **longest common subsequence** of two strings is the longest sequence of letters that appears in both, in the
same order but not necessarily side by side. `ABCBDAB` and `BDCABA` share `BCBA`, for example.

The DP table has one row per letter of the first string and one column per letter of the second. Cell `[i][j]`
holds the LCS length of the first `i` and first `j` letters. Each cell follows one rule:

- the two letters match → the cell diagonally up-left, plus 1
- they do not match → the larger of the cell above and the cell to the left

The app highlights the cells the current one depends on. Here the two `A`s match, so the cell takes the
diagonal value plus 1:

![LCS table early in the fill: the current cell for A versus A is orange, and the diagonal cell it depends on is highlighted blue](/assets/img/blog/dp_visualized/dp_lcs_deps.png)

<video controls muted loop playsinline preload="metadata" width="880" poster="/assets/img/blog/dp_visualized/dp_lcs_poster.png" aria-label="The LCS table fills row by row with each cell's dependencies highlighted, then a yellow trace walks back from the bottom-right corner to recover the subsequence BCBA">
  <source src="/assets/img/blog/dp_visualized/dp_lcs.mp4" type="video/mp4">
</video>

The table only gives the length (4, bottom-right). To get the letters, walk back from the corner: on a match go
diagonally and keep the letter, otherwise move toward the larger neighbour. In code:

```python
def lcs(a, b):
    t = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                t[i][j] = t[i - 1][j - 1] + 1
            else:
                t[i][j] = max(t[i - 1][j], t[i][j - 1])
    i, j, out = len(a), len(b), []          # trace back from the corner
    while i and j:
        if a[i - 1] == b[j - 1]:
            out.append(a[i - 1]); i -= 1; j -= 1
        elif t[i - 1][j] >= t[i][j - 1]:    # tie: go up
            i -= 1
        else:
            j -= 1
    return t[-1][-1], "".join(reversed(out))

print("LCS(ABCBDAB, BDCABA) =", lcs("ABCBDAB", "BDCABA"))
print("LCS(BDCABA, ABCBDAB) =", lcs("BDCABA", "ABCBDAB"))
```

```text
LCS(ABCBDAB, BDCABA) = (4, 'BCBA')
LCS(BDCABA, ABCBDAB) = (4, 'BDAB')
```

Look at the `>=` in the trace back. When the cell above and the cell to the left are equal, one has to be chosen,
and this code (like the app) goes up. That choice matters: swap the two strings and the same rule produces a
different answer, `BDAB`, also of length 4.

![The LCS table with the strings swapped: the yellow trace now spells BDAB, still length 4](/assets/img/blog/dp_visualized/dp_lcs_swapped.png)

Both are correct. An LCS is *a* longest common subsequence; there can be several. If your code must give a stable
answer, the tie rule is part of the specification.

## 0/1 knapsack: take it or skip it

Three items, a bag that holds weight 10. Each item can be taken once or not at all:

| Item | Weight | Value |
|---|---:|---:|
| 1 | 2 | 12 |
| 2 | 4 | 20 |
| 3 | 6 | 24 |

Row `i` of the table answers "using only the first `i` items, what is the best value for each capacity
0 to 10?". For each cell there are two choices:

- **skip** item `i`: the value in the row above, same capacity
- **take** item `i` (if it fits): the row above at capacity minus its weight, plus its value

![The finished knapsack table, 4 rows by 11 columns; the bottom-right cell is 44 and the yellow trace marks the cells that select items 2 and 3](/assets/img/blog/dp_visualized/dp_knapsack.png)

The "take" option looks at the **row above**, never the current row. That is what makes it 0/1: the row above
was built without item `i`, so item `i` can only be counted once.

The same table in Python, with the trace back:

```python
def knapsack(items, capacity):
    t = [[0] * (capacity + 1) for _ in range(len(items) + 1)]
    for i, (w, v) in enumerate(items, 1):
        for c in range(capacity + 1):
            t[i][c] = t[i - 1][c]                            # skip item i
            if w <= c:
                t[i][c] = max(t[i][c], t[i - 1][c - w] + v)  # or take it once
    chosen, c = [], capacity                                 # trace back
    for i in range(len(items), 0, -1):
        if t[i][c] != t[i - 1][c]:
            chosen.append(i); c -= items[i - 1][0]
    return t, t[-1][-1], sorted(chosen)

table, best, chosen = knapsack([(2, 12), (4, 20), (6, 24)], 10)
print("knapsack best value:", best, "items:", chosen)
for row in table:
    print(row)
```

```text
knapsack best value: 44 items: [2, 3]
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
[0, 0, 12, 12, 12, 12, 12, 12, 12, 12, 12]
[0, 0, 12, 12, 20, 20, 32, 32, 32, 32, 32]
[0, 0, 12, 12, 20, 20, 32, 32, 36, 36, 44]
```

The best bag is items 2 and 3: weight 10, value 44. Item 1 has the best value per weight (6 per unit), yet it is
not in the answer. Picking by value per weight takes items 1 and 2 and stops at 32, because item 3 no longer fits.
The **Compare greedy** button in the app runs exactly that comparison, and greedy choices are the topic of the next
post in this series.

All three snippets printed the same results as the app.

## How to tell a problem is a DP problem

Two signs, both visible in the pictures above:

1. **Repeated subproblems.** The yellow nodes in the Fibonacci tree. Without repetition, caching saves nothing.
2. **The answer is built from smaller answers.** Each LCS or knapsack cell needs only two or three cells that were
   already filled. If you can write that rule, you can fill the table.

The cost is memory: the LCS table has `(m+1)(n+1)` cells and the knapsack table `(items+1)(capacity+1)`. That is
fine at this size; for long strings or large capacities you can keep only the rows you still need, as long as you
do not need the trace back.

## Try it

```bash
git clone https://github.com/zayunsna/Algorithm-visualizer.git
cd Algorithm-visualizer
python3.12 -m venv .venv
.venv/bin/python -m pip install --only-binary=:all: -r requirements.txt
./run.sh
```

Pick **DP** in the lesson menu. Each lesson has three numbered steps (for LCS: **Fill table**, **Trace back**,
**Swap strings**), and you can type your own strings or items into the input fields. `→` and `←` step one event
forward and back.

{% include series.html title="Algorithm Lab series: one app, four kinds of algorithm" %}

Related: [What is a KV cache? Memoization for LLMs, measured](/tips/2026-10-20-kv_cache_memoization/)
