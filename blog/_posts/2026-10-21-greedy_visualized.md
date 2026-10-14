---
layout: post
title: "When does a greedy algorithm fail? Coins, meetings and knapsacks, checked against the optimum"
description: >
  Three greedy rules run next to an exact answer in a small pygame app. Greedy change-making used 3 coins where 2 were
  enough, "earliest start" booked 1 meeting where 3 fit, and value-per-weight packing got 32 where 44 was possible.
  The rules that do work are just as concrete.
image: /assets/img/blog/greedy_visualized/cover.png
series: algorithm-lab
series_label: "Greedy algorithms: when the best-looking choice fails"
sitemap:
  changefreq: daily
  priority: 1.0
---

# When does a greedy algorithm fail? Coins, meetings and knapsacks, checked against the optimum

A greedy algorithm makes the choice that looks best right now and never takes it back. It is fast and easy to
write. Sometimes it is also the optimal answer, and sometimes it is not, and the difference is not obvious from the
code.

So every lesson in this post runs the greedy rule **next to an exact answer**: a dynamic programming table or a
check of every possible subset. All results come from the same pygame app as the
[DP post](/blog/2026-10-20-dp_visualized/):

| Problem | Greedy rule | Greedy | Optimum |
|---|---|---:|---:|
| Coins {1, 3, 4}, amount 6 | largest coin that fits | 3 coins | **2 coins** |
| Coins {1, 5, 10, 25}, amount 63 | largest coin that fits | 6 coins | 6 coins |
| 4 meetings | earliest start | 1 meeting | **3 meetings** |
| 3 meetings | shortest first | 1 meeting | **2 meetings** |
| Both meeting sets | earliest finish | 3 and 2 | 3 and 2 |
| Bag of capacity 10, whole items | best value per weight | 32 | **44** |
| Same bag, items can be split | best value per weight | 48 | 48 |

_Tested on 2026-10-14 with Python 3.12.13 and pygame 2.6.1 on macOS. App source:
[github.com/zayunsna/Algorithm-visualizer](https://github.com/zayunsna/Algorithm-visualizer)._

## Coins: the largest coin first

To make 6 from coins {1, 3, 4}, greedy takes the largest coin that fits: 4, then 1, then 1. Three coins. But
3 + 3 is two. Taking the 4 left a remainder of 2 that only 1s can fill, and greedy never reconsiders.

The app finds the real minimum with a small DP table: `dp[a]` is the fewest coins for amount `a`, computed as
`1 + min(dp[a − c])` over the coins `c` that fit.

![Coin lesson finished: greedy used 4, 1, 1 for three coins, the DP row from 0 to 6 shows the optimum of two coins, 3 and 3](/assets/img/blog/greedy_visualized/gr_coin.png)

With US coins {1, 5, 10, 25} and amount 63, greedy and DP agree: 25 + 25 + 10 + 1 + 1 + 1, six coins. Coin systems
where greedy is always optimal are called **canonical**. One amount is not proof, so I checked every amount from
1 to 1,000:

```python
def greedy_coins(coins, amount):
    used = []
    for c in sorted(coins, reverse=True):
        while amount >= c:
            amount -= c; used.append(c)
    return used

def fewest_coins(coins, amount):
    best = [[]] + [None] * amount
    for a in range(1, amount + 1):
        options = [best[a - c] + [c] for c in coins if c <= a and best[a - c] is not None]
        best[a] = min(options, key=len) if options else None
    return best[amount]

for coins, amount in (([1, 3, 4], 6), ([1, 5, 10, 25], 63)):
    print(coins, amount, "greedy:", greedy_coins(coins, amount), "fewest:", fewest_coins(coins, amount))

for coins in ([1, 5, 10, 25], [1, 3, 4]):
    bad = [a for a in range(1, 1001) if len(greedy_coins(coins, a)) != len(fewest_coins(coins, a))]
    print(coins, "greedy is not optimal for", len(bad), "amounts in 1..1000, first:", bad[:5])
```

```text
[1, 3, 4] 6 greedy: [4, 1, 1] fewest: [3, 3]
[1, 5, 10, 25] 63 greedy: [25, 25, 10, 1, 1, 1] fewest: [25, 25, 10, 1, 1, 1]
[1, 5, 10, 25] greedy is not optimal for 0 amounts in 1..1000, first: []
[1, 3, 4] greedy is not optimal for 249 amounts in 1..1000, first: [6, 10, 14, 18, 22]
```

With {1, 3, 4} greedy is wrong for about a quarter of all amounts. Whether greedy works depends on the coins, not
on the algorithm. And a checked range is strong evidence, not a proof for every amount.

## Meetings: which rule do you sort by?

One room, a list of meetings, and you want to fit as many as possible without overlaps. The greedy plan: sort the
meetings by some rule, then go down the list and take every meeting that does not clash with one already taken.
The only question is the sort rule.

**Earliest start** sounds reasonable. Here it books the long meeting from 0 to 10 first, and that blocks the three
short ones:

![Earliest start rule: meeting 1 from 0 to 10 is selected and drawn green, meetings 2, 3 and 4 are rejected in purple, while the yellow underlines mark the optimal set of three short meetings](/assets/img/blog/greedy_visualized/gr_activity_start.png)

**Shortest first** handles that input, but the app has a counterexample for it too: a short meeting from 3 to 5
sits across the boundary of two meetings (0–4 and 4–8) that would both fit. Shortest first takes it and gets 1
meeting instead of 2.

![Shortest-first rule: the short meeting from 3 to 5 is selected and blocks the two meetings 0 to 4 and 4 to 8 that form the optimal set](/assets/img/blog/greedy_visualized/gr_activity_duration.png)

**Earliest finish** gets both inputs right. The video runs earliest start, then earliest finish, on the first set.
After each greedy pass the app checks all 16 subsets to find the true maximum:

<video controls muted loop playsinline preload="metadata" width="880" poster="/assets/img/blog/greedy_visualized/gr_activity_poster.png" aria-label="Earliest start picks the long meeting and rejects the other three; earliest finish then rejects the long meeting and selects the three short ones, matching the optimal set">
  <source src="/assets/img/blog/greedy_visualized/gr_activity.mp4" type="video/mp4">
</video>

Why earliest finish? Take any best schedule. Its first meeting cannot end earlier than the meeting that ends first
overall, so you can swap that one in without blocking anything that comes later. Repeat the argument for the rest
of the list. This **exchange argument** is the usual way to prove a greedy rule correct, and it is exactly what
the other two rules cannot pass.

A proof sketch is easy to get wrong, so I also ran the three rules on 3,000 random sets of up to 8 meetings and
compared each result with a check of every subset:

```text
3000 random meeting sets, times each rule chose fewer than the optimum:
{'start': 376, 'finish': 0, 'duration': 64}
```

## Knapsack: value per weight works only if you can split

The bag from the DP post again: capacity 10, items (weight 2, value 12), (4, 20) and (6, 24). Value per weight is
6, 5 and 4, so greedy takes items in that order.

- **Items can be split** (gold dust, not gold bars): take items 1 and 2 whole, then 4 of item 3's 6 units of weight.
  12 + 20 + 24 × 4/6 = **48**, and that is optimal.
- **Whole items only**: items 1 and 2 use weight 6, item 3 does not fit in the remaining 4. Greedy stops at
  **32**. The DP table finds **44** with items 2 and 3.

![Fractional versus whole-item knapsack: the fractional column takes 1, 1 and 0.67 of the three items for 48, the 0/1 greedy column takes items 1 and 2 for 32, and the DP column takes items 2 and 3 for 44](/assets/img/blog/greedy_visualized/gr_bag.png)

The same rule is optimal for one problem and wrong for its close cousin. With splitting allowed, leftover capacity
can always be filled with the next best ratio. Without it, capacity can be left unused, and a "worse" item that
fills the bag exactly can win.

```python
from fractions import Fraction
from itertools import combinations

items, capacity = [(2, 12), (4, 20), (6, 24)], 10        # (weight, value)
by_ratio = sorted(items, key=lambda it: Fraction(it[1], it[0]), reverse=True)
room, fractional = Fraction(capacity), Fraction(0)
for w, v in by_ratio:
    take = min(Fraction(1), room / w)
    fractional += take * v; room -= take * w
room, whole = capacity, 0
for w, v in by_ratio:
    if w <= room:
        whole += v; room -= w
best = max(sum(v for _, v in s) for r in range(len(items) + 1)
           for s in combinations(items, r) if sum(w for w, _ in s) <= capacity)
print("fractional greedy:", fractional, "| 0/1 greedy:", whole, "| 0/1 best:", best)
```

```text
fractional greedy: 48 | 0/1 greedy: 32 | 0/1 best: 44
```

## How to decide whether greedy is safe

- Look for a **proof or a known result** for your exact problem: earliest finish for interval scheduling, value per
  weight for the fractional knapsack.
- Otherwise, compare greedy with an exact method (DP or brute force) **on many small inputs**, the way the scripts
  above did. One counterexample settles it; many passes only make it likely.
- If greedy fails but is much faster, it can still be a good heuristic. Just call it that, not "the solution".

## Try it

```bash
git clone https://github.com/zayunsna/Algorithm-visualizer.git
cd Algorithm-visualizer
python3.12 -m venv .venv
.venv/bin/python -m pip install --only-binary=:all: -r requirements.txt
./run.sh
```

Pick **Greedy** in the lesson menu. **Run greedy** pauses after the greedy choices so you can look at them before
**Run optimal** fills in the exact answer. The coin and meeting inputs are editable, so you can hunt for your own
counterexamples.

{% include series.html title="Algorithm Lab series: one app, four kinds of algorithm" %}

Related: [Greedy decoding vs beam search: why the most likely next word can give a worse answer](/tips/2026-10-21-greedy_vs_beam_search/)
