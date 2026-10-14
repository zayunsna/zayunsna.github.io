---
layout: post
title: "Bit tricks in Python, one bit at a time: why ~5 is -6 and how to get 8-bit results"
description: >
  Python integers have no fixed width, so ~5 is -6 and 152 << 2 is 608. An 8-bit visualizer shows what the same
  operations do inside a fixed word: two's complement, logical and arithmetic shifts, x & (x - 1), x & -x and the
  XOR trick, each checked against plain Python.
image: /assets/img/blog/bits_visualized/cover.png
series: algorithm-lab
series_label: "Bit operations: two's complement, shifts and classic tricks"
sitemap:
  changefreq: daily
  priority: 1.0
---

# Bit tricks in Python, one bit at a time: why ~5 is -6 and how to get 8-bit results

Try this in Python:

```python
>>> ~5
-6
>>> 152 << 2
608
```

In C, with an 8-bit unsigned value, you would get 250 and 96. Neither answer is wrong. Python integers have no
fixed width: the documentation says bitwise results are computed "as though carried out in two's complement with an
infinite number of sign bits"
([Python docs, bitwise operations](https://docs.python.org/3/library/stdtypes.html#bitwise-operations-on-integer-types),
checked 2026-10-14). The C-style answers come from cutting the result to 8 bits.

This post uses the bit lessons of the same pygame app as the rest of the series. The app draws each bit of an
8-bit word, so you can see where that cut happens. Every result below was also checked in plain Python.

_Tested on 2026-10-14 with Python 3.12.13 and pygame 2.6.1 on macOS. App source:
[github.com/zayunsna/Algorithm-visualizer](https://github.com/zayunsna/Algorithm-visualizer)._

## The one-line fix: mask to the width you mean

To get an 8-bit result, keep only the lowest 8 bits with `& 0xFF` (in general, `& ((1 << W) - 1)` for `W` bits).
The app does the same with every result. To read a word as a signed number, check its top bit:

```python
W = 8
MASK = (1 << W) - 1          # 0xFF

def signed(word):            # read an 8-bit word as two's complement
    return word - (1 << W) if word & (1 << (W - 1)) else word

print("~5 =", ~5, "| -5 & 0xFF =", -5 & MASK, "| (~5 + 1) & 0xFF =", (~5 + 1) & MASK)
print("152 as signed 8-bit:", signed(152))
```

```text
~5 = -6 | -5 & 0xFF = 251 | (~5 + 1) & 0xFF = 251
152 as signed 8-bit: -104
```

AND, OR and XOR of two non-negative 8-bit values never leave the 8 bits, so they need no mask. With the app's
default inputs, 152 & 60 = 24, 152 | 60 = 188 and 152 ^ 60 = 164, and the app fills the result one column at a
time from the truth table:

![AND of 152 and 60 in 8 bits: two input rows, a result row, and the AND truth table with the current row highlighted; result 24](/assets/img/blog/bits_visualized/bt_and.png)

NOT is different: `~152` is -153 in Python and 103 after masking, because NOT also flips the endless leading
zeros of a positive Python integer.

## Two's complement: invert and add one

How does an 8-bit word store -5? Invert all bits of 5, then add 1. The carry runs from right to left:

<video controls muted loop playsinline preload="metadata" width="880" poster="/assets/img/blog/bits_visualized/bt_negate_poster.png" aria-label="The bits of 5 are inverted one by one to 11111010, then one is added from the right with a carry, giving 11111011, which is 251 unsigned or minus 5 signed">
  <source src="/assets/img/blog/bits_visualized/bt_negate.mp4" type="video/mp4">
</video>

`00000101` becomes `11111010` (250), plus one is `11111011`: 251 as an unsigned byte, -5 as a signed one.
That is the same 251 Python gave for `-5 & 0xFF`.

The edge case is the most negative value. In 8 bits, -128 is `10000000`. Invert it and add one, and you get
`10000000` again: +128 does not fit in a signed byte, so negating -128 gives -128.

![Negating minus 128 in 8 bits: X is 10000000, the inverted row is 01111111, and adding one returns 10000000, still minus 128](/assets/img/blog/bits_visualized/bt_negate_min.png)

Python never hits this, because its integers just grow. A fixed-width system (C, NumPy `int8`, a hardware
register) does. If you port bit code from C to Python, this is where results start to differ.

## Right shift: fill with zeros or with the sign?

Shifting 152 (`10011000`) right by 2 can mean two things:

- **logical** shift: fill the new bits with 0 → `00100110` = 38
- **arithmetic** shift: copy the sign bit → `11100110` = 230 unsigned, -26 signed

![Shifting 152 right by 2: the two discarded bits are purple, the logical row is filled with zeros to give 38, and the arithmetic row is filled with ones to give 230, which is minus 26 signed](/assets/img/blog/bits_visualized/bt_shift_arith.png)

Python's `>>` is always arithmetic, but the sign it copies is the sign of the Python integer. 152 is positive in
Python, so `152 >> 2` is 38. To get the arithmetic result for the 8-bit word, convert to signed first:

```python
print("logical    152 >> 2:", 152 >> 2)
print("arithmetic 152 >> 2:", (signed(152) >> 2) & MASK, "-> signed", signed(152) >> 2)
print("left       152 << 2:", 152 << 2, "| masked:", (152 << 2) & MASK)
```

```text
logical    152 >> 2: 38
arithmetic 152 >> 2: 230 -> signed -26
left       152 << 2: 608 | masked: 96
```

Left shift has the opposite issue: the bits that fall off the top of an 8-bit word stay in a Python integer. Mask
after the shift.

## Four classic tricks, and why they work

**`x & (x - 1)` clears the lowest 1 bit.** Subtracting 1 flips the lowest 1 bit and every 0 to its right. AND with
the original keeps everything above and clears that part. 152 = `10011000` becomes `10010000` = 144. Repeat until
zero and you have counted the 1 bits (Kernighan's method): three rounds for 152. Since Python 3.10,
`int.bit_count()` gives that count directly. It counts the bits of the absolute value, so mask a negative number
first if you want its two's complement count.

![Counting the ones in 152: the last round of x AND x minus 1 clears the final set bit, after 3 iterations in total](/assets/img/blog/bits_visualized/bt_popcount.png)

**Power of two: `x != 0 and x & (x - 1) == 0`.** A power of two has exactly one 1 bit, so clearing it leaves zero.
The `x != 0` guard is not decoration: 0 & (0 − 1) is also 0, so without the guard 0 counts as a power of two.

![The power-of-two lesson with x equal to 0: x AND x minus 1 is 0, but the nonzero guard is false, so the answer is false](/assets/img/blog/bits_visualized/bt_power0.png)

**`x & -x` keeps only the lowest 1 bit.** `-x` is "invert, add one", which leaves the lowest 1 bit in place and
inverts everything above it. AND keeps just that bit: 152 & -152 = 8, and for any odd number the result is 1.
This works in Python without a mask.

**XOR finds the value without a partner.** In `[5, 12, 5, 7, 12]` every number appears twice except 7. Since
`a ^ a = 0` and `a ^ 0 = a`, XOR-ing the whole list cancels the pairs, in any order:

![XOR accumulator over 5, 12, 5, 7, 12: after the last item the accumulator holds 7](/assets/img/blog/bits_visualized/bt_xor_single.png)

The same four checks in plain Python:

```python
n, count = 152, 0
while n:
    n &= n - 1; count += 1
print("ones in 152:", count, "| int.bit_count:", (152).bit_count())
print("lowest set bit, x & -x:", 152 & -152, 37 & -37, 0 & -0)

def is_power_of_two(n):
    return n != 0 and n & (n - 1) == 0
print("power of two:", {n: is_power_of_two(n) for n in (12, 0, 16)}, "| without the guard, 0 gives", 0 & (0 - 1) == 0)

acc = 0
for v in [5, 12, 5, 7, 12]:
    acc ^= v
print("XOR of 5, 12, 5, 7, 12:", acc)
```

```text
ones in 152: 3 | int.bit_count: 3
lowest set bit, x & -x: 8 1 0
power of two: {12: False, 0: False, 16: True} | without the guard, 0 gives True
XOR of 5, 12, 5, 7, 12: 7
```

## Bitmasks as subsets

One more use connects back to the [knapsack in the DP post](/blog/2026-10-20-dp_visualized/). With 3 items, the
numbers 0 to 7 in binary (`000` to `111`) list every possible choice: bit `k` set means "take item `k + 1`". Looping
over all masks is a brute-force knapsack:

```python
items, capacity = [(2, 12), (4, 20), (6, 24)], 10
best = (0, 0)
for mask in range(1 << len(items)):
    picked = [i for i in range(len(items)) if mask >> i & 1]
    weight = sum(items[i][0] for i in picked); value = sum(items[i][1] for i in picked)
    if weight <= capacity and value > best[0]:
        best = (value, mask)
print("best subset value:", best[0], "mask:", f"{best[1]:03b}")
```

```text
best subset value: 44 mask: 110
```

Mask `110` is items 2 and 3, value 44: the same answer as the DP table, and the one that value-per-weight greedy
missed in the [greedy post](/blog/2026-10-21-greedy_visualized/). Brute force is fine for 3 items, but the number
of masks doubles with every item, which is why DP exists.

![The subset lesson after all 8 masks: the best feasible value found is 44 with items 2 and 3](/assets/img/blog/bits_visualized/bt_subsets.png)

## Summary

| You want | Python |
|---|---|
| A `W`-bit result | `value & ((1 << W) - 1)` |
| The signed value of a `W`-bit word | `word - (1 << W) if word >> (W - 1) else word` |
| Arithmetic right shift of a `W`-bit word | `(signed(word) >> n) & MASK` |
| Count 1 bits | `x.bit_count()` (3.10+), on the masked value for negatives |
| Clear the lowest 1 bit | `x & (x - 1)` |
| Keep only the lowest 1 bit | `x & -x` |
| Power of two | `x != 0 and x & (x - 1) == 0` |

## Try it

```bash
git clone https://github.com/zayunsna/Algorithm-visualizer.git
cd Algorithm-visualizer
python3.12 -m venv .venv
.venv/bin/python -m pip install --only-binary=:all: -r requirements.txt
./run.sh
```

Pick **Bits** in the lesson menu. You can switch between 4, 8 and 16 bits, toggle signed and unsigned views, and
type your own values. Where Python's own operator gives a different number, the panel shows that raw result next
to the masked one.

{% include series.html title="Algorithm Lab series: one app, four kinds of algorithm" %}

Related: [fp32, fp16, bf16 and int8: what each format stores, and where it breaks](/tips/2026-10-22-float16_bfloat16_int8/)
