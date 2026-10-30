"""Execute the concrete claims from both static review reports."""
import helpers as bad
import helpers_fixed as fixed

checks = [
    ("page", ([1, 2, 3], 1, 2), [1, 2]),
    ("discounted_price", (200, 10), 180),
    ("cache_key", ("a", 1), "a:1"),
    ("cache_key", ("b", 1), "b:1"),
    ("p95", (list(range(1, 21)),), 19),
    ("word_count", ("a  b",), 2),
    ("word_count", ("a\tb",), 2),
    ("word_count", ("",), 0),
    ("word_count", ("   ",), 0),
    ("mean_or_none", ([],), None),
]
for name, args, expected in checks:
    try:
        actual = getattr(bad, name)(*args)
    except ZeroDivisionError:
        actual = "ZeroDivisionError"
    assert actual != expected, (name, args, actual)
    assert getattr(fixed, name)(*args) == expected, (name, args)
    print(f"{name}{args!r}: broken={actual!r}, fixed={expected!r}")
assert bad.stable_unique([1, True, 1.0]) == [1]
assert [bad.clamp(v, 0, 10) for v in [-1, 0, 5, 10, 11]] == [0, 0, 5, 10, 10]
print("PASS: 10 counterexamples reproduced; 2 control probes passed")
