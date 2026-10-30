from helpers import mean_or_none, p95, word_count


def test_p95_twenty_values():
    assert p95(list(range(1, 21))) == 19


def test_whitespace():
    assert word_count(" \thello   world\n ") == 2
    assert word_count("") == 0
    assert word_count(" \t\n ") == 0


def test_empty_mean():
    assert mean_or_none([]) is None
