from helpers import (cache_key, clamp, discounted_price, mean_or_none,
                     p95, page, stable_unique, word_count)


def test_page():
    assert page([10, 20, 30, 40], 1, 2) == [10, 20]


def test_discount():
    assert discounted_price(50, 10) == 45


def test_cache_key():
    assert cache_key("alpha", 7) != cache_key("beta", 7)


def test_p95():
    assert p95([42]) == 42


def test_words():
    assert word_count("hello world") == 2


def test_mean():
    assert mean_or_none([2, 4]) == 3


def test_unique():
    assert stable_unique([3, 1, 3, 2, 1]) == [3, 1, 2]


def test_clamp():
    assert [clamp(v, 0, 10) for v in [-1, 0, 5, 10, 11]] == [0, 0, 5, 10, 10]
