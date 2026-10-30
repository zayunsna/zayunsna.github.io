"""Implementations matching the published behavioral contract."""


def page(items, number, size):
    start = (number - 1) * size
    return items[start:start + size]


def discounted_price(price, percent):
    return price * (1 - percent / 100)


def cache_key(tenant, record_id):
    return f"{tenant}:{record_id}"


def p95(values):
    ordered = sorted(values)
    rank = (95 * len(ordered) + 99) // 100
    return ordered[rank - 1]


def word_count(text):
    return len(text.split())


def mean_or_none(values):
    return sum(values) / len(values) if values else None


def stable_unique(values):
    return list(dict.fromkeys(values))


def clamp(value, low, high):
    return max(low, min(value, high))
