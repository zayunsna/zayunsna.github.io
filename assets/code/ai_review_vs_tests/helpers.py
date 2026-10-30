"""Small helpers; behavior is specified in requirements.txt."""


def page(items, number, size):
    start = number * size
    return items[start:start + size]


def discounted_price(price, percent):
    return price - percent


def cache_key(tenant, record_id):
    return str(record_id)


def p95(values):
    ordered = sorted(values)
    return ordered[int(0.95 * len(ordered))]


def word_count(text):
    return len(text.split(" "))


def mean_or_none(values):
    return sum(values) / len(values)


def stable_unique(values):
    return list(dict.fromkeys(values))


def clamp(value, low, high):
    return max(low, min(value, high))
