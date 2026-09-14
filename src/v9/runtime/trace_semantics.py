from __future__ import annotations


def rows(value, key=None):
    if key is not None:
        value = value.get(key, ())
    return tuple(tuple(item) for item in value)
