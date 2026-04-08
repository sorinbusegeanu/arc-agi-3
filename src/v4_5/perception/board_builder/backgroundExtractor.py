from __future__ import annotations


def extract_background_masks(frame: tuple[tuple[int, ...], ...]) -> tuple[tuple[tuple[bool, ...], ...], tuple[tuple[bool, ...], ...]]:
    traversable = []
    blocking = []
    for row in frame:
        traversable.append(tuple(int(value) in {0, 1, 2} for value in row))
        blocking.append(tuple(int(value) in {5, 8} for value in row))
    return (tuple(traversable), tuple(blocking))

