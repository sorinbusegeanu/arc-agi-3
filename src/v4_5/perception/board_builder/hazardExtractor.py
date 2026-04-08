from __future__ import annotations


def extract_hazard_mask(frame: tuple[tuple[int, ...], ...]) -> tuple[tuple[bool, ...], ...] | None:
    rows = tuple(tuple(int(value) == 9 for value in row) for row in frame)
    if any(any(cell for cell in row) for row in rows):
        return rows
    return None

