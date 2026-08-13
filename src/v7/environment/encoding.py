from __future__ import annotations

from collections import Counter, defaultdict
from hashlib import blake2b

import numpy as np

_SIGNATURE_MASK = (1 << 63) - 1


def _digest_parts(*parts: bytes) -> int:
    digest = blake2b(digest_size=8)
    for part in parts:
        digest.update(part)
    return int.from_bytes(digest.digest(), "little") & _SIGNATURE_MASK


def grid_signature(grid: np.ndarray) -> int:
    array = np.ascontiguousarray(grid, dtype=np.int64)
    return _digest_parts(np.asarray(array.shape, dtype=np.int64).tobytes(), array.tobytes())


def _normalized_change(before: np.ndarray, after: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int]] | None:
    left = np.ascontiguousarray(before, dtype=np.int64)
    right = np.ascontiguousarray(after, dtype=np.int64)
    if left.shape != right.shape or left.ndim != 2:
        return None
    changed = np.argwhere(left != right)
    if changed.size == 0:
        return np.empty((0, 2), dtype=np.int64), np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), (0, 0)
    mins = changed.min(axis=0)
    maxs = changed.max(axis=0)
    normalized = (changed - mins).astype(np.int64, copy=False)
    flat = np.ravel_multi_index((changed[:, 0], changed[:, 1]), left.shape)
    before_values = left.ravel()[flat]
    after_values = right.ravel()[flat]
    box = (int(maxs[0] - mins[0] + 1), int(maxs[1] - mins[1] + 1))
    order = np.lexsort((normalized[:, 1], normalized[:, 0]))
    return normalized[order], before_values[order], after_values[order], box


def transition_signature(before: np.ndarray, after: np.ndarray) -> int:
    normalized = _normalized_change(before, after)
    if normalized is None:
        left = np.ascontiguousarray(before, dtype=np.int64)
        right = np.ascontiguousarray(after, dtype=np.int64)
        return _digest_parts(np.asarray(left.shape, dtype=np.int64).tobytes(), np.asarray(right.shape, dtype=np.int64).tobytes(), left.tobytes(), right.tobytes())
    coords, before_values, after_values, box = normalized
    return _digest_parts(np.asarray(box, dtype=np.int64).tobytes(), coords.tobytes(), before_values.tobytes(), after_values.tobytes())


def carrier_signature(before: np.ndarray, after: np.ndarray) -> int | None:
    normalized = _normalized_change(before, after)
    if normalized is None:
        return None
    coords, before_values, _after_values, box = normalized
    if coords.size == 0:
        return None
    return _digest_parts(b"carrier-v1", np.asarray(box, dtype=np.int64).tobytes(), coords.tobytes(), before_values.tobytes())


class SupportedPredictionTracker:
    def __init__(self, *, minimum_support: int = 2) -> None:
        if minimum_support < 1:
            raise ValueError("minimum_support must be positive")
        self.minimum_support = int(minimum_support)
        self._counts: dict[tuple[int, int], Counter[int]] = defaultdict(Counter)

    def prediction_error(self, context_signature: int, action_id: int, outcome_signature: int) -> float:
        counts = self._counts[(int(context_signature), int(action_id))]
        if sum(counts.values()) < self.minimum_support:
            return 0.0
        expected, _ = min(counts.items(), key=lambda item: (-item[1], item[0]))
        return 0.0 if int(outcome_signature) == expected else 1.0

    def observe(self, context_signature: int, action_id: int, outcome_signature: int) -> None:
        self._counts[(int(context_signature), int(action_id))][int(outcome_signature)] += 1
