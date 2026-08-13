from __future__ import annotations

from collections import Counter, defaultdict
from hashlib import blake2b

import numpy as np

_SIGNATURE_MASK = (1 << 63) - 1


def grid_signature(grid: np.ndarray) -> int:
    """Color-label-invariant context signature; no object semantics are introduced."""
    array = np.ascontiguousarray(grid, dtype=np.int64)
    canonical = _canonicalize_values(array)
    digest = blake2b(digest_size=8)
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(canonical.tobytes())
    return int.from_bytes(digest.digest(), "little") & _SIGNATURE_MASK


def transition_signature(before: np.ndarray, after: np.ndarray) -> int:
    """Encode observable change structurally rather than by absolute color identity."""
    left = np.ascontiguousarray(before, dtype=np.int64)
    right = np.ascontiguousarray(after, dtype=np.int64)
    digest = blake2b(digest_size=8)
    digest.update(np.asarray(left.shape, dtype=np.int64).tobytes())
    digest.update(np.asarray(right.shape, dtype=np.int64).tobytes())
    if left.shape != right.shape:
        digest.update(_canonicalize_values(left).tobytes())
        digest.update(_canonicalize_values(right).tobytes())
        return int.from_bytes(digest.digest(), "little") & _SIGNATURE_MASK

    changed_mask = left != right
    coordinates = np.argwhere(changed_mask)
    digest.update(np.asarray([len(coordinates)], dtype=np.int64).tobytes())
    if not len(coordinates):
        return int.from_bytes(digest.digest(), "little") & _SIGNATURE_MASK

    minimum = coordinates.min(axis=0)
    maximum = coordinates.max(axis=0)
    relative = (coordinates - minimum).astype(np.int64, copy=False)
    bbox = (maximum - minimum + 1).astype(np.int64, copy=False)
    digest.update(bbox.tobytes())
    digest.update(relative.tobytes())

    pairs = [(int(left[tuple(position)]), int(right[tuple(position)])) for position in coordinates]
    pair_codes: dict[tuple[int, int], int] = {}
    codes: list[int] = []
    for pair in pairs:
        if pair not in pair_codes:
            pair_codes[pair] = len(pair_codes)
        codes.append(pair_codes[pair])
    digest.update(np.asarray(codes, dtype=np.int64).tobytes())
    digest.update(np.asarray([len(pair_codes)], dtype=np.int64).tobytes())
    return int.from_bytes(digest.digest(), "little") & _SIGNATURE_MASK


def _canonicalize_values(array: np.ndarray) -> np.ndarray:
    mapping: dict[int, int] = {}
    result = np.empty(array.size, dtype=np.int64)
    for index, value in enumerate(array.ravel()):
        key = int(value)
        if key not in mapping:
            mapping[key] = len(mapping)
        result[index] = mapping[key]
    return result.reshape(array.shape)


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
