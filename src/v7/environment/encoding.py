from __future__ import annotations

from collections import Counter, defaultdict
from hashlib import blake2b

import numpy as np

_SIGNATURE_MASK = (1 << 63) - 1


def grid_signature(grid: np.ndarray) -> int:
    array = np.ascontiguousarray(grid, dtype=np.int64)
    digest = blake2b(digest_size=8)
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return int.from_bytes(digest.digest(), "little") & _SIGNATURE_MASK


def transition_signature(before: np.ndarray, after: np.ndarray) -> int:
    left = np.ascontiguousarray(before, dtype=np.int64)
    right = np.ascontiguousarray(after, dtype=np.int64)
    digest = blake2b(digest_size=8)
    digest.update(np.asarray(left.shape, dtype=np.int64).tobytes())
    digest.update(np.asarray(right.shape, dtype=np.int64).tobytes())
    if left.shape == right.shape:
        changed = np.flatnonzero(left.ravel() != right.ravel()).astype(np.int64, copy=False)
        digest.update(changed.tobytes())
        if changed.size:
            digest.update(left.ravel()[changed].tobytes())
            digest.update(right.ravel()[changed].tobytes())
    else:
        digest.update(left.tobytes())
        digest.update(right.tobytes())
    return int.from_bytes(digest.digest(), "little") & _SIGNATURE_MASK


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
