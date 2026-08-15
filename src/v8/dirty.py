from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Hashable


@dataclass(slots=True)
class DirtyState:
    last_processed_version: int = 0
    latest_required_version: int = 0
    queued: bool = False


class DirtyKeyTracker:
    """Coalesce repeated invalidations into at most one queued key."""

    def __init__(self) -> None:
        self._states: dict[Hashable, DirtyState] = {}
        self._lock = Lock()

    def invalidate(self, key: Hashable, version: int) -> bool:
        version = int(version)
        with self._lock:
            state = self._states.setdefault(key, DirtyState())
            state.latest_required_version = max(state.latest_required_version, version)
            if state.queued:
                return False
            state.queued = True
            return True

    def begin(self, key: Hashable) -> int:
        with self._lock:
            return int(self._states.setdefault(key, DirtyState()).latest_required_version)

    def complete(self, key: Hashable, processed_version: int) -> bool:
        """Return True when one requeue is required because newer evidence arrived."""
        with self._lock:
            state = self._states.setdefault(key, DirtyState())
            state.last_processed_version = max(state.last_processed_version, int(processed_version))
            if state.latest_required_version > state.last_processed_version:
                state.queued = True
                return True
            state.queued = False
            return False

    def state(self, key: Hashable) -> DirtyState:
        with self._lock:
            value = self._states.get(key, DirtyState())
            return DirtyState(
                value.last_processed_version,
                value.latest_required_version,
                value.queued,
            )
