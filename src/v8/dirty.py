from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Generic, Hashable, TypeVar


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


T = TypeVar("T")


@dataclass(slots=True)
class DirtyItem(Generic[T]):
    payload: T
    version: int
    multiplicity: int


class DirtyAccumulator(Generic[T]):
    """Persistent per-worker canonical invalidation accumulator.

    Repeated changes to one canonical key replace the payload with the newest value while
    preserving the exact accumulated multiplicity. Draining marks the key processed, so
    later evidence can invalidate it again without replaying the earlier raw events.
    """

    def __init__(self) -> None:
        self.tracker = DirtyKeyTracker()
        self._pending: dict[Hashable, DirtyItem[T]] = {}

    def add(
        self,
        key: Hashable,
        payload: T,
        *,
        version: int,
        multiplicity: int = 1,
    ) -> None:
        multiplicity = max(1, int(multiplicity))
        self.tracker.invalidate(key, int(version))
        prior = self._pending.get(key)
        if prior is None:
            self._pending[key] = DirtyItem(payload, int(version), multiplicity)
        else:
            prior.payload = payload
            prior.version = max(prior.version, int(version))
            prior.multiplicity += multiplicity

    def drain(self) -> tuple[tuple[Hashable, DirtyItem[T]], ...]:
        rows = tuple(self._pending.items())
        self._pending.clear()
        for key, item in rows:
            self.tracker.complete(key, item.version)
        return rows

    @property
    def pending_count(self) -> int:
        return len(self._pending)
