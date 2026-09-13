from __future__ import annotations

from collections import deque
from threading import Condition
from typing import Generic, TypeVar

from v9.modalities.contract import InteractionEvent, PassiveSymbolEvent, TimelineEvent

T = TypeVar("T")


class RingClosed(RuntimeError):
    pass


class BoundedRing(Generic[T]):
    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("ring capacity must be positive")
        self.capacity = int(capacity)
        self._values: deque[T] = deque()
        self._condition = Condition()
        self._closed = False

    def put(self, value: T) -> bool:
        with self._condition:
            if self._closed:
                raise RingClosed("ring is closed")
            if len(self._values) >= self.capacity:
                return False
            self._values.append(value)
            self._condition.notify()
            return True

    def get_nowait(self) -> T | None:
        with self._condition:
            return self._values.popleft() if self._values else None

    def drain(self, limit: int | None = None) -> tuple[T, ...]:
        with self._condition:
            count = len(self._values) if limit is None else min(len(self._values), int(limit))
            return tuple(self._values.popleft() for _ in range(count))

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()

    def __len__(self) -> int:
        with self._condition:
            return len(self._values)


class MultimodalTimeline:
    """Causally ordered bounded ingress; passive events never count as actions."""

    def __init__(self, *, capacity: int, symbol_budget: int, symbol_payload_bytes: int) -> None:
        self.ring: BoundedRing[TimelineEvent] = BoundedRing(capacity)
        self.symbol_budget = int(symbol_budget)
        self.symbol_payload_bytes = int(symbol_payload_bytes)
        self.last_ordering_key: tuple[int, int, int] | None = None
        self.events_seen = 0
        self.events_dropped = 0
        self.actions_committed = 0

    def append(self, event: TimelineEvent) -> bool:
        key = event.identity.ordering_key
        if self.last_ordering_key is not None and key < self.last_ordering_key:
            raise ValueError("timeline events must preserve causal producer order")
        self.events_seen += 1
        admitted = self.ring.put(event)
        if not admitted:
            self.events_dropped += 1
            return False
        self.last_ordering_key = key
        return True

    def append_symbols(self, events: tuple[PassiveSymbolEvent, ...], raw_sizes: tuple[int, ...]) -> tuple[PassiveSymbolEvent, ...]:
        admitted: list[PassiveSymbolEvent] = []
        payload = 0
        for event, size in zip(events, raw_sizes, strict=True):
            if len(admitted) >= self.symbol_budget or payload + int(size) > self.symbol_payload_bytes:
                self.events_seen += 1
                self.events_dropped += 1
                continue
            payload += int(size)
            if self.append(event):
                admitted.append(event)
        return tuple(admitted)

    def pop_next(self) -> TimelineEvent | None:
        row = self.ring.get_nowait()
        if isinstance(row, InteractionEvent):
            self.actions_committed += 1
        return row

    def state_dict(self) -> dict[str, object]:
        if len(self.ring):
            raise RuntimeError("snapshot requires a drained multimodal ingress ring")
        return {
            "capacity": self.ring.capacity, "symbol_budget": self.symbol_budget,
            "symbol_payload_bytes": self.symbol_payload_bytes,
            "last_ordering_key": list(self.last_ordering_key) if self.last_ordering_key else None,
            "events_seen": self.events_seen, "events_dropped": self.events_dropped,
            "actions_committed": self.actions_committed,
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, object]) -> "MultimodalTimeline":
        result = cls(capacity=int(state["capacity"]), symbol_budget=int(state["symbol_budget"]), symbol_payload_bytes=int(state["symbol_payload_bytes"]))
        raw_key = state.get("last_ordering_key")
        result.last_ordering_key = None if raw_key is None else tuple(int(value) for value in raw_key)
        result.events_seen = int(state.get("events_seen", 0))
        result.events_dropped = int(state.get("events_dropped", 0))
        result.actions_committed = int(state.get("actions_committed", 0))
        return result
