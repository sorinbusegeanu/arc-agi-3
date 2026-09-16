from __future__ import annotations

from collections import defaultdict, deque
from typing import Any, Iterable

from .pipeline_service_v2 import MemoryPipelineServiceV2


class _PreviousInteractionCursor:
    """Replay the exact ordered previous-interaction state inside one true batch."""

    def __init__(self, rows: dict[tuple[int, int], deque[Any | None]]) -> None:
        self._rows = rows

    def get(self, key: tuple[int, int], default: Any = None) -> Any:
        values = self._rows.get(key)
        if not values:
            return default
        return values.popleft()


def _ordered_previous_interactions(self: Any, prepared_rows: Iterable[Any]) -> _PreviousInteractionCursor:
    with self._lock:
        latest = dict(self._latest_interaction_grounding)
    rows: dict[tuple[int, int], deque[Any | None]] = defaultdict(deque)
    for prepared in prepared_rows:
        grounding = getattr(prepared, "m1g", None)
        if grounding is None:
            continue
        key = (int(grounding.environment_instance_id), int(grounding.episode_id))
        rows[key].append(latest.get(key))
        latest[key] = grounding
    return _PreviousInteractionCursor(rows)


def _block_for_result(self: MemoryPipelineServiceV2, *, timeout: float = 0.05) -> bool:
    """Bounded blocking wait used while draining ingestion/derivation at epoch boundaries."""
    progressed = self.pump_ingest_tasks()
    progressed = self.pump_derivation_tasks() or progressed

    ingest_outstanding = bool(
        self.ingested < self.sampled
        or self.pending_ingest
        or self.ingest_results
    )
    if ingest_outstanding:
        progressed = self.drain_ingest_results(block=True, timeout=float(timeout)) or progressed
        progressed = self.apply_ingest_ready() or progressed
        progressed = self.pump_ingest_tasks() or progressed
        progressed = self.pump_derivation_tasks() or progressed
        return progressed

    derivation_outstanding = bool(
        self.pending_derivation
        or self.inflight
        or self.derive_results
    )
    if derivation_outstanding:
        progressed = self.drain_derivation_results(block=True, timeout=float(timeout)) or progressed
        progressed = self.apply_derivation_ready() or progressed
        progressed = self.pump_derivation_tasks() or progressed
    return progressed


def install_integration_repairs(runtime_cls: type[Any]) -> None:
    """Install compatibility repairs required by the unified v9.7.8/v9.7.9 runtime."""
    runtime_cls._previous_interactions = _ordered_previous_interactions
    MemoryPipelineServiceV2.block_for_result = _block_for_result
