from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Hashable, Mapping, Protocol


class CanonicalCollection(str, Enum):
    GRAPH = "graph"
    PAYLOAD = "payload"
    LEVEL_INDEX = "level_index"
    SIGNATURE_INDEX = "signature_index"
    GROUNDING_INDEX = "grounding_index"
    PROVENANCE_INDEX = "provenance_index"
    LIFECYCLE_INDEX = "lifecycle_index"
    POLICY_SOURCE_INDEX = "policy_source_index"


class CanonicalTransactionStatus(str, Enum):
    READY = "READY"
    APPLIED = "APPLIED"
    OVERSIZED_CANONICAL_TRANSACTION = "OVERSIZED_CANONICAL_TRANSACTION"
    OVERSIZED_CANONICAL_PRIMITIVE = "OVERSIZED_CANONICAL_PRIMITIVE"


class CanonicalTransactionQuarantined(RuntimeError):
    def __init__(
        self,
        status: CanonicalTransactionStatus,
        *,
        sequences: tuple[int, ...],
        estimate: "CanonicalWorkEstimate",
    ) -> None:
        if status not in {
            CanonicalTransactionStatus.OVERSIZED_CANONICAL_TRANSACTION,
            CanonicalTransactionStatus.OVERSIZED_CANONICAL_PRIMITIVE,
        }:
            raise ValueError("only oversized canonical work can be quarantined")
        self.status = status
        self.sequences = tuple(int(value) for value in sequences)
        self.estimate = estimate
        super().__init__(
            f"{status.value}: sequences={self.sequences} rows={estimate.rows} "
            f"input_bytes={estimate.input_bytes} mutation_bytes={estimate.materialized_mutation_bytes} "
            f"writes={estimate.write_count} work_units={estimate.work_units}"
        )


@dataclass(frozen=True, slots=True)
class CanonicalWorkEstimate:
    rows: int
    input_bytes: int
    materialized_mutation_bytes: int
    write_count: int
    symbol_count: int = 0
    derived_relation_count: int = 0
    grounding_operations: int = 0
    work_units: int = 0

    def __post_init__(self) -> None:
        if min(vars_without_slots(self).values()) < 0:
            raise ValueError("canonical work estimates must be non-negative")


@dataclass(frozen=True, slots=True)
class CanonicalWorkBudget:
    max_rows: int = 1024
    max_input_bytes: int = 64 * 1024 * 1024
    max_materialized_mutation_bytes: int = 64 * 1024 * 1024
    max_continuation_bytes: int = 16 * 1024 * 1024
    max_writes: int = 8192
    max_work_units: int = 1_000_000

    def __post_init__(self) -> None:
        if min(vars_without_slots(self).values()) <= 0:
            raise ValueError("canonical work budgets must be positive")

    def status(self, estimate: CanonicalWorkEstimate) -> CanonicalTransactionStatus:
        if estimate.materialized_mutation_bytes > self.max_continuation_bytes and estimate.write_count <= 1:
            return CanonicalTransactionStatus.OVERSIZED_CANONICAL_PRIMITIVE
        if (
            estimate.rows > self.max_rows
            or estimate.input_bytes > self.max_input_bytes
            or estimate.materialized_mutation_bytes > self.max_materialized_mutation_bytes
            or estimate.write_count > self.max_writes
            or estimate.work_units > self.max_work_units
        ):
            return CanonicalTransactionStatus.OVERSIZED_CANONICAL_TRANSACTION
        return CanonicalTransactionStatus.READY


def vars_without_slots(value: object) -> dict[str, int]:
    return {
        name: int(getattr(value, name))
        for name in value.__dataclass_fields__  # type: ignore[attr-defined]
    }


@dataclass(frozen=True, slots=True)
class CanonicalContinuationFragment:
    index: int
    changes: tuple[tuple[CanonicalCollection, Hashable, object], ...]
    encoded_bytes: int


@dataclass(frozen=True, slots=True)
class CanonicalFragmentTiming:
    fragment_index: int
    elapsed_ms: float
    primitive_over_100ms: bool


def _estimate_bytes(key: object, value: object | None) -> int:
    def fallback(item: object) -> object:
        if hasattr(item, "hex") and callable(getattr(item, "hex")):
            return {"__type__": type(item).__name__, "hex": item.hex()}
        if hasattr(item, "__dict__"):
            return {"__type__": type(item).__name__, **vars(item)}
        return {"__type__": type(item).__name__, "repr": repr(item)}

    return len(
        json.dumps([key, value], sort_keys=True, separators=(",", ":"), default=fallback).encode("utf-8")
    )


class CanonicalStoreReader(Protocol):
    def read_from_handle(self, handle: object, collection: CanonicalCollection | str, key: Hashable, default: Any = None) -> Any: ...


_TOMBSTONE = object()


@dataclass(slots=True)
class TransactionOverlay:
    """Bounded private delta over one immutable canonical handle."""

    base_handle: object
    target_lsn: int
    max_entries: int = 8192
    max_bytes: int = 64 * 1024 * 1024
    _changes: dict[CanonicalCollection, dict[Hashable, object]] = field(default_factory=dict)
    _entry_count: int = 0
    _estimated_bytes: int = 0
    _closed: bool = False

    def __post_init__(self) -> None:
        if self.target_lsn < 0:
            raise ValueError("target_lsn must be non-negative")
        if min(self.max_entries, self.max_bytes) <= 0:
            raise ValueError("overlay bounds must be positive")

    @property
    def entry_count(self) -> int:
        return self._entry_count

    @property
    def estimated_bytes(self) -> int:
        return self._estimated_bytes

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def changes(self) -> Mapping[CanonicalCollection, Mapping[Hashable, object]]:
        return MappingProxyType(
            {collection: MappingProxyType(dict(rows)) for collection, rows in self._changes.items()}
        )

    def _write(self, collection: CanonicalCollection | str, key: Hashable, value: object) -> None:
        if self._closed:
            raise RuntimeError("transaction overlay is closed")
        selected = collection if isinstance(collection, CanonicalCollection) else CanonicalCollection(str(collection))
        rows = self._changes.setdefault(selected, {})
        previous = rows.get(key, None)
        previous_bytes = 0 if key not in rows else _estimate_bytes(key, None if previous is _TOMBSTONE else previous)
        next_bytes = _estimate_bytes(key, None if value is _TOMBSTONE else value)
        next_entries = self.entry_count + int(key not in rows)
        total_bytes = self._estimated_bytes - previous_bytes + next_bytes
        if next_entries > self.max_entries:
            raise OverflowError("canonical transaction overlay entry ceiling exceeded")
        if total_bytes > self.max_bytes:
            raise OverflowError("canonical transaction overlay byte ceiling exceeded")
        if key not in rows:
            self._entry_count += 1
        rows[key] = value
        self._estimated_bytes = total_bytes

    def put(self, collection: CanonicalCollection | str, key: Hashable, value: object) -> None:
        self._write(collection, key, value)

    def delete(self, collection: CanonicalCollection | str, key: Hashable) -> None:
        self._write(collection, key, _TOMBSTONE)

    def get(
        self,
        store: CanonicalStoreReader,
        collection: CanonicalCollection | str,
        key: Hashable,
        default: Any = None,
    ) -> Any:
        selected = collection if isinstance(collection, CanonicalCollection) else CanonicalCollection(str(collection))
        rows = self._changes.get(selected, {})
        if key in rows:
            value = rows[key]
            return default if value is _TOMBSTONE else value
        return store.read_from_handle(self.base_handle, selected, key, default)

    def iter_changes(self) -> tuple[tuple[CanonicalCollection, Hashable, object], ...]:
        return tuple(
            (collection, key, value)
            for collection in CanonicalCollection
            for key, value in sorted(self._changes.get(collection, {}).items(), key=lambda row: repr(row[0]))
        )

    def close(self) -> None:
        self._closed = True

    def abort(self) -> None:
        self._changes.clear()
        self._entry_count = 0
        self._estimated_bytes = 0
        self._closed = True


@dataclass(frozen=True, slots=True)
class CanonicalTransaction:
    transaction_id: str
    target_lsn: int
    overlay: TransactionOverlay
    estimate: CanonicalWorkEstimate | None = None
    status: CanonicalTransactionStatus = CanonicalTransactionStatus.READY

    def __post_init__(self) -> None:
        if not self.transaction_id:
            raise ValueError("transaction_id is required")
        if self.target_lsn != self.overlay.target_lsn:
            raise ValueError("transaction and overlay LSNs must match")


def estimate_overlay_work(overlay: TransactionOverlay, *, rows: int, input_bytes: int, work_units: int | None = None) -> CanonicalWorkEstimate:
    return CanonicalWorkEstimate(
        rows=int(rows),
        input_bytes=int(input_bytes),
        materialized_mutation_bytes=overlay.estimated_bytes,
        write_count=overlay.entry_count,
        work_units=overlay.entry_count if work_units is None else int(work_units),
    )


def compile_continuation_fragments(
    overlay: TransactionOverlay, *, max_fragment_bytes: int = 16 * 1024 * 1024
) -> tuple[CanonicalContinuationFragment, ...]:
    if max_fragment_bytes <= 0:
        raise ValueError("max_fragment_bytes must be positive")
    fragments: list[CanonicalContinuationFragment] = []
    pending: list[tuple[CanonicalCollection, Hashable, object]] = []
    pending_bytes = 0
    for change in overlay.iter_changes():
        size = _estimate_bytes(change[1], None if is_tombstone(change[2]) else change[2])
        if size > max_fragment_bytes:
            raise OverflowError(CanonicalTransactionStatus.OVERSIZED_CANONICAL_PRIMITIVE.value)
        if pending and pending_bytes + size > max_fragment_bytes:
            fragments.append(CanonicalContinuationFragment(len(fragments), tuple(pending), pending_bytes))
            pending = []
            pending_bytes = 0
        pending.append(change)
        pending_bytes += size
    if pending:
        fragments.append(CanonicalContinuationFragment(len(fragments), tuple(pending), pending_bytes))
    return tuple(fragments)


def execute_private_continuations(
    overlay: TransactionOverlay,
    *,
    max_fragment_bytes: int = 16 * 1024 * 1024,
    primitive: Any | None = None,
) -> tuple[CanonicalFragmentTiming, ...]:
    """Execute bounded private work without publishing the overlay."""
    callback = primitive or (lambda _fragment: None)
    timings = []
    for fragment in compile_continuation_fragments(overlay, max_fragment_bytes=max_fragment_bytes):
        started = time.perf_counter()
        callback(fragment)
        elapsed_ms = 1000.0 * (time.perf_counter() - started)
        timings.append(CanonicalFragmentTiming(fragment.index, elapsed_ms, elapsed_ms > 100.0))
    return tuple(timings)


def is_tombstone(value: object) -> bool:
    return value is _TOMBSTONE


__all__ = [
    "CanonicalCollection",
    "CanonicalContinuationFragment",
    "CanonicalFragmentTiming",
    "CanonicalTransaction",
    "CanonicalTransactionQuarantined",
    "CanonicalTransactionStatus",
    "CanonicalWorkBudget",
    "CanonicalWorkEstimate",
    "TransactionOverlay",
    "compile_continuation_fragments",
    "estimate_overlay_work",
    "execute_private_continuations",
    "is_tombstone",
]
