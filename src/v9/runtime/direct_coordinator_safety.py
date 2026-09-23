from __future__ import annotations

import time
from typing import Any

from .canonical_commit import (
    apply_canonical_commit_batch,
    canonical_commit_prefix_length,
)
from .memory_pipeline import PreparedCommitBatch


_DIRECT_ACTOR_FACTORY = "v9.environments.direct_adapter:make_adapter"


def _split_bytes(total_bytes: int, rows: int) -> tuple[int, ...]:
    rows = int(rows)
    if rows <= 0:
        return ()
    base, remainder = divmod(max(0, int(total_bytes)), rows)
    return tuple(base + int(index < remainder) for index in range(rows))


def _batch_row_bytes(batch: Any) -> tuple[int, ...]:
    rows = tuple(getattr(batch, "rows", ()))
    explicit = tuple(int(value) for value in (getattr(batch, "row_input_bytes", ()) or ()))
    if explicit:
        if len(explicit) != len(rows):
            raise ValueError("prepared commit byte measurements must match rows")
        return explicit
    return _split_bytes(int(getattr(batch, "input_bytes", 0)), len(rows))


def _consume_ingest_prefix(service: Any, row_count: int) -> None:
    remaining = int(row_count)
    sequence = int(service.ingest_apply)
    while remaining > 0:
        batch = service.ingest_results.pop(sequence)
        rows = tuple(batch.rows)
        row_bytes = _batch_row_bytes(batch)
        if remaining < len(rows):
            rest_rows = rows[remaining:]
            rest_bytes = row_bytes[remaining:]
            new_start = int(rest_rows[0].sequence)
            service.ingest_results[new_start] = PreparedCommitBatch(
                new_start,
                int(rest_rows[-1].sequence),
                rest_rows,
                sum(rest_bytes),
                rest_bytes,
            )
            service.ingest_apply = new_start
            return
        remaining -= len(rows)
        sequence = int(batch.end_sequence) + 1
    service.ingest_apply = sequence


def _apply_ingest_ready_direct_safe(service: Any) -> bool:
    plans: list[Any] = []
    row_input_bytes: list[int] = []
    transaction_row_limit = min(
        int(service.canonical_batch_size),
        int(getattr(service.runtime.config, "canonical_transaction_max_rows", 1024)),
    )
    sequence = int(service.ingest_apply)
    while sequence in service.ingest_results:
        batch = service.ingest_results[sequence]
        batch_rows = tuple(batch.rows)
        if plans and len(plans) + len(batch_rows) > transaction_row_limit:
            break
        plans.extend(batch_rows)
        row_input_bytes.extend(_batch_row_bytes(batch))
        sequence = int(batch.end_sequence) + 1
        if len(plans) >= transaction_row_limit:
            break
    if not plans:
        return False

    prefix = canonical_commit_prefix_length(
        service.runtime,
        plans,
        row_input_bytes=tuple(row_input_bytes),
    )
    if prefix <= 0:
        raise RuntimeError("canonical budget prefixer returned no admissible row")

    started = time.perf_counter()
    selected_rows = tuple(plans[:prefix])
    selected_bytes = sum(row_input_bytes[:prefix])
    result = apply_canonical_commit_batch(
        service.runtime,
        selected_rows,
        input_bytes=selected_bytes,
    )
    elapsed = time.perf_counter() - started

    service.canonical_apply_seconds += elapsed
    service.canonical_apply_events += prefix
    service.last_canonical_ingest_batch = prefix
    service.ingested += prefix
    service.release_ingest_input_bytes(selected_bytes)
    _consume_ingest_prefix(service, prefix)

    resident_manager = getattr(service.runtime, "_resident_memory", None)
    if resident_manager is not None:
        resident_manager.maybe_compact()

    for candidate in result.derivation_candidates:
        service._consider_candidate(candidate)
    backlog = max(
        len(service.pending_ingest),
        len(service.ingest_results),
        max(0, int(service.sampled) - int(service.ingested)),
    )
    from .parallel_memory_coordinator import _adaptive_canonical_batch_size

    service.canonical_batch_size = _adaptive_canonical_batch_size(
        service.canonical_batch_size,
        backlog,
    )
    return True


def _install_plain_actor_publication() -> None:
    from . import multiprocess

    if getattr(multiprocess, "_plain_direct_publication_installed", False):
        return

    def _put_plain_stage_batch(
        stage_queue: Any,
        counters: Any,
        counter_index: int,
        envelope: Any,
        transport_pool: Any | None = None,
    ) -> None:
        # Direct coordinator mode should not depend on shared transport slabs.
        # Plain multiprocessing queues are slower but observable and cannot block
        # on slab ownership/semaphore release before counters move.
        stage_queue.put(envelope)
        if counters is not None:
            counters[int(counter_index)] = int(counters[int(counter_index)]) + len(envelope.transitions)

    multiprocess._put_counted_stage_batch = _put_plain_stage_batch
    multiprocess._plain_direct_publication_installed = True


def _install_direct_actor_factory() -> None:
    from . import multiprocess

    if getattr(multiprocess.ProcessTopology, "_direct_actor_adapter_installed", False):
        return
    original_start_actor = multiprocess.ProcessTopology.start_actor

    def start_actor(self: Any, *args: Any, **kwargs: Any) -> Any:
        if kwargs.get("adapter_factory_path") == "v9.cli:make_adapter":
            kwargs = dict(kwargs)
            kwargs["adapter_factory_path"] = _DIRECT_ACTOR_FACTORY
        return original_start_actor(self, *args, **kwargs)

    multiprocess.ProcessTopology.start_actor = start_actor
    multiprocess.ProcessTopology._direct_actor_adapter_installed = True


def install(memory_pipeline_service_cls: type) -> None:
    if not getattr(memory_pipeline_service_cls, "_direct_coordinator_safety_installed", False):
        memory_pipeline_service_cls.apply_ingest_ready = _apply_ingest_ready_direct_safe
        memory_pipeline_service_cls._direct_coordinator_safety_installed = True
    _install_plain_actor_publication()
    _install_direct_actor_factory()
