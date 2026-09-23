from __future__ import annotations

from dataclasses import replace
import time
from typing import Any, Iterable

from v9.memory.model import CanonicalNode, MemoryLevel, MemoryType

from . import canonical_commit as _canonical_commit
from . import canonical_commit_derivation as _canonical_derivation
from . import publication_throughput as _publication_throughput
from . import residency as _residency


_MISSING = object()
_EPOCH_COMPACTION_MAX_BATCHES = 32
_EPOCH_COMPACTION_MAX_SECONDS = 5.0
_IDLE_COMPACTION_BACKLOG_MULTIPLIER = 1


def _set_gauge(runtime: Any, key: str, value: Any) -> None:
    setter = getattr(runtime, "set_telemetry_gauge", None)
    if callable(setter):
        setter(key, value)


def _aggregate_family_support(runtime: Any, rows: Iterable[Any]) -> int:
    """Count all observations behind retained representatives once per M1N signature."""
    retained = tuple(rows)
    signature_support = getattr(runtime, "signature_support", None)
    if not callable(signature_support):
        # Minimal/test runtimes do not maintain aggregate support. Preserve the
        # historical candidate semantics rather than requiring telemetry state.
        return len(retained)
    signatures = {
        int(getattr(row, "structural_signature", 0))
        for row in retained
        if int(getattr(row, "structural_signature", 0)) != 0
    }
    return sum(
        max(1, int(signature_support(signature)))
        for signature in signatures
    )


def _append_dirty_normalized_coalesced(
    runtime: Any,
    relation: Any,
    rows: list[Any],
) -> int | None:
    """Publish at most one support update per normalized signature per canonical batch.

    The first row is kept mutable until the enclosing canonical transaction publishes
    its deferred groups. Later observations update that row in place, preserving the
    final support/provenance while avoiding hundreds of redundant writes of the same
    normalized M1 node inside one transaction.
    """
    if not hasattr(runtime, "canonical_store"):
        return None
    signature = int(relation.structural_signature)
    if signature not in runtime._m1n_dirty:
        return None

    cache = runtime.__dict__.get("_canonical_dirty_coalesce")
    if cache is None:
        return _ORIGINAL_APPEND_DIRTY(runtime, relation, rows)

    occurrences = runtime._m1n_occurrences.get(signature, ())
    parents = tuple(
        uid for occurrence in occurrences for uid in occurrence.provenance.parents
    )
    evidence = [
        uid for occurrence in occurrences for uid in occurrence.provenance.evidence
    ]
    support = int(runtime.signature_support(signature, len(occurrences)))

    existing = cache.get(signature)
    if existing is None:
        payload = {
            "observable_relation": relation.observable_relation,
            "channel": relation.channel.value,
            "structural_signature": signature,
            "support": support,
            "parents": [[uid.hi, uid.lo] for uid in parents],
        }
        # A list is intentional: deferred publication consumes the row only after
        # the full canonical batch has been reduced, so later duplicate updates can
        # replace the node watermark and evidence without allocating another write.
        row = [
            CanonicalNode(
                relation.uid,
                MemoryLevel.M1,
                MemoryType.NORMALIZED_RELATION,
                (signature,),
                int(runtime._watermark),
            ),
            payload,
            evidence,
        ]
        rows.append(row)
        cache[signature] = row
        runtime.telemetry["dirty_m1n_support_updates_published"] = int(
            runtime.telemetry.get("dirty_m1n_support_updates_published", 0)
        ) + 1
        return signature

    existing[0] = CanonicalNode(
        relation.uid,
        MemoryLevel.M1,
        MemoryType.NORMALIZED_RELATION,
        (signature,),
        int(runtime._watermark),
    )
    payload = existing[1]
    payload["observable_relation"] = relation.observable_relation
    payload["channel"] = relation.channel.value
    payload["support"] = support
    payload["parents"] = [[uid.hi, uid.lo] for uid in parents]
    existing[2][:] = evidence
    runtime.telemetry["dirty_m1n_support_updates_coalesced"] = int(
        runtime.telemetry.get("dirty_m1n_support_updates_coalesced", 0)
    ) + 1
    return signature


def _apply_canonical_commit_coalesced(
    runtime: Any,
    rows: Iterable[Any],
    *,
    input_bytes: int = 0,
):
    previous = runtime.__dict__.get("_canonical_dirty_coalesce", _MISSING)
    runtime.__dict__["_canonical_dirty_coalesce"] = {}
    try:
        return _ORIGINAL_APPLY_CANONICAL(
            runtime,
            rows,
            input_bytes=int(input_bytes),
        )
    finally:
        if previous is _MISSING:
            runtime.__dict__.pop("_canonical_dirty_coalesce", None)
        else:
            runtime.__dict__["_canonical_dirty_coalesce"] = previous


def _derivation_candidates_with_aggregate_support(
    runtime: Any,
    signatures: set[int],
):
    candidates = _ORIGINAL_DERIVATION_CANDIDATES(runtime, signatures)
    upgraded = []
    max_support = 0
    for candidate in candidates:
        aggregate_support = max(
            int(candidate.support),
            _aggregate_family_support(runtime, candidate.rows),
        )
        max_support = max(max_support, aggregate_support)
        upgraded.append(
            candidate
            if aggregate_support == int(candidate.support)
            else replace(candidate, support=aggregate_support)
        )
    if max_support:
        _set_gauge(runtime, "m1_touched_family_max_aggregate_support", max_support)
    return tuple(upgraded)


def _publish_family_diagnostics(runtime: Any) -> None:
    if not callable(getattr(runtime, "set_telemetry_gauge", None)):
        return
    family_index = getattr(runtime, "_m1n_family_occurrences", {})
    supports: list[int] = []
    recurrent = 0
    independently_evidenced = 0
    for rows in family_index.values():
        retained = tuple(rows)
        if not retained:
            continue
        support = _aggregate_family_support(runtime, retained)
        supports.append(support)
        if support >= 2:
            recurrent += 1
        evidence = {
            uid
            for row in retained
            for uid in getattr(row.provenance, "evidence", ())
        }
        if support >= 2 and len(evidence) >= 2:
            independently_evidenced += 1

    m2_count = len(getattr(runtime, "_m2", {}))
    _set_gauge(runtime, "m1_family_count", len(supports))
    _set_gauge(runtime, "m1_recurrent_family_count", recurrent)
    _set_gauge(runtime, "m1_independently_evidenced_family_count", independently_evidenced)
    _set_gauge(runtime, "m1_family_max_aggregate_support", max(supports, default=0))
    _set_gauge(runtime, "m2_family_coverage", float(m2_count) / max(1, independently_evidenced))


def _request_residency_work(service: Any, *, force: bool = False) -> Any | None:
    manager = getattr(service.runtime, "_resident_memory", None)
    if manager is None:
        return None
    manager.request_compaction(force=force)
    return manager


def _service_prepared_compaction_if_idle(
    service: Any,
    *,
    max_batches: int = 1,
) -> int:
    manager = _request_residency_work(service)
    if manager is None:
        return 0
    canonical_backlog = max(0, int(service.sampled) - int(service.ingested))
    idle_threshold = max(
        256,
        int(service.canonical_batch_size) * _IDLE_COMPACTION_BACKLOG_MULTIPLIER,
    )
    if canonical_backlog > idle_threshold:
        return 0
    if bool(getattr(service, "_reducer_inflight", {})):
        return 0
    deleted = int(manager.service_prepared_compaction(max_batches=max_batches))
    if deleted:
        _set_gauge(service.runtime, "idle_compaction_last_deleted", deleted)
    return deleted


def _catch_up_residency_at_epoch_boundary(service: Any) -> int:
    manager = _request_residency_work(service, force=True)
    if manager is None:
        return 0
    started = time.perf_counter()
    deleted_total = 0
    batches = 0
    starting_backlog = int(manager.backlog())
    while (
        manager.backlog() > 0
        and batches < _EPOCH_COMPACTION_MAX_BATCHES
        and time.perf_counter() - started < _EPOCH_COMPACTION_MAX_SECONDS
    ):
        deleted = int(manager.service_prepared_compaction(max_batches=1))
        if deleted <= 0:
            deleted = int(manager.compact_once(force=True))
        if deleted <= 0:
            break
        deleted_total += deleted
        batches += 1
    elapsed = time.perf_counter() - started
    _set_gauge(service.runtime, "epoch_compaction_start_backlog", starting_backlog)
    _set_gauge(service.runtime, "epoch_compaction_deleted", deleted_total)
    _set_gauge(service.runtime, "epoch_compaction_batches", batches)
    _set_gauge(service.runtime, "epoch_compaction_seconds", elapsed)
    _set_gauge(service.runtime, "epoch_compaction_remaining_backlog", int(manager.backlog()))
    return deleted_total


def install(pipeline_cls: type) -> None:
    if getattr(pipeline_cls, "_canonical_hotpath_installed", False):
        return

    global _ORIGINAL_APPEND_DIRTY
    global _ORIGINAL_APPLY_CANONICAL
    global _ORIGINAL_DERIVATION_CANDIDATES

    _ORIGINAL_APPEND_DIRTY = _canonical_commit._append_dirty_normalized
    _ORIGINAL_APPLY_CANONICAL = _canonical_commit.apply_canonical_commit_batch
    _ORIGINAL_DERIVATION_CANDIDATES = _canonical_derivation.derivation_candidates

    _canonical_commit._append_dirty_normalized = _append_dirty_normalized_coalesced
    _canonical_commit.apply_canonical_commit_batch = _apply_canonical_commit_coalesced
    _publication_throughput.apply_canonical_commit_batch = _apply_canonical_commit_coalesced
    _canonical_derivation.derivation_candidates = _derivation_candidates_with_aggregate_support
    _canonical_commit.derivation_candidates = _derivation_candidates_with_aggregate_support

    # Keep enough prepared work to clear a full 50k-class epoch backlog during
    # the bounded catch-up window without increasing the 2,048-node delete batch.
    _residency._PREPARED_COMPACTION_BATCH_LIMIT = max(
        int(_residency._PREPARED_COMPACTION_BATCH_LIMIT), 16
    )

    original_service = pipeline_cls.service
    original_apply_ingest_ready = pipeline_cls.apply_ingest_ready
    original_shutdown = pipeline_cls.shutdown_parallel_pipeline

    def apply_ingest_ready(self: Any) -> bool:
        progressed = bool(original_apply_ingest_ready(self))
        _request_residency_work(self)
        return progressed

    def service(self: Any) -> bool:
        progressed = bool(original_service(self))
        deleted = _service_prepared_compaction_if_idle(self, max_batches=1)
        return bool(progressed or deleted)

    def shutdown_parallel_pipeline(self: Any) -> None:
        original_shutdown(self)
        _catch_up_residency_at_epoch_boundary(self)
        _publish_family_diagnostics(self.runtime)

    pipeline_cls.apply_ingest_ready = apply_ingest_ready
    pipeline_cls.service = service
    pipeline_cls.shutdown_parallel_pipeline = shutdown_parallel_pipeline
    pipeline_cls._canonical_hotpath_installed = True
