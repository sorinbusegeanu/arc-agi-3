from __future__ import annotations

import time
from typing import Any

from .canonical_commit import (
    apply_canonical_commit_batch,
    canonical_commit_prefix_length,
)
from .memory_pipeline import PreparedCommitBatch


def _row_input_bytes(batch: PreparedCommitBatch) -> tuple[int, ...]:
    rows = tuple(batch.rows)
    measured = tuple(int(value) for value in (getattr(batch, "row_input_bytes", ()) or ()))
    if measured:
        if len(measured) != len(rows):
            raise ValueError("prepared commit byte measurements must match rows")
        return measured
    total = int(getattr(batch, "input_bytes", 0))
    if not rows:
        return ()
    if total < 0:
        raise ValueError("prepared commit input bytes cannot be negative")
    base, remainder = divmod(total, len(rows))
    return tuple(base + int(index < remainder) for index in range(len(rows)))


def _slice_batch(batch: PreparedCommitBatch, start: int) -> PreparedCommitBatch:
    rows = tuple(batch.rows)[int(start):]
    row_bytes = _row_input_bytes(batch)[int(start):]
    if not rows:
        raise ValueError("cannot create empty prepared commit batch slice")
    return PreparedCommitBatch(
        int(rows[0].sequence),
        int(rows[-1].sequence),
        rows,
        sum(row_bytes),
        row_bytes,
    )


def install(service_cls: type) -> None:
    if getattr(service_cls, "_canonical_batch_budgeting_installed", False):
        return

    def apply_ingest_ready(self: Any) -> bool:
        plans: list[Any] = []
        plan_input_bytes: list[int] = []
        transaction_row_limit = min(
            int(self.canonical_batch_size),
            int(getattr(self.runtime.config, "canonical_transaction_max_rows", 1024)),
        )

        while self.ingest_apply in self.ingest_results:
            batch = self.ingest_results[self.ingest_apply]
            rows = tuple(batch.rows)
            row_bytes = _row_input_bytes(batch)
            if not rows:
                self.ingest_results.pop(self.ingest_apply)
                self.ingest_apply = int(batch.end_sequence) + 1
                continue

            row_cap = min(len(rows), max(0, transaction_row_limit - len(plans)))
            if row_cap <= 0:
                break

            candidate_rows = tuple(plans) + rows[:row_cap]
            candidate_bytes = tuple(plan_input_bytes) + row_bytes[:row_cap]
            accepted_prefix = canonical_commit_prefix_length(
                self.runtime,
                candidate_rows,
                row_input_bytes=candidate_bytes,
            )
            accepted_from_batch = int(accepted_prefix) - len(plans)
            if accepted_from_batch <= 0:
                break

            self.ingest_results.pop(self.ingest_apply)
            plans.extend(rows[:accepted_from_batch])
            plan_input_bytes.extend(row_bytes[:accepted_from_batch])

            if accepted_from_batch < len(rows):
                tail = _slice_batch(batch, accepted_from_batch)
                self.ingest_results[int(tail.start_sequence)] = tail
                self.ingest_apply = int(tail.start_sequence)
                break

            self.ingest_apply = int(batch.end_sequence) + 1
            if len(plans) >= transaction_row_limit:
                break

        if not plans:
            return False

        input_bytes = sum(plan_input_bytes)
        started = time.perf_counter()
        result = apply_canonical_commit_batch(self.runtime, plans, input_bytes=input_bytes)
        elapsed = time.perf_counter() - started
        self.canonical_apply_seconds += elapsed
        self.canonical_apply_events += len(plans)
        self.last_canonical_ingest_batch = len(plans)
        self.ingested += len(plans)
        self.release_ingest_input_bytes(input_bytes)

        resident_manager = getattr(self.runtime, "_resident_memory", None)
        if resident_manager is not None:
            resident_manager.maybe_compact()

        for candidate in result.derivation_candidates:
            self._consider_candidate(candidate)

        backlog = max(
            len(self.pending_ingest),
            len(self.ingest_results),
            max(0, self.sampled - self.ingested),
        )
        # Keep the direct coordinator's existing adaptive policy, but it now
        # applies only after budget-safe prefixing has selected the transaction.
        from .parallel_memory_coordinator import _adaptive_canonical_batch_size

        self.canonical_batch_size = _adaptive_canonical_batch_size(
            self.canonical_batch_size,
            backlog,
        )
        return True

    service_cls.apply_ingest_ready = apply_ingest_ready
    service_cls._canonical_batch_budgeting_installed = True
