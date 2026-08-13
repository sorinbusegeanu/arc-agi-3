from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from v7.memory.delta import GenerationDelta
from v7.memory.generation import GenerationState
from v7.memory.publisher import GenerationPublisher, PublicationRecord
from v7.memory.read_view import MemoryReadView
from v7.memory.writer import CanonicalMemoryWriter


class DurableGenerationSink(Protocol):
    def persist_generation_delta(
        self,
        state: GenerationState,
        delta: GenerationDelta,
        *,
        batch_id: int = 0,
    ) -> None:
        ...


@dataclass(frozen=True, slots=True)
class GenerationCommitResult:
    state: GenerationState
    view: MemoryReadView
    delta: GenerationDelta
    publication: PublicationRecord


class GenerationCommitCoordinator:
    """Enforce durable-before-publish ordering for one canonical generation."""

    def __init__(
        self,
        *,
        writer: CanonicalMemoryWriter,
        durable_store: DurableGenerationSink,
        publisher: GenerationPublisher,
    ) -> None:
        self._writer = writer
        self._durable_store = durable_store
        self._publisher = publisher

    def commit(self, *, batch_id: int = 0) -> GenerationCommitResult:
        state, view, delta = self._writer.prepare_generation()
        try:
            self._durable_store.persist_generation_delta(
                state,
                delta,
                batch_id=batch_id,
            )
        except Exception:
            self._writer.abort_generation()
            raise

        publication = self._publisher.publish(view)
        self._writer.finalize_generation()
        return GenerationCommitResult(
            state=state,
            view=view,
            delta=delta,
            publication=publication,
        )
