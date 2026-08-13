from __future__ import annotations

from dataclasses import dataclass

from v7.memory.generation import GenerationId
from v7.memory.read_view import MemoryReadView
from v7.memory.transport.base import ReadViewHandle, ReadViewTransport


@dataclass(frozen=True, slots=True)
class PublicationRecord:
    generation_id: GenerationId
    handle: ReadViewHandle


class GenerationPublisher:
    """Single-writer atomic publication point for immutable read generations.

    The publisher owns only the tiny publication record. Read-view storage and
    attachment are delegated to a transport implementation. Readers first observe
    the record and attach only when the generation changes.
    """

    def __init__(self, transport: ReadViewTransport) -> None:
        self._transport = transport
        self._record: PublicationRecord | None = None

    @property
    def current_record(self) -> PublicationRecord | None:
        return self._record

    def publish(self, view: MemoryReadView) -> PublicationRecord:
        current = self._record
        if current is not None and int(view.generation_id) <= int(current.generation_id):
            raise ValueError("published generations must increase monotonically")

        handle = self._transport.publish(view)
        record = PublicationRecord(generation_id=view.generation_id, handle=handle)
        self._record = record
        return record

    def ensure_published(self, view: MemoryReadView) -> PublicationRecord:
        """Publish a generation or return its existing record on a safe retry."""

        current = self._record
        if current is not None:
            current_generation = int(current.generation_id)
            requested_generation = int(view.generation_id)
            if current_generation == requested_generation:
                return current
            if current_generation > requested_generation:
                raise ValueError("cannot publish an older generation")
        return self.publish(view)

    def attach_current(self) -> MemoryReadView | None:
        record = self._record
        if record is None:
            return None
        return self._transport.attach(record.handle)

    def attach_if_newer(self, generation_id: GenerationId | int) -> MemoryReadView | None:
        record = self._record
        if record is None or int(record.generation_id) <= int(generation_id):
            return None
        return self._transport.attach(record.handle)

    def release(self, record: PublicationRecord) -> None:
        current = self._record
        if current is not None and current == record:
            raise ValueError("cannot release the currently published generation")
        self._transport.release(record.handle)
