from __future__ import annotations

import pytest

from v7.memory.coordinator import GenerationCommitCoordinator
from v7.memory.durable_store import DurableGenerationStore
from v7.memory.ids import MemoryIdAllocator, MemoryLevel
from v7.memory.models import NodeMutation
from v7.memory.publisher import GenerationPublisher
from v7.memory.transport.base import ReadViewHandle
from v7.memory.transport.local import LocalReadViewTransport
from v7.memory.writer import CanonicalMemoryWriter


class FailOnceTransport:
    def __init__(self) -> None:
        self.inner = LocalReadViewTransport()
        self.failed = False

    def publish(self, view):
        if not self.failed:
            self.failed = True
            raise OSError("publication unavailable")
        return self.inner.publish(view)

    def attach(self, handle: ReadViewHandle):
        return self.inner.attach(handle)

    def release(self, handle: ReadViewHandle) -> None:
        self.inner.release(handle)


def test_publication_failure_keeps_prepared_generation_for_retry(tmp_path) -> None:
    memory_id = MemoryIdAllocator().allocate()
    writer = CanonicalMemoryWriter()
    writer.apply_mutation_batch(
        [NodeMutation(memory_id, MemoryLevel.M1, 10, support_delta=1)]
    )
    durable = DurableGenerationStore(tmp_path / "state.sqlite")
    publisher = GenerationPublisher(FailOnceTransport())
    coordinator = GenerationCommitCoordinator(
        writer=writer,
        durable_store=durable,
        publisher=publisher,
    )
    try:
        with pytest.raises(OSError, match="publication unavailable"):
            coordinator.commit(batch_id=3)

        assert writer.has_pending_generation
        assert int(writer.published_view.generation_id) == 0
        assert durable.connection.execute(
            "SELECT committed FROM generations WHERE generation_id=1"
        ).fetchone() == (1,)

        result = coordinator.commit(batch_id=3)
        assert int(result.state.generation_id) == 1
        assert not writer.has_pending_generation
        assert writer.published_view is result.view
        assert publisher.attach_current() is result.view
    finally:
        durable.close()


def test_ensure_published_is_idempotent_for_same_generation() -> None:
    memory_id = MemoryIdAllocator().allocate()
    writer = CanonicalMemoryWriter()
    writer.apply_mutation_batch(
        [NodeMutation(memory_id, MemoryLevel.M1, 10, support_delta=1)]
    )
    _, view, _ = writer.commit_generation()
    publisher = GenerationPublisher(LocalReadViewTransport())

    first = publisher.ensure_published(view)
    second = publisher.ensure_published(view)

    assert first == second
    assert publisher.attach_current() is view
