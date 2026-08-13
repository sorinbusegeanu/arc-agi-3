from __future__ import annotations

import pytest

from v7.memory.coordinator import GenerationCommitCoordinator
from v7.memory.durable_store import DurableGenerationStore
from v7.memory.generation import GenerationId
from v7.memory.ids import MemoryIdAllocator, MemoryLevel
from v7.memory.indexes.cognition import ContingencyIndexMutation
from v7.memory.models import NodeMutation
from v7.memory.publisher import GenerationPublisher
from v7.memory.transport.local import LocalReadViewTransport
from v7.memory.writer import CanonicalMemoryWriter


def _commit_with_contingency(writer: CanonicalMemoryWriter, *, context: int, action: int):
    memory_id = MemoryIdAllocator().allocate()
    writer.apply_mutation_batch(
        [NodeMutation(memory_id, MemoryLevel.M1, 10, support_delta=1)]
    )
    writer.apply_contingency_index_batch(
        [ContingencyIndexMutation(context, action, memory_id)]
    )
    _, view, _ = writer.commit_generation()
    return memory_id, view


def test_publisher_exposes_only_committed_immutable_generation() -> None:
    writer = CanonicalMemoryWriter()
    transport = LocalReadViewTransport()
    publisher = GenerationPublisher(transport)

    memory_id, view = _commit_with_contingency(writer, context=11, action=2)
    record = publisher.publish(view)

    assert int(record.generation_id) == 1
    attached = publisher.attach_current()
    assert attached is view
    assert attached is not None
    assert attached.score_inputs(context_signature=11, action_ids=[2])[0].contingency_ids == (memory_id,)


def test_reader_refreshes_only_when_generation_advances() -> None:
    writer = CanonicalMemoryWriter()
    publisher = GenerationPublisher(LocalReadViewTransport())

    _, view_1 = _commit_with_contingency(writer, context=1, action=1)
    publisher.publish(view_1)

    assert publisher.attach_if_newer(GenerationId(0)) is view_1
    assert publisher.attach_if_newer(GenerationId(1)) is None

    _, view_2 = _commit_with_contingency(writer, context=2, action=2)
    publisher.publish(view_2)

    assert publisher.attach_if_newer(GenerationId(1)) is view_2
    assert publisher.attach_if_newer(GenerationId(2)) is None


def test_publisher_rejects_duplicate_or_regressing_generations() -> None:
    writer = CanonicalMemoryWriter()
    publisher = GenerationPublisher(LocalReadViewTransport())
    _, view_1 = _commit_with_contingency(writer, context=1, action=1)
    publisher.publish(view_1)

    with pytest.raises(ValueError, match="increase monotonically"):
        publisher.publish(view_1)


def test_transport_keeps_old_generation_attachable_until_release() -> None:
    writer = CanonicalMemoryWriter()
    transport = LocalReadViewTransport()
    publisher = GenerationPublisher(transport)

    _, view_1 = _commit_with_contingency(writer, context=1, action=1)
    record_1 = publisher.publish(view_1)
    _, view_2 = _commit_with_contingency(writer, context=2, action=2)
    record_2 = publisher.publish(view_2)

    assert transport.attach(record_1.handle) is view_1
    assert transport.attach(record_2.handle) is view_2
    assert transport.retained_generations == (1, 2)

    publisher.release(record_1)
    assert transport.retained_generations == (2,)
    with pytest.raises(KeyError):
        transport.attach(record_1.handle)

    with pytest.raises(ValueError, match="currently published"):
        publisher.release(record_2)


def test_unpublished_mutations_are_not_visible_to_attached_reader() -> None:
    ids = MemoryIdAllocator()
    first_id, second_id = ids.allocate_many(2)
    writer = CanonicalMemoryWriter()
    publisher = GenerationPublisher(LocalReadViewTransport())

    writer.apply_mutation_batch(
        [NodeMutation(first_id, MemoryLevel.M1, 10, support_delta=1)]
    )
    _, view_1, _ = writer.commit_generation()
    publisher.publish(view_1)

    writer.apply_mutation_batch(
        [NodeMutation(second_id, MemoryLevel.M1, 10, support_delta=1)]
    )

    attached = publisher.attach_current()
    assert attached is not None
    assert attached.get_nodes([first_id, second_id])[0] is not None
    assert attached.get_nodes([first_id, second_id])[1] is None


def test_prepare_blocks_mutation_until_finalize_or_abort() -> None:
    ids = MemoryIdAllocator()
    first_id, second_id = ids.allocate_many(2)
    writer = CanonicalMemoryWriter()
    writer.apply_mutation_batch(
        [NodeMutation(first_id, MemoryLevel.M1, 10, support_delta=1)]
    )

    prepared = writer.prepare_generation()
    assert writer.has_pending_generation
    assert writer.prepare_generation() is prepared

    with pytest.raises(RuntimeError, match="generation is prepared"):
        writer.apply_mutation_batch(
            [NodeMutation(second_id, MemoryLevel.M1, 10, support_delta=1)]
        )

    writer.abort_generation()
    assert not writer.has_pending_generation
    writer.apply_mutation_batch(
        [NodeMutation(second_id, MemoryLevel.M1, 10, support_delta=1)]
    )


def test_coordinator_persists_before_publishing_and_finalizing(tmp_path) -> None:
    memory_id = MemoryIdAllocator().allocate()
    writer = CanonicalMemoryWriter()
    writer.apply_mutation_batch(
        [NodeMutation(memory_id, MemoryLevel.M1, 10, support_delta=1)]
    )
    durable = DurableGenerationStore(tmp_path / "state.sqlite")
    publisher = GenerationPublisher(LocalReadViewTransport())
    coordinator = GenerationCommitCoordinator(
        writer=writer,
        durable_store=durable,
        publisher=publisher,
    )
    try:
        result = coordinator.commit(batch_id=7)
        assert int(result.state.generation_id) == 1
        assert publisher.attach_current() is result.view
        assert writer.published_view is result.view
        assert int(writer.mutable_generation_id) == 2
        assert not writer.has_pending_generation
        assert durable.connection.execute(
            "SELECT committed FROM generations WHERE generation_id=1"
        ).fetchone() == (1,)
        assert durable.connection.execute(
            "SELECT mutation_count FROM generation_batches WHERE generation_id=1 AND batch_id=7"
        ).fetchone() == (1,)
    finally:
        durable.close()


def test_durability_failure_does_not_publish_or_advance_writer() -> None:
    class FailingStore:
        def persist_generation_delta(self, state, delta, *, batch_id=0):
            raise OSError("disk unavailable")

    memory_id = MemoryIdAllocator().allocate()
    writer = CanonicalMemoryWriter()
    writer.apply_mutation_batch(
        [NodeMutation(memory_id, MemoryLevel.M1, 10, support_delta=1)]
    )
    publisher = GenerationPublisher(LocalReadViewTransport())
    coordinator = GenerationCommitCoordinator(
        writer=writer,
        durable_store=FailingStore(),
        publisher=publisher,
    )

    with pytest.raises(OSError, match="disk unavailable"):
        coordinator.commit()

    assert publisher.current_record is None
    assert int(writer.published_view.generation_id) == 0
    assert int(writer.mutable_generation_id) == 1
    assert not writer.has_pending_generation
    assert writer.dirty_counts["nodes"] == 1
