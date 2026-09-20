from __future__ import annotations

import queue

import pytest

from v9.runtime.multiprocess import EncodedTransition, ProcessTopology, TransitionBatchEnvelope, WorkerStop, shard_worker_main, stage_worker_main
from v9.runtime.shared_batch_transport import ProducerAffinityRouter


def _transition(sequence: int, *, actor_id: int = 1) -> EncodedTransition:
    return EncodedTransition(
        actor_id, sequence, sequence, ("family", "type", "config", "instance"), 1, 1, 1, 1, 2, 2, 0, (), None, "game"
    )


def test_transition_batch_envelope_is_contiguous_and_bounded() -> None:
    rows = (_transition(1), _transition(2))
    assert TransitionBatchEnvelope(1, 1, 2, rows).transitions == rows
    with pytest.raises(ValueError, match="contiguous"):
        TransitionBatchEnvelope(9, 1, 3, rows)
    with pytest.raises(ValueError, match="1-64"):
        TransitionBatchEnvelope(9, 1, 1, ())


def test_stage_routing_keeps_every_sequence_for_one_producer_on_one_shard() -> None:
    stage: queue.Queue[object] = queue.Queue()
    shards = tuple(queue.Queue() for _ in range(4))
    rows = tuple(_transition(sequence, actor_id=17) for sequence in range(1, 5))
    for row in rows:
        stage.put(row)
    stage.put(WorkerStop())

    stage_worker_main(stage, shards)

    expected = ProducerAffinityRouter(4).shard_for(17)
    routed = tuple(shards[expected].get_nowait() for _ in rows)
    assert tuple(envelope.transitions[0] for envelope in routed) == rows
    assert all(shard.empty() for index, shard in enumerate(shards) if index != expected)


def test_shard_causal_admission_reorders_one_producer_before_publication() -> None:
    shard: queue.Queue[object] = queue.Queue()
    publication: queue.Queue[object] = queue.Queue()
    rows = tuple(_transition(sequence, actor_id=23) for sequence in range(1, 4))
    for index in (1, 0, 2):
        shard.put(rows[index])
    shard.put(WorkerStop())

    shard_worker_main(2, shard, publication, reorder_row_limit=4)

    published = tuple(publication.get_nowait() for _ in range(2))
    assert tuple(row for item in published for row in item[3].transitions) == rows
    assert publication.get_nowait() == ("shard_done", 2, 3)


def test_shard_shutdown_reports_an_explicit_producer_gap() -> None:
    shard: queue.Queue[object] = queue.Queue()
    publication: queue.Queue[object] = queue.Queue()
    shard.put(_transition(2, actor_id=29))
    shard.put(WorkerStop())

    shard_worker_main(1, shard, publication, reorder_row_limit=4)

    error = publication.get_nowait()
    assert error[:3] == ("shard_error", 1, 0)
    assert "producer causal gap" in error[3]


def test_envelope_protocol_crosses_stage_and_shard_processes() -> None:
    topology = ProcessTopology(
        actors=0,
        stage_workers=2,
        shards=2,
        queue_capacity=16,
        start_method="spawn",
    )
    rows = tuple(_transition(sequence, actor_id=31) for sequence in range(1, 4))
    try:
        topology.start_workers()
        topology.stage_queue.put(TransitionBatchEnvelope(31, 2, 3, rows[1:]))
        topology.stage_queue.put(TransitionBatchEnvelope(31, 1, 1, rows[:1]))
        topology.signal_stage_stop()
        topology.join_stage_workers()
        topology.stop_shard_workers()
        topology.join_shard_workers()

        published = []
        completed = 0
        while completed < 2:
            item = topology.publication_queue.get(timeout=5.0)
            if item[0] == "transition_batch":
                published.extend(item[3].transitions)
            elif item[0] == "shard_done":
                completed += 1
        assert tuple(published) == rows
    finally:
        topology.terminate()
        topology.close(drain=True)
