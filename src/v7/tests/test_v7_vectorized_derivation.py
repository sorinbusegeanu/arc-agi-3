from __future__ import annotations

import numpy as np

from v7.derivation.batches import DerivedMutationBatch, DeterministicDerivedBatchMerger
from v7.derivation.vectorized import VectorizedDerivationEngine
from v7.derivation.workers import DerivationTask, DerivationTaskResult
from v7.memory.generation import GenerationId
from v7.memory.ids import MemoryIdAllocator, MemoryLevel
from v7.memory.models import MemoryNode, MemoryScore, NodeMutation, ScoreMutation
from v7.memory.read_view import MemoryReadView
from v7.memory.writer import CanonicalMemoryWriter


def _view():
    ids = MemoryIdAllocator()
    first_id, second_id, target_id = ids.allocate_many(3)
    return MemoryReadView.freeze(
        generation_id=GenerationId(1),
        nodes={
            first_id: MemoryNode(first_id, MemoryLevel.M3, 31, GenerationId(1), GenerationId(1), support_count=2),
            second_id: MemoryNode(second_id, MemoryLevel.M3, 31, GenerationId(1), GenerationId(1), status_flags=4, support_count=5),
            target_id: MemoryNode(target_id, MemoryLevel.M4, 41, GenerationId(1), GenerationId(1), support_count=1),
        },
        scores={
            first_id: MemoryScore(first_id, significance=0.25, learning_value=0.5),
            second_id: MemoryScore(second_id, significance=0.75, future_option_delta=2.0),
        },
        adjacency={},
    ), (first_id, second_id, target_id)


def test_vectorized_engine_builds_read_only_dense_batch() -> None:
    view, ids = _view()
    first_id, second_id, _ = ids
    task = DerivationTask(GenerationId(1), MemoryLevel.M3, (first_id, second_id))
    batch = VectorizedDerivationEngine().build_input(view, task)

    assert batch.count == 2
    assert np.array_equal(batch.support_counts, np.array([2, 5], dtype=np.int64))
    assert np.array_equal(batch.status_flags, np.array([0, 4], dtype=np.uint64))
    assert np.allclose(batch.score_matrix[:, 0], [0.25, 0.75])
    assert np.allclose(batch.score_matrix[:, 2], [0.5, 0.0])
    assert np.allclose(batch.score_matrix[:, 5], [0.0, 2.0])
    assert not batch.memory_ids.flags.writeable
    assert not batch.score_matrix.flags.writeable


def test_vectorized_kernel_runs_once_for_whole_chunk() -> None:
    view, ids = _view()
    first_id, second_id, _ = ids
    task = DerivationTask(GenerationId(1), MemoryLevel.M3, (first_id, second_id))
    calls = []

    def kernel(batch):
        calls.append(batch.count)
        return DerivedMutationBatch(
            generation_id=batch.generation_id,
            source_level=batch.level,
            source_ids=task.memory_ids,
            score_mutations=tuple(
                ScoreMutation(memory_id, learning_value=float(value) / 10.0)
                for memory_id, value in zip(task.memory_ids, batch.support_counts, strict=True)
            ),
        )

    result = VectorizedDerivationEngine().run(view, task, kernel)
    assert calls == [2]
    assert [m.learning_value for m in result.output.score_mutations] == [0.2, 0.5]


def test_vectorized_engine_rejects_wrong_level_ids() -> None:
    view, ids = _view()
    target_id = ids[2]
    task = DerivationTask(GenerationId(1), MemoryLevel.M3, (target_id,))
    try:
        VectorizedDerivationEngine().build_input(view, task)
    except ValueError as exc:
        assert "different memory level" in str(exc)
    else:
        raise AssertionError("wrong-level task was accepted")


def _writer_with_target(target_id):
    writer = CanonicalMemoryWriter(initial_generation=1)
    writer.apply_mutation_batch([NodeMutation(target_id, MemoryLevel.M4, 41, support_delta=1)])
    return writer


def test_merge_is_independent_of_worker_completion_order() -> None:
    _, ids = _view()
    first_id, second_id, target_id = ids
    task_a = DerivationTask(GenerationId(1), MemoryLevel.M3, (first_id,))
    task_b = DerivationTask(GenerationId(1), MemoryLevel.M3, (second_id,))
    result_a = DerivationTaskResult(task_a, DerivedMutationBatch(
        GenerationId(1), MemoryLevel.M3, (first_id,),
        score_mutations=(ScoreMutation(target_id, significance=0.25),),
    ))
    result_b = DerivationTaskResult(task_b, DerivedMutationBatch(
        GenerationId(1), MemoryLevel.M3, (second_id,),
        score_mutations=(ScoreMutation(target_id, significance=0.75),),
    ))

    observed = []
    for results in ((result_a, result_b), (result_b, result_a)):
        writer = _writer_with_target(target_id)
        stats = DeterministicDerivedBatchMerger().apply(results, writer=writer)
        _, published, _ = writer.commit_generation()
        observed.append(published.get_scores([target_id])[0].significance)
        assert stats.batches == 2
        assert stats.scores == 2

    assert observed == [0.75, 0.75]


def test_merge_rejects_provenance_mismatch_before_writes() -> None:
    _, ids = _view()
    first_id, _, target_id = ids
    task = DerivationTask(GenerationId(1), MemoryLevel.M3, (first_id,))
    bad = DerivationTaskResult(task, DerivedMutationBatch(
        GenerationId(1), MemoryLevel.M3, (target_id,),
        score_mutations=(ScoreMutation(target_id, significance=1.0),),
    ))
    writer = _writer_with_target(target_id)
    before = dict(writer.dirty_counts)
    try:
        DeterministicDerivedBatchMerger().apply([bad], writer=writer)
    except ValueError as exc:
        assert "provenance" in str(exc)
    else:
        raise AssertionError("invalid provenance was accepted")
    assert writer.dirty_counts == before
