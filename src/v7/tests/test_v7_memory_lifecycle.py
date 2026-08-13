from __future__ import annotations

from v7.derivation.pipeline import MemoryLearningPipeline
from v7.derivation.scientific import EpisodeEvidence
from v7.memory.evidence_lifecycle import ContradictionRecord, EvidenceLifecycleStore, TransferTrialRecord
from v7.memory.evidence_store import EvidenceStore
from v7.memory.ids import MemoryIdAllocator, MemoryLevel
from v7.memory.lifecycle import LifecyclePolicy, MemoryLifecycleController, MemoryStatus, ReplayQueue, ReplayRequest
from v7.memory.lifecycle_runtime import MemoryLifecycleRuntime
from v7.memory.models import MemoryNode, MemoryScore, NodeMutation, ScoreMutation
from v7.memory.generation import GenerationId
from v7.memory.read_view import MemoryReadView
from v7.memory.writer import CanonicalMemoryWriter


def test_lifecycle_promotes_demotes_and_queues_replay() -> None:
    ids = MemoryIdAllocator()
    strong_id, weak_id = ids.allocate_many(2)
    view = MemoryReadView.freeze(
        generation_id=GenerationId(1),
        nodes={
            strong_id: MemoryNode(strong_id, MemoryLevel.M2, 20, GenerationId(1), GenerationId(1), support_count=4),
            weak_id: MemoryNode(weak_id, MemoryLevel.M2, 20, GenerationId(1), GenerationId(1), support_count=1),
        },
        scores={
            strong_id: MemoryScore(strong_id, significance=1.0, prediction_error=0.7, learning_value=0.8, transfer_prior=0.8, explanatory_potential=0.9),
            weak_id: MemoryScore(weak_id),
        },
        adjacency={},
    )
    controller = MemoryLifecycleController(LifecyclePolicy(promote_threshold=0.5, retain_threshold=0.1))
    decisions = controller.evaluate(view)
    by_id = {item.memory_id: item for item in decisions}
    assert by_id[strong_id].promote is True
    assert by_id[strong_id].replay is True
    assert by_id[weak_id].demote is True
    assert controller.replay_queue.snapshot()[0].memory_id == strong_id


def test_replay_queue_is_bounded_and_deterministic() -> None:
    ids = MemoryIdAllocator()
    a, b, c = ids.allocate_many(3)
    queue = ReplayQueue(limit=2)
    queue.push(ReplayRequest(a, 0.5, 1))
    queue.push(ReplayRequest(b, 0.9, 1))
    queue.push(ReplayRequest(c, 0.7, 1))
    assert tuple(item.memory_id for item in queue.snapshot()) == (b, c)
    assert queue.pop_all() == (ReplayRequest(b, 0.9, 1), ReplayRequest(c, 0.7, 1))
    assert queue.snapshot() == ()


def test_lifecycle_runtime_updates_writer_flags_and_appends_evidence(tmp_path) -> None:
    memory_id = MemoryIdAllocator().allocate()
    writer = CanonicalMemoryWriter()
    writer.apply_mutation_batch((NodeMutation(memory_id, MemoryLevel.M2, 20, support_delta=3),))
    writer.apply_score_batch((ScoreMutation(memory_id, significance=1.0, prediction_error=0.8, learning_value=0.8, transfer_prior=0.8, explanatory_potential=0.8),))
    _, view, _ = writer.commit_generation()

    store = EvidenceStore(tmp_path / "events.sqlite")
    try:
        runtime = MemoryLifecycleRuntime(
            MemoryLifecycleController(LifecyclePolicy(promote_threshold=0.5)),
            evidence_store=store,
        )
        _, stats = runtime.run(view, writer=writer)
        assert stats.promoted == 1
        assert stats.replay_queued == 1
        assert stats.evidence_records == 2
        _, updated, _ = writer.commit_generation()
        flags = updated.nodes[memory_id].status_flags
        assert flags & int(MemoryStatus.PROMOTED)
        assert flags & int(MemoryStatus.REPLAY_QUEUED)
        assert store.connection.execute("SELECT COUNT(*) FROM evidence_records").fetchone() == (2,)
    finally:
        store.close()


def test_provenance_transfer_and_contradiction_ledgers_are_append_only(tmp_path) -> None:
    ledger = EvidenceLifecycleStore(tmp_path / "ledger.sqlite")
    try:
        writer = CanonicalMemoryWriter()
        pipeline = MemoryLearningPipeline(writer, ledger)
        m1a = pipeline.observe_episode(EpisodeEvidence(10, 2, 100, True, source_game="game-a", source_context="ctx-a", source_global_step=11))
        m1b = pipeline.observe_episode(EpisodeEvidence(11, 2, 101, True, source_game="game-b", source_context="ctx-b", source_global_step=12))
        m2 = pipeline.derive_m2(action_id=2, member_ids=(m1a, m1b), outcome_class=7)

        assert ledger.provenance_parents(m2) == tuple(sorted((m1a, m1b), key=int))
        ledger.append_transfer_trials((
            TransferTrialRecord(m2, 1, "game-a", "game-c", True, 0.9),
            TransferTrialRecord(m2, 1, "game-a", "game-d", False, 0.2),
        ))
        ledger.append_contradictions((ContradictionRecord(m2, 1, 0.75, source_game="game-e"),))
        summary = ledger.transfer_summary((m2,))
        assert summary[m2][0:2] == (2, 1)
        assert abs(summary[m2][2] - 0.55) < 1e-9
        assert ledger.connection.execute("SELECT COUNT(*) FROM contradiction_records").fetchone() == (1,)
    finally:
        ledger.close()


def test_runtime_keeps_empirical_transfer_distinct_and_uses_it_as_lifecycle_signal(tmp_path) -> None:
    memory_id = MemoryIdAllocator().allocate()
    writer = CanonicalMemoryWriter()
    writer.apply_mutation_batch((NodeMutation(memory_id, MemoryLevel.M3, 30, support_delta=3),))
    writer.apply_score_batch((ScoreMutation(memory_id, significance=0.2, transfer_prior=0.0),))
    _, view, _ = writer.commit_generation()

    ledger = EvidenceLifecycleStore(tmp_path / "transfer.sqlite")
    try:
        ledger.append_transfer_trials((
            TransferTrialRecord(memory_id, 1, "source", "target-a", True, 0.8),
            TransferTrialRecord(memory_id, 1, "source", "target-b", True, 0.9),
        ))
        policy = LifecyclePolicy(
            promote_threshold=0.10,
            retain_threshold=0.01,
            replay_prediction_error=2.0,
            replay_learning_value=2.0,
            replay_transfer_prior=2.0,
            replay_empirical_transfer=0.5,
            replay_explanatory_potential=2.0,
        )
        decisions, stats = MemoryLifecycleRuntime(
            MemoryLifecycleController(policy),
            evidence_lifecycle=ledger,
        ).run(view, writer=writer)
        assert stats.transfer_signals == 1
        assert decisions[0].empirical_transfer == 1.0
        assert decisions[0].replay is True
        assert view.scores[memory_id].transfer_prior == 0.0
    finally:
        ledger.close()
