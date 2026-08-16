from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace

from v8.arena import EdgeRecord, NodeRecord
from v8.lifecycle import LifecycleController
from v8.model import (
    CognitiveState,
    EventId,
    MemoryLevel,
    MemoryProposal,
    MemoryType,
    MemoryUid,
    RelationType,
    ValidationState,
    proposal_fingerprint,
)
from v8.pruning import PruningPlanner
from v8.runtime import ContinuousMemoryRuntime, V8RuntimeConfig


def _node(
    level: MemoryLevel,
    memory_type: MemoryType,
    key: tuple[int, ...],
    *,
    state: CognitiveState = CognitiveState.ACTIVE,
    validation: ValidationState = ValidationState.UNTESTED,
    support: int = 1,
    valence_sum: float = 0.0,
    valence_weight: float = 0.0,
) -> NodeRecord:
    return NodeRecord(
        uid=MemoryUid.from_key(level, memory_type, key),
        fingerprint=proposal_fingerprint(level, memory_type, key),
        level=int(level),
        memory_type=int(memory_type),
        key_parts=key,
        support_count=int(support),
        significance_sum=0.0,
        prediction_error_sum=0.0,
        learning_value_sum=0.0,
        transfer_prior_sum=0.0,
        explanatory_sum=0.0,
        future_option_sum=0.0,
        score_weight=1.0,
        updated_watermark=1,
        cognitive_state=int(state),
        validation_state=int(validation),
        primary_valence_sum=float(valence_sum),
        primary_valence_sq_sum=abs(float(valence_sum)),
        primary_valence_weight=float(valence_weight),
    )


def _edge(source: NodeRecord, relation: RelationType, target: NodeRecord) -> EdgeRecord:
    return EdgeRecord(source.uid, int(relation), target.uid, 1, 1)


class LowLevelLifecycleTests(unittest.TestCase):
    def test_m0_uses_same_hysteresis_state_machine(self) -> None:
        controller = LifecycleController()
        controller.set_developmental_stage(6)
        row = _node(MemoryLevel.M0, MemoryType.EPISODE, (1, 1))

        self.assertIsNone(controller.decide(row))
        self.assertIsNone(controller.decide(row))
        decision = controller.decide(row)
        self.assertIsNotNone(decision)
        self.assertEqual(decision.cognitive_state, int(CognitiveState.QUARANTINED))
        self.assertEqual(decision.reason, "three low-fitness windows")

        quarantined = replace(row, cognitive_state=int(CognitiveState.QUARANTINED))
        self.assertIsNone(controller.decide(quarantined))
        self.assertIsNone(controller.decide(quarantined))
        decision = controller.decide(quarantined)
        self.assertIsNotNone(decision)
        self.assertEqual(decision.cognitive_state, int(CognitiveState.RETIRE_PENDING))
        self.assertEqual(decision.reason, "six low-fitness windows")

    def test_m0_requires_active_semantic_replacement(self) -> None:
        m0 = _node(
            MemoryLevel.M0,
            MemoryType.EPISODE,
            (2, 1),
            state=CognitiveState.RETIRE_PENDING,
        )
        m1 = _node(MemoryLevel.M1, MemoryType.CONTINGENCY, (10, 1, 20, 11))
        planner = PruningPlanner()

        blocked = planner.candidates((m0, m1), (_edge(m1, RelationType.EXPLAINS, m0),))[0]
        self.assertFalse(blocked.safe_to_retire)

        safe = planner.candidates(
            (m0, m1),
            (
                _edge(m1, RelationType.EXPLAINS, m0),
                _edge(m1, RelationType.SUPERSEDES, m0),
            ),
        )[0]
        self.assertTrue(safe.safe_to_retire)
        self.assertTrue(safe.has_semantic_replacement)

    def test_m0_replacement_must_preserve_valence_direction(self) -> None:
        m0 = _node(
            MemoryLevel.M0,
            MemoryType.EPISODE,
            (3, 1),
            state=CognitiveState.RETIRE_PENDING,
            valence_sum=2.0,
            valence_weight=2.0,
        )
        bad = _node(
            MemoryLevel.M1,
            MemoryType.CONTINGENCY,
            (20, 1, 21, 22),
            valence_sum=-2.0,
            valence_weight=2.0,
        )
        good = replace(bad, primary_valence_sum=2.0)
        planner = PruningPlanner()

        candidate = planner.candidates(
            (m0, bad), (_edge(bad, RelationType.SUPERSEDES, m0),)
        )[0]
        self.assertFalse(candidate.safe_to_retire)

        candidate = planner.candidates(
            (m0, good), (_edge(good, RelationType.SUPERSEDES, m0),)
        )[0]
        self.assertTrue(candidate.safe_to_retire)

    def test_normalized_m1_requires_m2_or_higher_replacement(self) -> None:
        m1n = _node(
            MemoryLevel.M1,
            MemoryType.CONTINGENCY,
            (777,),
            state=CognitiveState.RETIRE_PENDING,
        )
        m1 = _node(MemoryLevel.M1, MemoryType.CONTINGENCY, (30, 1, 31, 32))
        m2 = _node(MemoryLevel.M2, MemoryType.FAMILY, (888,))
        planner = PruningPlanner()

        candidate = planner.candidates(
            (m1n, m1), (_edge(m1, RelationType.SUPERSEDES, m1n),)
        )[0]
        self.assertFalse(candidate.safe_to_retire)

        candidate = planner.candidates(
            (m1n, m2), (_edge(m2, RelationType.SUPERSEDES, m1n),)
        )[0]
        self.assertTrue(candidate.safe_to_retire)

    def test_grounded_m1_cannot_physically_retire_while_it_is_control_anchor(self) -> None:
        m1g = _node(
            MemoryLevel.M1,
            MemoryType.CONTINGENCY,
            (40, 1, 41, 42),
            state=CognitiveState.RETIRE_PENDING,
        )
        m2 = _node(MemoryLevel.M2, MemoryType.FAMILY, (999,))
        candidate = PruningPlanner().candidates(
            (m1g, m2), (_edge(m2, RelationType.SUPERSEDES, m1g),)
        )[0]
        self.assertFalse(candidate.safe_to_retire)
        self.assertTrue(candidate.protected_by_dependencies)

    def test_compaction_promotes_m0_game_provenance_to_superseder(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = ContinuousMemoryRuntime(
                V8RuntimeConfig.from_path(
                    tmp,
                    shards=1,
                    stage_workers=1,
                    enable_snapshots=False,
                    enable_peers=False,
                    node_capacity_per_shard=128,
                    edge_capacity_per_shard=256,
                    action_capacity_per_shard=64,
                )
            )
            runtime.start()
            m0_key = (51, 52)
            m0_uid = MemoryUid.from_key(MemoryLevel.M0, MemoryType.EPISODE, m0_key)
            runtime.submit_proposal(
                MemoryProposal(
                    uid=m0_uid,
                    fingerprint=proposal_fingerprint(MemoryLevel.M0, MemoryType.EPISODE, m0_key),
                    event_id=EventId.from_producer(90, 1),
                    watermark=1,
                    level=MemoryLevel.M0,
                    memory_type=MemoryType.EPISODE,
                    key_parts=m0_key,
                    source_game_hash=12345,
                    cognitive_state=int(CognitiveState.RETIRED),
                )
            )
            m1_key = (60, 1, 61, 62)
            m1_uid = MemoryUid.from_key(MemoryLevel.M1, MemoryType.CONTINGENCY, m1_key)
            runtime.submit_proposal(
                MemoryProposal(
                    uid=m1_uid,
                    fingerprint=proposal_fingerprint(MemoryLevel.M1, MemoryType.CONTINGENCY, m1_key),
                    event_id=EventId.from_producer(90, 2),
                    watermark=2,
                    level=MemoryLevel.M1,
                    memory_type=MemoryType.CONTINGENCY,
                    key_parts=m1_key,
                    parent_uid=m0_uid,
                    relation_type=RelationType.SUPERSEDES,
                    source_game_hash=0,
                    cognitive_state=int(CognitiveState.ACTIVE),
                )
            )
            runtime.wait_quiescent(timeout=10)

            result = runtime.compact_retired_memory(timeout=10)
            self.assertEqual(result.retired_nodes, 1)
            self.assertFalse(runtime.read_view.has_uid(m0_uid))
            self.assertTrue(runtime.read_view.has_uid(m1_uid))
            game_uid = MemoryUid(0, 12345)
            self.assertTrue(
                any(
                    edge.source_uid == m1_uid
                    and int(edge.relation_type) == int(RelationType.GAME_PROVENANCE)
                    and edge.target_uid == game_uid
                    for edge in runtime.read_view.edge_records()
                )
            )
            runtime.close(normal=True, timeout=10)

    def test_compactor_fail_closes_on_retired_grounded_m1(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = ContinuousMemoryRuntime(
                V8RuntimeConfig.from_path(
                    tmp,
                    shards=1,
                    stage_workers=1,
                    enable_snapshots=False,
                    enable_peers=False,
                    node_capacity_per_shard=64,
                    edge_capacity_per_shard=128,
                    action_capacity_per_shard=64,
                )
            )
            runtime.start()
            key = (70, 2, 71, 72)
            uid = MemoryUid.from_key(MemoryLevel.M1, MemoryType.CONTINGENCY, key)
            runtime.submit_proposal(
                MemoryProposal(
                    uid=uid,
                    fingerprint=proposal_fingerprint(MemoryLevel.M1, MemoryType.CONTINGENCY, key),
                    event_id=EventId.from_producer(91, 1),
                    watermark=1,
                    level=MemoryLevel.M1,
                    memory_type=MemoryType.CONTINGENCY,
                    key_parts=key,
                    cognitive_state=int(CognitiveState.RETIRED),
                )
            )
            runtime.wait_quiescent(timeout=10)
            result = runtime.compact_retired_memory(timeout=10)
            self.assertEqual(result.retired_nodes, 0)
            self.assertTrue(runtime.read_view.has_uid(uid))
            runtime.close(normal=True, timeout=10)


if __name__ == "__main__":
    unittest.main()
