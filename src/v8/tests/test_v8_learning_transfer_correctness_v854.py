from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import v8
from v8.arena import EdgeRecord, NodeRecord
from v8.learning_transfer_correctness_v854 import (
    OrderedTransferSequence,
    _adaptive_composites,
    _build_restart_v854,
    _credit_v854,
    _ordered_action,
    _record_trial_v854,
    _relation_adjustment,
    _similarity_v854,
    transfer_relation_uid,
)
from v8.model import (
    CognitiveState,
    EventId,
    MemoryLevel,
    MemoryType,
    MemoryUid,
    RelationType,
    ValidationState,
    proposal_fingerprint,
    stable_u64,
)
from v8.similarity import BoundedNeighborhoodSimilarity, NeighborhoodDescriptor
from v8.structural_events import NormalizedPrimitive, StructuralFact
from v8.transfer import TransferValidator


def _node(level, memory_type, key, *, uid=None, support=5, future=1.0, transfer=0.0, watermark=1, game_mask=0):
    level = MemoryLevel(level)
    memory_type = MemoryType(memory_type)
    key = tuple(int(x) for x in key)
    uid = uid or MemoryUid.from_key(level, memory_type, key)
    weight = float(max(1, support))
    return NodeRecord(
        uid=uid,
        fingerprint=proposal_fingerprint(level, memory_type, key),
        level=int(level),
        memory_type=int(memory_type),
        key_parts=key,
        support_count=int(support),
        significance_sum=0.8 * weight,
        prediction_error_sum=0.0,
        learning_value_sum=0.8 * weight,
        transfer_prior_sum=float(transfer) * weight,
        explanatory_sum=0.5 * weight,
        future_option_sum=float(future) * weight,
        score_weight=weight,
        updated_watermark=int(watermark),
        game_mask=int(game_mask),
        cognitive_state=int(CognitiveState.ACTIVE),
        validation_state=int(ValidationState.TESTED),
    )


def _grounded(before, action, after, transition, *, support=5):
    return _node(MemoryLevel.M1, MemoryType.CONTINGENCY, (before, action, transition, after), support=support)


class _RestartView:
    def __init__(self, nodes, edges):
        self._strategy_version = (2, 2)
        self._node_by_uid = {row.uid: row for row in nodes}
        self._strategy_by_context = {}
        self._strategy_fallback = []
        self._behavior_strategy_dependencies = {}
        self._v815_restart_index_key = None
        self._v815_same_game_action_priors = {}
        self._v815_normalized_action_priors = {}
        self._v815_same_game_strategies = ()
        self._v815_session_action_priors = {}
        self._v815_session_trajectory = []
        self._v815_score_origins = {}
        self._v854_runtime_view = True
        self._v854_restart_key = None
        self._edges = tuple(edges)

    def edge_records(self):
        return self._edges


class TransferRelationTests(unittest.TestCase):
    def _supervisor(self, row):
        submitted = []
        evidence = []
        fake = SimpleNamespace(
            transfer=TransferValidator(),
            read_view=SimpleNamespace(node_records=lambda: (row,)),
            _submit=submitted.append,
            _event_id=lambda: EventId(1, len(submitted) + 1),
            current_watermark=lambda: 7,
            _append_evidence=lambda *args, **kwargs: evidence.append((args, kwargs)),
            _existing_proposal=lambda source, **kwargs: SimpleNamespace(uid=source.uid, **kwargs),
        )
        return fake, submitted, evidence

    def test_failed_targets_do_not_fail_or_quarantine_source(self):
        source = _node(MemoryLevel.M4, MemoryType.CONCEPT, (10, 20, 30, 40))
        supervisor, submitted, _evidence = self._supervisor(source)
        for target in (101, 102, 103):
            _record_trial_v854(
                supervisor,
                source.uid,
                target_game_hash=target,
                metric_on=0.0,
                metric_off=1.0,
            )
        self.assertEqual(len(submitted), 3)
        self.assertTrue(all(int(row.memory_type) == int(MemoryType.TRANSFER_EVIDENCE) for row in submitted))
        self.assertTrue(all(row.parent_uid == source.uid for row in submitted))
        self.assertTrue(all(int(row.validation_state) == int(ValidationState.TESTED) for row in submitted))
        self.assertTrue(all(float(row.transfer_prior_sum) < 0.0 for row in submitted))
        self.assertNotIn(source.uid, {row.uid for row in submitted})

    def test_one_success_validates_only_relation_two_targets_validate_concept(self):
        source = _node(MemoryLevel.M4, MemoryType.CONCEPT, (11, 22, 33, 44))
        supervisor, submitted, _evidence = self._supervisor(source)
        _record_trial_v854(supervisor, source.uid, target_game_hash=201, metric_on=1.0, metric_off=0.0)
        self.assertEqual([row.uid for row in submitted], [transfer_relation_uid(source.uid, 201)])
        _record_trial_v854(supervisor, source.uid, target_game_hash=202, metric_on=1.0, metric_off=0.0)
        source_updates = [row for row in submitted if getattr(row, "uid", None) == source.uid]
        self.assertEqual(len(source_updates), 1)
        self.assertEqual(source_updates[0].validation_state, int(ValidationState.VALIDATED))

    def test_negative_relation_blocks_only_its_target(self):
        source = _node(MemoryLevel.M7, MemoryType.STRATEGY, (1, 2, 3, 4))
        target_a = 501
        relation = _node(
            MemoryLevel.M4,
            MemoryType.TRANSFER_EVIDENCE,
            (
                source.uid.hi if source.uid.hi < (1 << 63) else source.uid.hi - (1 << 64),
                source.uid.lo if source.uid.lo < (1 << 63) else source.uid.lo - (1 << 64),
                target_a,
                1,
            ),
            uid=transfer_relation_uid(source.uid, target_a),
            support=2,
            transfer=-0.5,
        )
        view = SimpleNamespace(
            _node_by_uid={source.uid: source, relation.uid: relation},
            _parents={},
            _behavior_strategy_dependencies={},
            edge_records=lambda: (),
        )
        self.assertTrue(_relation_adjustment(view, source.uid, target_a)[0])
        self.assertFalse(_relation_adjustment(view, source.uid, 502)[0])


class NormalizedGroundingTests(unittest.TestCase):
    def test_foreign_raw_action_is_not_reused_in_target_world(self):
        from v8 import restart_memory_v815 as restart

        foreign = _grounded(10, 2, 77, 11)
        target = _grounded(20, 5, 88, 21)
        token = StructuralFact(NormalizedPrimitive.COMPONENT_RELOCATED, 55).token
        normalized = _node(MemoryLevel.M1, MemoryType.CONTINGENCY, (token,), support=8)
        target_hash, foreign_hash = 1001, 2002
        edges = (
            EdgeRecord(normalized.uid, int(RelationType.EXPLAINS), foreign.uid, 4, 1),
            EdgeRecord(normalized.uid, int(RelationType.EXPLAINS), target.uid, 4, 1),
            EdgeRecord(foreign.uid, int(RelationType.GAME_PROVENANCE), MemoryUid(0, foreign_hash), 4, 1),
            EdgeRecord(target.uid, int(RelationType.GAME_PROVENANCE), MemoryUid(0, target_hash), 4, 1),
        )
        view = _RestartView((foreign, target, normalized), edges)
        with patch.object(restart, "_current_game_hash", return_value=target_hash):
            _build_restart_v854(view)
        self.assertIn(5, view._v815_normalized_action_priors)
        self.assertNotIn(2, view._v815_normalized_action_priors)


class OrderedTransferTests(unittest.TestCase):
    def test_composite_transfer_executes_in_learned_order(self):
        first = _grounded(100, 4, 101, 1)
        second = _grounded(101, 2, 102, 2)
        strategy = MemoryUid(7, 7)
        sequence = OrderedTransferSequence(2.0, strategy, MemoryUid(8, 8), (first, second))
        game = "target"
        game_hash = stable_u64(game, person=b"v8-game")
        view = SimpleNamespace(
            _v854_transfer_active={},
            _v854_ordered_key=((9,), game_hash),
            _v854_ordered=(sequence,),
            _strategy_version=(9,),
            _refresh_strategy_cache=lambda: None,
        )
        first_choice = _ordered_action(view, game, 100, {2, 4})
        self.assertEqual(first_choice[0], 4)
        second_choice = _ordered_action(view, game, 101, {2, 4})
        self.assertEqual(second_choice[0], 2)


class CompositeFormationTests(unittest.TestCase):
    def test_strategy_formation_can_exceed_six_actions(self):
        from v8 import behavior_recovery as behavior

        chain = [_grounded(i, (i % 5) + 1, i + 1, 100 + i, support=8) for i in range(8)]
        outcome = _node(MemoryLevel.M6, MemoryType.OUTCOME, (1, 2, 3), support=8)
        second_outcome = _node(
            MemoryLevel.M6, MemoryType.OUTCOME, (4, 5, 6), support=8
        )
        engine = SimpleNamespace(min_contingency_support=1, _admissible=lambda _row: True)
        with (
            patch.object(
                behavior,
                "causal_m1_ancestors",
                return_value={chain[-1].uid},
            ),
            patch.object(
                behavior,
                "_parent_map",
                wraps=behavior._parent_map,
            ) as parent_map,
        ):
            rows = _adaptive_composites(
                engine,
                tuple(chain) + (outcome, second_outcome),
                (),
                limit=256,
            )
        self.assertTrue(rows)
        self.assertGreaterEqual(max(len(row.parents) - 1 for row in rows), 8)
        parent_map.assert_called_once()


class SessionCreditTests(unittest.TestCase):
    def test_noops_and_disconnected_actions_get_no_success_credit(self):
        view = SimpleNamespace(
            _v854_session_transitions=[
                (10, 9, 10, False),
                (10, 1, 11, True),
                (90, 8, 91, True),
                (11, 2, 12, True),
            ],
            _v815_session_action_priors={},
            _v815_session_trajectory=[],
        )
        _credit_v854(view, success=True, failure=False)
        self.assertEqual(set(view._v815_session_action_priors), {1, 2})


class SimilaritySelectionTests(unittest.TestCase):
    @staticmethod
    def _descriptor(uid, *, exact: bool):
        return NeighborhoodDescriptor(
            uid=uid,
            level=int(MemoryLevel.M3),
            memory_type=int(MemoryType.ROLE),
            incoming_relations=((1, 2),) if exact else ((1, 1),),
            outgoing_relations=((2, 2),) if exact else ((2, 1),),
            neighbor_levels=((1, 2),) if exact else ((1, 1),),
            neighbor_types=((2, 2),) if exact else ((2, 1),),
            dependency_signature=10 if exact else 20,
            enable_block_signature=30 if exact else 40,
            future_option_bucket=1 if exact else -1,
            consequence_bucket=50 if exact else 60,
            context_bucket=0,
            descriptor_version=1,
        )

    def test_exact_provenance_reserves_cross_world_candidate_despite_same_game_mask(self):
        source_uid, same_uid, cross_uid = MemoryUid(1, 1), MemoryUid(1, 2), MemoryUid(9, 9)
        source = SimpleNamespace(uid=source_uid, game_mask=1)
        same = SimpleNamespace(uid=same_uid, game_mask=1)
        cross = SimpleNamespace(uid=cross_uid, game_mask=1)
        source_d = self._descriptor(source_uid, exact=True)
        same_d = self._descriptor(same_uid, exact=True)
        cross_d = self._descriptor(cross_uid, exact=False)
        edges = (
            EdgeRecord(source_uid, int(RelationType.GAME_PROVENANCE), MemoryUid(0, 101), 1, 1),
            EdgeRecord(same_uid, int(RelationType.GAME_PROVENANCE), MemoryUid(0, 101), 1, 1),
            EdgeRecord(cross_uid, int(RelationType.GAME_PROVENANCE), MemoryUid(0, 202), 1, 1),
        )
        sim = BoundedNeighborhoodSimilarity(max_candidates=1, top_results=1, threshold=0.0)
        sim.descriptors = lambda _nodes, _edges: {source_uid: source_d, same_uid: same_d, cross_uid: cross_d}
        rows = _similarity_v854(sim, (source, same, cross), edges)
        self.assertTrue(any(cross_uid in {row.source_uid, row.target_uid} for row in rows))

    def test_structural_prerank_not_uid_order_selects_best_candidate(self):
        source_uid, bad_uid, good_uid = MemoryUid(2, 2), MemoryUid(0, 1), MemoryUid(99, 99)
        nodes = tuple(SimpleNamespace(uid=uid, game_mask=0) for uid in (source_uid, bad_uid, good_uid))
        source_d = self._descriptor(source_uid, exact=True)
        bad_d = self._descriptor(bad_uid, exact=False)
        good_d = self._descriptor(good_uid, exact=True)
        sim = BoundedNeighborhoodSimilarity(max_candidates=1, top_results=1, threshold=0.0)
        sim.descriptors = lambda _nodes, _edges: {source_uid: source_d, bad_uid: bad_d, good_uid: good_d}
        rows = _similarity_v854(sim, nodes, ())
        self.assertTrue(any(good_uid in {row.source_uid, row.target_uid} for row in rows))
        self.assertFalse(any(bad_uid in {row.source_uid, row.target_uid} for row in rows))


if __name__ == "__main__":
    unittest.main()
