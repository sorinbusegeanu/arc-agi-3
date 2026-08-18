from __future__ import annotations

import unittest

import v8
from v8.arena import EdgeRecord, NodeRecord
from v8.evaluation import ScientificHypothesisEvaluator
from v8.evidence import EvidenceRecord
from v8.hypothesis_validation_v842 import (
    _BEHAVIORAL_TRANSFER_INTERVENTION,
    _H12_COMPARISON_INTERVENTION,
    _H13_HOLDOUT_INTERVENTION,
    _OLD_PROXY_TRANSFER_INTERVENTION,
    _auto_outcome_holdout_v842,
    _emit_strategy_efficiency_comparisons,
    _record_strategy_statistics_v842,
)
from v8.model import CognitiveState, MemoryLevel, MemoryType, MemoryUid, RelationType, ValidationState
from v8.outcomes import OutcomeEquivalenceEstimator
from v8.similarity import BoundedNeighborhoodSimilarity
from v8.strategies import StrategyEstimator


def node(
    uid: MemoryUid,
    *,
    level: MemoryLevel,
    memory_type: MemoryType,
    key_parts: tuple[int, ...],
    support: int = 4,
    watermark: int = 1,
    game_mask: int = 1,
    state: CognitiveState = CognitiveState.ACTIVE,
    attempts: float = 0.0,
    successes: float = 0.0,
    cost: float = 0.0,
) -> NodeRecord:
    return NodeRecord(
        uid,
        1,
        int(level),
        int(memory_type),
        key_parts,
        support,
        1.0,
        0.0,
        1.0,
        0.0,
        1.0,
        0.0,
        1.0,
        watermark,
        game_mask,
        int(state),
        int(ValidationState.STRUCTURAL),
        successes,
        cost,
        attempts,
    )


def evidence(
    evidence_id: str,
    kind: str,
    *,
    target: int = 0,
    provenance: tuple[int, ...] = (),
    causal: str = "",
    effect: int = 1,
) -> EvidenceRecord:
    return EvidenceRecord.for_uid(
        evidence_id,
        MemoryUid(10, len(evidence_id)),
        evidence_kind=kind,
        watermark=10,
        raw_value=1.0,
        normalized_value=1.0,
        developmental_stage=1,
        validation_state=int(ValidationState.VALIDATED),
        target_game_hash=target,
        provenance_games=provenance,
        causal_intervention=causal,
        effect_direction=effect,
    )


class DependencyEvaluationTests(unittest.TestCase):
    def test_forward_h12_dependency_resolves_after_h13(self) -> None:
        rows = [
            evidence("h01a", "contingency_recurrence"),
            evidence("h01b", "contingency_recurrence"),
            evidence("h03", "family_compression"),
            evidence("h04", "carrier_emergence"),
            evidence("h05", "role_emergence"),
            evidence("h06", "transfer_trial_pass", target=2, provenance=(1,), causal="behavioral", effect=1),
            evidence("h07", "concept_transfer_pass", target=2, provenance=(1,), causal="behavioral", effect=1),
            evidence("h08", "world_model_component"),
            evidence("h13", "outcome_consistency_holdout", target=3, provenance=(1, 2), effect=1),
            evidence("h12", "strategy_efficiency", causal=_H12_COMPARISON_INTERVENTION, effect=1),
        ]
        decisions = ScientificHypothesisEvaluator().evaluate(rows)
        status = {row.hypothesis_id: row.final_decision for row in decisions}
        gates = {row.hypothesis_id: row.dependency_gate for row in decisions}
        self.assertEqual(status["H13"], "VALID")
        self.assertEqual(status["H12"], "VALID")
        self.assertEqual(gates["H12"], "PASS")

    def test_old_structural_proxy_transfer_cannot_validate_h06(self) -> None:
        rows = [
            evidence("h01a", "contingency_recurrence"),
            evidence("h01b", "contingency_recurrence"),
            evidence("h03", "family_compression"),
            evidence("h04", "carrier_emergence"),
            evidence("h05", "role_emergence"),
            evidence(
                "transfer_trial_pass:legacy",
                "transfer_trial_pass",
                target=2,
                provenance=(1,),
                causal=_OLD_PROXY_TRANSFER_INTERVENTION,
                effect=1,
            ),
        ]
        status = {row.hypothesis_id: row.final_decision for row in ScientificHypothesisEvaluator().evaluate(rows)}
        self.assertNotEqual(status["H06"], "VALID")


class OutcomeHoldoutTests(unittest.TestCase):
    def test_h13_rebuild_excludes_target_game(self) -> None:
        formation_uid = MemoryUid(20, 1)
        target_uid = MemoryUid(20, 2)
        formation = node(
            formation_uid,
            level=MemoryLevel.M6,
            memory_type=MemoryType.OUTCOME,
            key_parts=(1, 7, 0),
            support=6,
            game_mask=1,
        )
        target = node(
            target_uid,
            level=MemoryLevel.M6,
            memory_type=MemoryType.OUTCOME,
            key_parts=(1, 7, 1),
            support=2,
            game_mask=2,
        )
        edges = (
            EdgeRecord(formation_uid, int(RelationType.GAME_PROVENANCE), MemoryUid(0, 101), 1, 1),
            EdgeRecord(target_uid, int(RelationType.GAME_PROVENANCE), MemoryUid(0, 202), 1, 1),
        )

        class Peer:
            candidate_budget = 16
            outcomes = OutcomeEquivalenceEstimator()

            def __init__(self):
                self.rows = []

            def _fresh(self, *args, **kwargs):
                return True

            def _append_evidence(self, kind, row, value, **kwargs):
                self.rows.append((kind, row, value, kwargs))

        peer = Peer()
        _auto_outcome_holdout_v842(peer, (formation, target), edges)
        self.assertEqual(len(peer.rows), 1)
        kind, _row, _value, kwargs = peer.rows[0]
        self.assertEqual(kind, "outcome_consistency_holdout")
        self.assertEqual(kwargs["target_game_hash"], 202)
        self.assertEqual(kwargs["provenance_games"], (101,))
        self.assertEqual(kwargs["causal_intervention"], _H13_HOLDOUT_INTERVENTION)


class StrategyComparisonTests(unittest.TestCase):
    def test_h12_evidence_is_pairwise_outcome_comparable(self) -> None:
        outcome = MemoryUid(30, 1)
        first = node(
            MemoryUid(31, 1),
            level=MemoryLevel.M7,
            memory_type=MemoryType.STRATEGY,
            key_parts=(1, outcome.hi, outcome.lo, 9),
            attempts=10,
            successes=9,
            cost=10,
        )
        second = node(
            MemoryUid(31, 2),
            level=MemoryLevel.M7,
            memory_type=MemoryType.STRATEGY,
            key_parts=(2, outcome.hi, outcome.lo, 9),
            attempts=10,
            successes=5,
            cost=20,
        )

        class View:
            def node_records(self):
                return (first, second)

            def source_games(self, uid):
                return frozenset({1})

        class Peer:
            read_view = View()
            strategies = StrategyEstimator()

            def __init__(self):
                self.rows = []

            def _fresh(self, *args, **kwargs):
                return True

            def _append_evidence(self, kind, row, value, **kwargs):
                self.rows.append((kind, row, value, kwargs))

        peer = Peer()
        _emit_strategy_efficiency_comparisons(peer)
        self.assertEqual(len(peer.rows), 1)
        kind, _row, value, kwargs = peer.rows[0]
        self.assertEqual(kind, "strategy_efficiency")
        self.assertGreater(value, 0.0)
        self.assertEqual(kwargs["effect_direction"], 1)
        self.assertEqual(kwargs["causal_intervention"], _H12_COMPARISON_INTERVENTION)


class BehavioralTransferTests(unittest.TestCase):
    def test_special_m3_stat_becomes_behavioral_heldout_transfer_trial(self) -> None:
        uid = MemoryUid(40, 1)
        row = node(
            uid,
            level=MemoryLevel.M3,
            memory_type=MemoryType.ROLE,
            key_parts=(1, 1),
        )

        class View:
            def node_records(self, *, level=None):
                if level == MemoryLevel.M7:
                    return ()
                if level == MemoryLevel.M3:
                    return (row,)
                return ()

            def source_games(self, value):
                return frozenset({101})

        class Peer:
            read_view = View()

            def __init__(self):
                self.trials = []

            def record_transfer_trial(self, uid, **kwargs):
                self.trials.append((uid, kwargs))

        peer = Peer()
        accepted = _record_strategy_statistics_v842(
            peer,
            uid,
            attempts=4,
            successes=3,
            cost=1.0,
            source_game_hash=202,
        )
        self.assertTrue(accepted)
        self.assertEqual(len(peer.trials), 1)
        _uid, kwargs = peer.trials[0]
        self.assertEqual(kwargs["metric_on"], 0.75)
        self.assertEqual(kwargs["metric_off"], 0.25)
        self.assertEqual(kwargs["formation_games"], (101,))
        self.assertEqual(kwargs["intervention"], _BEHAVIORAL_TRANSFER_INTERVENTION)


class SimilarityBreadthTests(unittest.TestCase):
    def test_similarity_reserves_cross_game_result(self) -> None:
        rows = (
            node(MemoryUid(50, 1), level=MemoryLevel.M3, memory_type=MemoryType.ROLE, key_parts=(1, 0), game_mask=1),
            node(MemoryUid(50, 2), level=MemoryLevel.M3, memory_type=MemoryType.ROLE, key_parts=(2, 0), game_mask=1),
            node(MemoryUid(50, 3), level=MemoryLevel.M3, memory_type=MemoryType.ROLE, key_parts=(3, 0), game_mask=2),
        )
        estimator = BoundedNeighborhoodSimilarity(max_candidates=1, top_results=1, threshold=0.0)
        result = estimator.evaluate(rows, ())
        masks = {row.uid: row.game_mask for row in rows}
        self.assertTrue(any(masks[item.source_uid] != masks[item.target_uid] for item in result))


if __name__ == "__main__":
    unittest.main()
