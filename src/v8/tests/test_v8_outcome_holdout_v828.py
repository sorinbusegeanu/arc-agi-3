from __future__ import annotations

import unittest
from types import SimpleNamespace

from v8.evaluation import ScientificHypothesisEvaluator
from v8.evidence import EvidenceRecord
from v8.model import MemoryLevel, MemoryType, MemoryUid, RelationType
from v8.outcome_holdout_v828 import (
    _derive_class,
    _lineage_occurrences_by_world,
    _select_occurrence_holdout_class,
)
from v8.outcomes import OutcomeEquivalenceEstimator


def _row(uid_lo: int, *, support: int, variant: int):
    return SimpleNamespace(
        uid=MemoryUid(0, uid_lo),
        support_count=support,
        key_parts=(1, 2, variant),
    )


class _ReadView:
    def __init__(self, edges):
        self._edges = tuple(edges)

    def edge_records(self):
        return self._edges


class OutcomeHoldoutV828Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.estimator = OutcomeEquivalenceEstimator()
        self.descriptor = (1, 2)
        self.class_uid = MemoryUid.from_key(MemoryLevel.M6, MemoryType.OUTCOME, self.descriptor)

    def _class(self, rows):
        return _derive_class(
            self.descriptor,
            tuple(rows),
            uid=self.class_uid,
            version=1,
            estimator=self.estimator,
        )

    def test_lineage_provenance_is_counted_without_requiring_m0(self) -> None:
        root = MemoryUid(1, 1)
        m5 = MemoryUid(1, 2)
        m3 = MemoryUid(1, 3)
        world = 77
        edges = (
            SimpleNamespace(source_uid=root, target_uid=m5, relation_type=int(RelationType.EXPLAINS)),
            SimpleNamespace(source_uid=m5, target_uid=m3, relation_type=int(RelationType.EXPLAINS)),
            SimpleNamespace(source_uid=m3, target_uid=MemoryUid(0, world), relation_type=int(RelationType.GAME_PROVENANCE)),
        )
        occurrences = _lineage_occurrences_by_world(_ReadView(edges), (root,))
        self.assertEqual(occurrences[root], {world: 1})

    def test_multiple_direct_provenance_nodes_on_lineage_count_as_occurrences(self) -> None:
        root = MemoryUid(2, 1)
        left = MemoryUid(2, 2)
        right = MemoryUid(2, 3)
        world = 88
        edges = (
            SimpleNamespace(source_uid=root, target_uid=left, relation_type=int(RelationType.EXPLAINS)),
            SimpleNamespace(source_uid=root, target_uid=right, relation_type=int(RelationType.DEPENDS_ON)),
            SimpleNamespace(source_uid=left, target_uid=MemoryUid(0, world), relation_type=int(RelationType.GAME_PROVENANCE)),
            SimpleNamespace(source_uid=right, target_uid=MemoryUid(0, world), relation_type=int(RelationType.GAME_PROVENANCE)),
        )
        occurrences = _lineage_occurrences_by_world(_ReadView(edges), (root,))
        self.assertEqual(occurrences[root], {world: 2})

    def test_target_world_occurrences_are_excluded_before_validation_formation(self) -> None:
        rows = (_row(1, support=6, variant=1), _row(2, support=6, variant=1))
        by_uid = {row.uid: row for row in rows}
        occurrences = {
            rows[0].uid: {10: 3, 30: 3},
            rows[1].uid: {20: 3, 30: 3},
        }
        validation = _select_occurrence_holdout_class(
            self.estimator, self._class(rows), by_uid, occurrences
        )
        self.assertIsNotNone(validation)
        assert validation is not None
        self.assertNotIn(validation.target_game_hash, validation.training_games)
        self.assertEqual(validation.holdout_games, (validation.target_game_hash,))
        self.assertTrue(validation.training_persistent)
        self.assertTrue(validation.holdout_consistent)
        self.assertGreater(validation.training_occurrences, 0)
        self.assertGreater(validation.holdout_occurrences, 0)

    def test_aggregated_fine_m6_provenance_no_longer_prevents_world_holdout(self) -> None:
        rows = (_row(1, support=8, variant=1), _row(2, support=8, variant=1))
        by_uid = {row.uid: row for row in rows}
        occurrences = {
            rows[0].uid: {10: 4, 20: 4},
            rows[1].uid: {10: 4, 30: 4},
        }
        validation = _select_occurrence_holdout_class(
            self.estimator, self._class(rows), by_uid, occurrences
        )
        self.assertIsNotNone(validation)
        assert validation is not None
        self.assertNotIn(validation.target_game_hash, validation.training_games)
        self.assertGreaterEqual(len(validation.training_members), 2)

    def test_contradictory_holdout_fails_existing_m6_criterion(self) -> None:
        rows = (_row(1, support=8, variant=1), _row(2, support=8, variant=5))
        by_uid = {row.uid: row for row in rows}
        occurrences = {
            rows[0].uid: {10: 4, 20: 4},
            rows[1].uid: {20: 4, 30: 4},
        }
        validation = _select_occurrence_holdout_class(
            self.estimator, self._class(rows), by_uid, occurrences
        )
        if validation is not None:
            self.assertTrue(validation.training_persistent)
            self.assertFalse(validation.holdout_consistent)

    def test_insufficient_disjoint_training_support_emits_no_validation(self) -> None:
        rows = (_row(1, support=2, variant=1), _row(2, support=2, variant=1))
        by_uid = {row.uid: row for row in rows}
        occurrences = {rows[0].uid: {10: 1}, rows[1].uid: {10: 1}}
        validation = _select_occurrence_holdout_class(
            self.estimator, self._class(rows), by_uid, occurrences
        )
        self.assertIsNone(validation)

    def test_h13_accepts_distinct_heldout_target(self) -> None:
        evidence = EvidenceRecord.for_uid(
            "h13:test",
            self.class_uid,
            evidence_kind="outcome_consistency_holdout",
            watermark=10,
            raw_value=0.8,
            normalized_value=0.8,
            developmental_stage=int(MemoryLevel.M6),
            validation_state=1,
            target_game_hash=30,
            provenance_games=(10, 20),
            causal_intervention="m6_lineage_world_occurrence_holdout",
            effect_direction=1,
        )
        decisions = ScientificHypothesisEvaluator().evaluate((evidence,))
        h13 = next(row for row in decisions if row.hypothesis_id == "H13")
        self.assertEqual(h13.final_decision, "VALID")
        self.assertEqual(h13.quality_gate, "PASS")


if __name__ == "__main__":
    unittest.main()
