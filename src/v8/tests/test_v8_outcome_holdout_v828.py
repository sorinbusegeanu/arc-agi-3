from __future__ import annotations

import unittest
from types import SimpleNamespace

from v8.evaluation import ScientificHypothesisEvaluator
from v8.evidence import EvidenceRecord
from v8.model import MemoryLevel, MemoryType, MemoryUid
from v8.outcome_holdout_v828 import _derive_class, _select_occurrence_holdout_class
from v8.outcomes import OutcomeEquivalenceEstimator


def _row(uid_lo: int, *, support: int, variant: int):
    return SimpleNamespace(
        uid=MemoryUid(0, uid_lo),
        support_count=support,
        key_parts=(1, 2, variant),
    )


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
            causal_intervention="m6_m0_world_occurrence_holdout",
            effect_direction=1,
        )
        decisions = ScientificHypothesisEvaluator().evaluate((evidence,))
        h13 = next(row for row in decisions if row.hypothesis_id == "H13")
        self.assertEqual(h13.final_decision, "VALID")
        self.assertEqual(h13.quality_gate, "PASS")


if __name__ == "__main__":
    unittest.main()
