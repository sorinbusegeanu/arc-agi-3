from __future__ import annotations

import unittest
from types import SimpleNamespace

from v8.evaluation import ScientificHypothesisEvaluator
from v8.evidence import EvidenceRecord
from v8.model import MemoryLevel, MemoryType, MemoryUid
from v8.outcome_holdout_v828 import _derive_class, _select_holdout_class
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
        self.class_uid = MemoryUid.from_key(
            MemoryLevel.M6, MemoryType.OUTCOME, self.descriptor
        )

    def _class(self, rows):
        return _derive_class(
            self.descriptor,
            tuple(rows),
            uid=self.class_uid,
            version=1,
            estimator=self.estimator,
        )

    def test_matching_world_holdout_is_excluded_from_training_class(self) -> None:
        rows = (
            _row(1, support=4, variant=1),
            _row(2, support=1, variant=1),
            _row(3, support=1, variant=1),
        )
        by_uid = {row.uid: row for row in rows}
        provenance = {rows[0].uid: {10}, rows[1].uid: {20}, rows[2].uid: {30}}
        training, validation = _select_holdout_class(
            self.estimator, self._class(rows), by_uid, lambda uid: provenance[uid]
        )
        self.assertIsNotNone(validation)
        assert validation is not None
        self.assertTrue(validation.training_persistent)
        self.assertTrue(validation.holdout_consistent)
        self.assertNotIn(validation.target_game_hash, validation.training_games)
        for uid in training.members:
            self.assertNotIn(validation.target_game_hash, provenance[uid])
        for uid in validation.holdout_members:
            self.assertIn(validation.target_game_hash, provenance[uid])

    def test_every_member_carrying_target_world_is_removed_from_training(self) -> None:
        rows = (
            _row(1, support=3, variant=1),
            _row(2, support=1, variant=1),
            _row(3, support=3, variant=1),
            _row(4, support=1, variant=1),
        )
        by_uid = {row.uid: row for row in rows}
        provenance = {
            rows[0].uid: {10, 20},
            rows[1].uid: {10},
            rows[2].uid: {30},
            rows[3].uid: {30},
        }
        training, validation = _select_holdout_class(
            self.estimator, self._class(rows), by_uid, lambda uid: provenance[uid]
        )
        self.assertIsNotNone(validation)
        assert validation is not None
        target = validation.target_game_hash
        self.assertNotIn(target, validation.training_games)
        self.assertTrue(all(target not in provenance[uid] for uid in training.members))
        self.assertTrue(all(target in provenance[uid] for uid in validation.holdout_members))

    def test_contradictory_holdout_fails_existing_m6_criterion(self) -> None:
        rows = (
            _row(1, support=4, variant=1),
            _row(2, support=1, variant=1),
            _row(3, support=1, variant=5),
        )
        by_uid = {row.uid: row for row in rows}
        provenance = {rows[0].uid: {10}, rows[1].uid: {20}, rows[2].uid: {30}}
        training, validation = _select_holdout_class(
            self.estimator, self._class(rows), by_uid, lambda uid: provenance[uid]
        )
        self.assertIsNotNone(validation)
        assert validation is not None
        self.assertTrue(training.persistent)
        self.assertFalse(validation.full_class.persistent)
        self.assertFalse(validation.holdout_consistent)

    def test_insufficient_training_support_emits_no_validation(self) -> None:
        rows = (
            _row(1, support=1, variant=1),
            _row(2, support=1, variant=1),
            _row(3, support=1, variant=1),
        )
        by_uid = {row.uid: row for row in rows}
        provenance = {rows[0].uid: {10}, rows[1].uid: {20}, rows[2].uid: {30}}
        training, validation = _select_holdout_class(
            self.estimator, self._class(rows), by_uid, lambda uid: provenance[uid]
        )
        self.assertIsNone(validation)
        self.assertEqual(training, self._class(rows))

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
            causal_intervention="m6_world_withheld_before_coarse_merge",
            effect_direction=1,
        )
        decisions = ScientificHypothesisEvaluator().evaluate((evidence,))
        h13 = next(row for row in decisions if row.hypothesis_id == "H13")
        self.assertEqual(h13.final_decision, "VALID")
        self.assertEqual(h13.quality_gate, "PASS")


if __name__ == "__main__":
    unittest.main()
