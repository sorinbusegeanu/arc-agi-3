from __future__ import annotations

import unittest

from v8.learning_fixes_v088_immediate_structural_transfer_fix import (
    _BRANCH_STRUCTURAL,
    _structural_alignment,
    install_immediate_structural_transfer_fix,
)
from v8.model import MemoryLevel, MemoryType, MemoryUid
from v8.transfer import TransferValidator


class V8ImmediateStructuralTransferTests(unittest.TestCase):
    def test_alignment_uses_production_family_transformation_and_future_structure(self):
        expected = {
            "m2_key": (3, 44),
            "family_token": 99,
            "future_bucket": 1,
            "transformation_family_signature": 55,
        }
        matching = {
            "m2_key": (3, 44),
            "future_bucket": 1,
            "transformation_family_signature": 55,
        }
        control = {
            "m2_key": (2, 10),
            "future_bucket": 1,
            "transformation_family_signature": 77,
        }
        self.assertEqual(_structural_alignment(matching, expected), 1.0)
        self.assertAlmostEqual(_structural_alignment(control, expected), 0.2)

    def test_immediate_gain_can_validate_when_long_horizon_is_sparse(self):
        install_immediate_structural_transfer_fix()
        target = 123456
        _BRANCH_STRUCTURAL[target] = {
            "on": {"alignment": 1.0},
            "off": {"alignment": 0.2},
        }
        validator = TransferValidator(effect_threshold=0.0)
        uid = MemoryUid.from_key(MemoryLevel.M4, MemoryType.CONCEPT, (7, 0))
        trial = validator.record_trial(
            uid,
            target_game_hash=target,
            metric_on=0.0,
            metric_off=0.0,
            formation_games=(999,),
            intervention="matched_arc_target_memory_vs_memory_free",
        )
        self.assertAlmostEqual(trial.effect, 0.8)
        self.assertTrue(trial.passed)
        self.assertEqual(trial.metric_on, 0.0)
        self.assertEqual(trial.metric_off, 0.0)

    def test_structurally_neutral_sparse_trial_does_not_pass(self):
        install_immediate_structural_transfer_fix()
        target = 654321
        _BRANCH_STRUCTURAL[target] = {
            "on": {"alignment": 1.0},
            "off": {"alignment": 1.0},
        }
        validator = TransferValidator(effect_threshold=0.0)
        uid = MemoryUid.from_key(MemoryLevel.M4, MemoryType.CONCEPT, (8, 0))
        trial = validator.record_trial(
            uid,
            target_game_hash=target,
            metric_on=0.0,
            metric_off=0.0,
            formation_games=(999,),
            intervention="matched_arc_target_memory_vs_memory_free",
        )
        self.assertEqual(trial.effect, 0.0)
        self.assertFalse(trial.passed)


if __name__ == "__main__":
    unittest.main()
