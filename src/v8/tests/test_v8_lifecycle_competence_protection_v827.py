from __future__ import annotations

import unittest
from types import SimpleNamespace

import v8
from v8.lifecycle import LifecycleController
from v8.model import CognitiveState, MemoryUid, ValidationState


class LifecycleCompetenceProtectionV827Tests(unittest.TestCase):
    def test_validated_competence_dependency_reactivates_quarantined_memory(self) -> None:
        uid = MemoryUid(7, 9)
        controller = LifecycleController()
        controller._v827_protected_competence_uids = frozenset({uid})
        controller._low_windows[uid] = 6
        row = SimpleNamespace(
            uid=uid,
            cognitive_state=int(CognitiveState.QUARANTINED),
            validation_state=int(ValidationState.TESTED),
        )

        decision = controller.decide(row)

        self.assertIsNotNone(decision)
        self.assertEqual(decision.cognitive_state, int(CognitiveState.REACTIVATED))
        self.assertNotIn(uid, controller._low_windows)

    def test_failed_validation_is_not_protected_from_lifecycle(self) -> None:
        uid = MemoryUid(7, 10)
        controller = LifecycleController()
        controller._v827_protected_competence_uids = frozenset({uid})
        row = SimpleNamespace(
            uid=uid,
            level=7,
            memory_type=700,
            key_parts=(1, 2, 3, 4),
            support_count=0,
            significance=0.0,
            prediction_error=0.0,
            learning_value=0.0,
            transfer_potential=0.0,
            explanatory_potential=0.0,
            future_option_delta=0.0,
            novelty=0.0,
            game_evidence_count=0,
            cognitive_state=int(CognitiveState.ACTIVE),
            validation_state=int(ValidationState.FAILED),
            strategy_reliability=0.0,
        )

        # The protected fast path must not reactivate or exempt failed evidence.
        self.assertNotEqual(
            getattr(controller.decide(row), "reason", ""),
            "reactivated by validated competence dependency",
        )


if __name__ == "__main__":
    unittest.main()
