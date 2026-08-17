from __future__ import annotations

import unittest
from types import SimpleNamespace

import v8
from v8.arena import NodeRecord
from v8.lifecycle import LifecycleController
from v8.model import CognitiveState, MemoryLevel, MemoryType, MemoryUid, ValidationState


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

    def test_validated_competence_dependency_cannot_finalize_retirement(self) -> None:
        uid = MemoryUid(7, 11)
        controller = LifecycleController()
        controller._v827_protected_competence_uids = frozenset({uid})
        row = SimpleNamespace(
            uid=uid,
            cognitive_state=int(CognitiveState.RETIRE_PENDING),
            validation_state=int(ValidationState.TESTED),
        )

        self.assertIsNone(
            controller.finalize_retirement(row, protected_by_dependencies=False)
        )

    def test_failed_validation_is_not_exempted_by_competence_dependency(self) -> None:
        uid = MemoryUid.from_key(MemoryLevel.M7, MemoryType.STRATEGY, (1, 2, 3, 4))
        controller = LifecycleController()
        controller._v827_protected_competence_uids = frozenset({uid})
        row = NodeRecord(
            uid,
            uid.hi ^ uid.lo,
            int(MemoryLevel.M7),
            int(MemoryType.STRATEGY),
            (1, 2, 3, 4),
            0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            1,
            0,
            int(CognitiveState.ACTIVE),
            int(ValidationState.FAILED),
            0.0,
            0.0,
            0.0,
        )

        decision = controller.decide(row)

        self.assertNotEqual(
            getattr(decision, "reason", ""),
            "reactivated by validated competence dependency",
        )


if __name__ == "__main__":
    unittest.main()
