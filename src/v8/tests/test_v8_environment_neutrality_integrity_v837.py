from __future__ import annotations

import unittest
from types import SimpleNamespace

import v8  # noqa: F401
from v8 import sampling_transfer_v833 as transfer
from v8 import trajectory_optimizer_v814 as optimizer
from v8.environment_contract import BoundaryEvent, BoundaryScope


class EnvironmentNeutralityIntegrityV837Tests(unittest.TestCase):
    def test_legacy_win_and_generic_positive_episode_share_target_identity(self):
        anchor = optimizer.ReplayAnchor("env", 0, (), None)
        legacy = optimizer.TrajectoryTarget(5, "WIN")
        generic = optimizer.TrajectoryTarget(
            0,
            "BOUNDARY",
            BoundaryScope.EPISODE.value,
            +1,
            False,
        )
        negative = optimizer.TrajectoryTarget(
            0,
            "BOUNDARY",
            BoundaryScope.EPISODE.value,
            -1,
            False,
        )
        self.assertEqual(optimizer._anchor_hash(anchor, legacy), optimizer._anchor_hash(anchor, generic))
        self.assertNotEqual(optimizer._anchor_hash(anchor, generic), optimizer._anchor_hash(anchor, negative))

    def test_terminal_generic_boundary_does_not_register_destination(self):
        calls = []
        sampler = SimpleNamespace(
            base=SimpleNamespace(
                register_point=lambda **kwargs: calls.append(kwargs),
            )
        )
        transfer._register_destination(
            sampler,
            kwargs={
                "after_level": 0,
                "after_context": 4,
                "after_actions": (1, 2),
                "history_after": (1,),
                "structural_changed": True,
                "boundary_event": BoundaryEvent(BoundaryScope.EPISODE, -1, False),
            },
            priority=4,
        )
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
