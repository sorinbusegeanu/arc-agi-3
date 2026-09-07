from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

import v8
from v7.environment.encoding import transformation_family_signature
from v8 import learning_fixes_v088 as learning
from v8 import learning_fixes_v088_target_grounding_fix as grounding
from v8.model import MemoryLevel, MemoryType, MemoryUid
from v8.persistent_identity import world_id


class _FakeTargetEnv:
    def __init__(self, **kwargs):
        del kwargs
        self.grid = np.asarray([[0, 0], [0, 0]], dtype=np.int64)
        self.last_levels_completed = 0
        self.last_outcome_polarity = "neutral"

    def observe(self):
        return self.grid.copy()

    def available_actions(self):
        return [7]

    def step(self, action):
        assert int(action) == 7
        self.grid[0, 0] = 0 if int(self.grid[0, 0]) else 1
        self.last_outcome_polarity = "neutral"
        return self.observe()

    def reset(self):
        self.grid[:] = 0
        self.last_outcome_polarity = "neutral"
        return self.observe()


class _NoPlanReadView:
    def planned_action(self, *args, **kwargs):
        del args, kwargs
        return None


class V8TargetLocalTransferGroundingTests(unittest.TestCase):
    def tearDown(self):
        grounding._PENDING_TARGET_TEMPLATES.clear()
        grounding._CAPTURE_GROUNDINGS.clear()

    def test_interaction_grounded_target_action_is_executable_transfer(self):
        before = np.asarray([[0, 0], [0, 0]], dtype=np.int64)
        after = np.asarray([[1, 0], [0, 0]], dtype=np.int64)
        family = int(transformation_family_signature(before, after))
        source_uid = MemoryUid.from_key(MemoryLevel.M4, MemoryType.CONCEPT, (family, 0))
        correspondence_uid = MemoryUid.from_key(
            MemoryLevel.M3, MemoryType.ROLE, (family, 0)
        )
        grounding._PENDING_TARGET_TEMPLATES[world_id("hold01")] = {
            "source_structural_memory_uid": source_uid.hex(),
            "correspondence_uid": correspondence_uid.hex(),
            "source_role_entity": {
                "memory_uid": source_uid.hex(),
                "structural_key": [family, 0],
            },
            "target_role_entity": {
                "correspondence_anchor_uid": correspondence_uid.hex(),
                "structural_key": [family, 0],
                "expected_transformation_family_signatures": [family],
            },
            "expected_transformation_family_signatures": [family],
            "mapping_kind": "explicit_structural_role_to_target_interaction_grounding",
        }

        with patch("v7.environment.arc_adapter.ArcGridEnvironment", _FakeTargetEnv):
            captured = learning._capture_target_probe_state(
                game_id="hold01", env_root=None, seed=3
            )

        execution_evidence = {
            "kind": None,
            "action_ids": [],
            "target_action_evidence": [],
            "correspondence_conditioned_mapping": False,
            "failure_reason": "pending_target_local_structural_grounding",
        }
        diagnostic = {}
        metric, used = learning._probe_policy_v088(
            read_view=_NoPlanReadView(),
            game_id="hold01",
            env_root=None,
            seed=3,
            steps=2,
            required_ancestor=source_uid,
            execution_evidence=execution_evidence,
            diagnostic=diagnostic,
            environment=learning._restore_target_probe_state(captured),
            target_state_capture_id=captured.capture_id,
        )

        self.assertEqual(metric, 0.0)
        self.assertGreater(used, 0)
        self.assertTrue(execution_evidence["correspondence_conditioned_mapping"])
        self.assertEqual(execution_evidence["action_ids"], [7])
        self.assertGreater(
            diagnostic.get("correspondence_conditioned_actions_executed", 0), 0
        )
        applied = diagnostic["applied_transfer_mappings"][0]
        self.assertEqual(applied["derived_target_action"], 7)
        self.assertTrue(applied["target_interaction_evidence_id"])
        self.assertEqual(
            diagnostic.get("target_grounding_policy"), "shared_memory_free_prefix"
        )

    def test_numeric_action_without_target_interaction_grounding_is_rejected(self):
        source_uid = MemoryUid.from_key(MemoryLevel.M4, MemoryType.CONCEPT, (1, 0))
        correspondence_uid = MemoryUid.from_key(MemoryLevel.M3, MemoryType.ROLE, (1, 0))
        action, cursor, row = learning._mapped_evidence_action(
            (
                {
                    "source_structural_memory_uid": source_uid.hex(),
                    "correspondence_uid": correspondence_uid.hex(),
                    "source_role_entity": {"memory_uid": source_uid.hex()},
                    "target_role_entity": {"memory_uid": correspondence_uid.hex()},
                    "mapping_kind": "raw_numeric_replay",
                    "correspondence_conditioned_mapping": True,
                    "derived_target_action": 7,
                },
            ),
            (7,),
            0,
        )
        self.assertIsNone(action)
        self.assertEqual(cursor, 0)
        self.assertIsNone(row)


if __name__ == "__main__":
    unittest.main()
