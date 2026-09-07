from __future__ import annotations

import unittest

from v8 import learning_fixes_v088 as learning
from v8.learning_fixes_v088_target_mapping_lifetime_fix import (
    install_target_mapping_lifetime_fix,
)


class V8TargetMappingLifetimeTests(unittest.TestCase):
    def test_grounded_action_is_consumed_once(self):
        install_target_mapping_lifetime_fix()
        evidence = (
            {
                "correspondence_conditioned_mapping": True,
                "source_structural_memory_uid": "source",
                "correspondence_uid": "correspondence",
                "source_role_entity": {},
                "target_role_entity": {},
                "mapping_kind": "explicit_structural_role_to_target_interaction_grounding",
                "target_interaction_evidence_id": "evidence",
                "target_grounding_context_signature": 123,
                "derived_target_action": 4,
            },
        )

        action, cursor, applied = learning._mapped_evidence_action(evidence, (1, 4), 0)
        self.assertEqual(action, 4)
        self.assertEqual(cursor, 1)
        self.assertIsNotNone(applied)

        action, cursor, applied = learning._mapped_evidence_action(evidence, (1, 4), cursor)
        self.assertIsNone(action)
        self.assertEqual(cursor, 1)
        self.assertIsNone(applied)

    def test_mapping_requires_explicit_grounding_context(self):
        install_target_mapping_lifetime_fix()
        evidence = (
            {
                "correspondence_conditioned_mapping": True,
                "source_structural_memory_uid": "source",
                "correspondence_uid": "correspondence",
                "source_role_entity": {},
                "target_role_entity": {},
                "mapping_kind": "explicit_structural_role_to_target_interaction_grounding",
                "target_interaction_evidence_id": "evidence",
                "derived_target_action": 4,
            },
        )

        action, cursor, applied = learning._mapped_evidence_action(evidence, (4,), 0)
        self.assertIsNone(action)
        self.assertEqual(cursor, 0)
        self.assertIsNone(applied)


if __name__ == "__main__":
    unittest.main()
