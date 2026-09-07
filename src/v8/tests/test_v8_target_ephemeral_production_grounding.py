from __future__ import annotations

import unittest

from v8.learning_fixes_v088_target_ephemeral_production_grounding_fix import (
    _production_ephemeral_identity,
)
from v8.model import MemoryLevel, MemoryType, MemoryUid, stable_u64


class V8TargetEphemeralProductionGroundingTests(unittest.TestCase):
    def test_identity_matches_production_m1_m2_m3_rules(self):
        identity = _production_ephemeral_identity(
            context_signature=101,
            action_id=3,
            outcome_signature=0x123456,
            next_context_signature=202,
            future_option_delta=1.0,
        )

        self.assertEqual(identity["m1_key"], (101, 3, 0x123456, 202))
        self.assertEqual(identity["m2_key"], (3, 0x3456))

        family_uid = MemoryUid.from_key(
            MemoryLevel.M2,
            MemoryType.FAMILY,
            (3, 0x3456),
        )
        expected_family = stable_u64(
            family_uid.hi,
            family_uid.lo,
            person=b"v8.2-family",
        )
        expected_carrier = stable_u64(101, 202, person=b"v8.2-carrier")

        self.assertEqual(identity["family_uid"], family_uid)
        self.assertEqual(identity["family_token"], expected_family)
        self.assertEqual(identity["carrier_token"], expected_carrier)
        self.assertEqual(identity["future_bucket"], 1)
        self.assertEqual(identity["role_group"], (expected_family, 1))

    def test_distinct_context_transitions_can_form_two_carriers_in_same_role_group(self):
        first = _production_ephemeral_identity(
            context_signature=101,
            action_id=3,
            outcome_signature=0x123456,
            next_context_signature=202,
            future_option_delta=0.0,
        )
        second = _production_ephemeral_identity(
            context_signature=303,
            action_id=3,
            outcome_signature=0x223456,
            next_context_signature=404,
            future_option_delta=0.0,
        )

        self.assertEqual(first["m2_key"], second["m2_key"])
        self.assertEqual(first["role_group"], second["role_group"])
        self.assertNotEqual(first["carrier_token"], second["carrier_token"])


if __name__ == "__main__":
    unittest.main()
