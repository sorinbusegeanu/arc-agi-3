from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import v8  # noqa: F401 - install production runtime stack
from v8 import adaptive_memory_transfer_integrity_v856 as v856
from v8 import adaptive_memory_transfer_integrity_v856_fixups as fixups
from v8 import behavior_recovery as behavior
from v8 import click_exploration_v848 as v848
from v8 import sampling_persistence_v832 as persistence
from v8.model import MemoryUid
from v8.publication import PlannedAction


class AdaptiveMemoryTransferIntegrityV856FixupTests(unittest.TestCase):
    def setUp(self) -> None:
        self.prior_view = behavior._CURRENT_ACTOR_VIEW

    def tearDown(self) -> None:
        behavior._CURRENT_ACTOR_VIEW = self.prior_view

    def test_historical_forced_action_authority_is_preserved(self) -> None:
        self.assertIs(persistence._BASE_FORCED_ACTION, v848._sampler_forced_action_v848)

    def test_post_observation_clears_consumed_plan_without_hiding_it_from_credit(self) -> None:
        plan = PlannedAction(2, MemoryUid(6, 1), MemoryUid(7, 1), 1.0, False)
        view = SimpleNamespace(_behavior_last_plans=(plan,))
        behavior._CURRENT_ACTOR_VIEW = view
        seen = []

        def observed(**kwargs):
            del kwargs
            seen.extend(view._behavior_last_plans)
            return (MemoryUid.zero(), MemoryUid.zero())

        with patch.object(fixups, "_BASE_OBSERVED", side_effect=observed):
            result = fixups._observed_outcomes_clear_v856(terminal_polarity=0)

        self.assertEqual(result, (MemoryUid.zero(), MemoryUid.zero()))
        self.assertEqual(seen, [plan])
        self.assertEqual(view._behavior_last_plans, ())

    def test_publish_plans_is_safe_for_historical_plain_object_callers(self) -> None:
        plan = PlannedAction(2, MemoryUid(6, 1), MemoryUid(7, 1), 1.0, False)
        self.assertEqual(fixups._publish_plans_safe(object(), (plan,)), (plan,))

    def test_transfer_cut_cache_avoids_repeat_raw_edge_scan(self) -> None:
        view = SimpleNamespace(_strategy_version=(1, 2), _v856_transfer_cut_version=(1, 2))
        with patch.object(v856, "_stable_edge_rows") as scan:
            v856._augment_actor_transfer_cut(view)
        scan.assert_not_called()


if __name__ == "__main__":
    unittest.main()
