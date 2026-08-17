from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

import v8
from v8 import adaptive_learning_allocation_v819_performance_fix as perf
from v8 import decision_point_sampling_v821 as sampling
from v8 import learning_control_continuity_v826 as repair
from v8 import learning_performance_repair_v824 as v824
from v8 import runtime_repair_v822 as v822
from v8.publication import LiveReadView


class LearningControlContinuityV826Tests(unittest.TestCase):
    def tearDown(self) -> None:
        if hasattr(v822._PROBE_STATE, "before_plan"):
            v822._PROBE_STATE.before_plan = False

    def test_final_install_authorities(self) -> None:
        self.assertIs(LiveReadView.plan_candidates, repair._plan_candidates_v826)
        self.assertIs(v822._BASE_PLAN_CANDIDATES, repair._plan_candidates_v826)
        self.assertIs(v824._plan_candidates_v824, repair._plan_candidates_v826)
        self.assertIs(
            perf.__dict__["_v823_initial_unsolved_lease_steps"],
            repair.episode_aligned_unsolved_lease_steps_v826,
        )
        self.assertIs(
            v824.unsolved_lease_steps_v824,
            repair.episode_aligned_unsolved_lease_steps_v826,
        )
        self.assertEqual(sampling._VERIFICATION_REPEATS, 1)

    def test_existing_plan_is_not_suppressed_by_pending_discovery_probe(self) -> None:
        sentinel = (object(),)
        view = object()
        v822._PROBE_STATE.before_plan = True
        with patch.object(
            repair,
            "_BASE_PLAN_CANDIDATES",
            Mock(return_value=sentinel),
        ) as base:
            rows = repair._plan_candidates_v826(view, 123, (1, 2, 3, 4))
        self.assertIs(rows, sentinel)
        base.assert_called_once_with(view, 123, (1, 2, 3, 4))
        self.assertTrue(v822._PROBE_STATE.before_plan)

    def test_no_plan_preserves_probe_state_for_sampler_fallback(self) -> None:
        view = object()
        v822._PROBE_STATE.before_plan = True
        with patch.object(
            repair,
            "_BASE_PLAN_CANDIDATES",
            Mock(return_value=()),
        ) as base:
            rows = repair._plan_candidates_v826(view, 456, (1, 2))
        self.assertEqual(rows, ())
        base.assert_called_once_with(view, 456, (1, 2))
        self.assertTrue(v822._PROBE_STATE.before_plan)

    def test_unsolved_budget_is_not_split_at_2048(self) -> None:
        self.assertEqual(
            repair.episode_aligned_unsolved_lease_steps_v826(
                available=360_000,
                base_steps=10_000,
                initial_probe=True,
                worker_count=8,
                game_count=36,
            ),
            10_000,
        )
        self.assertEqual(
            repair.episode_aligned_unsolved_lease_steps_v826(
                available=5_000,
                base_steps=10_000,
                initial_probe=False,
                worker_count=8,
                game_count=36,
            ),
            5_000,
        )

    def test_former_boundary_allows_long_first_solution(self) -> None:
        required_actions = 3_000
        lease_steps = repair.episode_aligned_unsolved_lease_steps_v826(
            available=10_000,
            base_steps=10_000,
            initial_probe=True,
            worker_count=1,
            game_count=1,
        )
        self.assertGreaterEqual(lease_steps, required_actions)
        self.assertGreater(lease_steps, 2_048)


if __name__ == "__main__":
    unittest.main()
