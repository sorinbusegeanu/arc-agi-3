from __future__ import annotations

import os
import unittest

import v8
from v8 import actor as actor_module
from v8 import decision_point_sampling_v821 as sampling
from v8 import runtime_repair_v822 as repair


class _Env:
    def __init__(self) -> None:
        self.reset_count = 0

    def reset(self):
        self.reset_count += 1
        return None


class DecisionPointSamplingV821Tests(unittest.TestCase):
    def test_installed_actor_uses_v822_wrapper_over_v821_discovery_controller(self) -> None:
        self.assertIs(actor_module.actor_worker, repair._actor_worker_v822)
        self.assertIs(repair._BASE_ACTOR_WORKER, sampling._actor_worker_v821)

    def test_new_decision_point_suppresses_planner_until_probe_is_selected(self) -> None:
        sampler = sampling.DecisionPointSampler("ez01", seed=9)
        self.assertIsNone(
            sampler.forced_action(
                level=0,
                context=10,
                actions=(1, 2, 3, 4),
                history=(),
            )
        )
        self.assertTrue(bool(getattr(repair._PROBE_STATE, "before_plan", False)))
        self.assertEqual(repair._plan_candidates_v822(object(), 10, (1, 2, 3, 4)), ())
        self.assertEqual(
            sampler.discovery_action(level=0, context=10, actions=(1, 2, 3, 4), history=()),
            1,
        )
        self.assertFalse(bool(getattr(repair._PROBE_STATE, "before_plan", False)))

    def test_successful_point_is_not_reopened_as_frontier(self) -> None:
        sampler = sampling.DecisionPointSampler("ez01", seed=10)
        solved = sampler.register_point(level=0, context=10, anchor=(), actions=(1, 2), priority=6)
        solved.successful_action = 1
        other = sampler.register_point(level=1, context=20, anchor=(1,), actions=(1, 2), priority=2)
        self.assertIs(sampler._best_frontier(), other)

    def test_ordinary_probe_returns_to_shallow_decision_point_before_random_walk(self) -> None:
        sampler = sampling.DecisionPointSampler("ez01", seed=7)
        action = sampler.discovery_action(
            level=0,
            context=10,
            actions=(1, 2, 3, 4),
            history=(),
        )
        self.assertEqual(action, 1)
        sampler.observe_transition(
            before_level=0,
            before_context=10,
            action=1,
            after_level=0,
            after_context=20,
            after_actions=(1, 2, 3, 4),
            history_after=(1,),
            changed_cells=2,
            terminal_state="NOT_FINISHED",
            terminal_polarity=0,
            level_advanced=False,
            prediction_error=0.0,
            future_delta=0.0,
        )
        self.assertEqual(sampler.pending_reset, ((), (0, 10)))

        env = _Env()
        self.assertTrue(sampler.prepare_step(env))
        self.assertEqual(env.reset_count, 1)
        self.assertIsNone(
            sampler.forced_action(
                level=0,
                context=10,
                actions=(1, 2, 3, 4),
                history=(),
            )
        )
        self.assertEqual(
            sampler.discovery_action(
                level=0,
                context=10,
                actions=(1, 2, 3, 4),
                history=(),
            ),
            2,
        )

    def test_bad_probe_replays_same_anchor_for_next_untested_action(self) -> None:
        sampler = sampling.DecisionPointSampler("ez01", seed=1)
        self.assertEqual(
            sampler.discovery_action(level=0, context=10, actions=(1, 2, 3), history=()),
            1,
        )
        sampler.observe_transition(
            before_level=0,
            before_context=10,
            action=1,
            after_level=0,
            after_context=10,
            after_actions=(1, 2, 3),
            history_after=(1,),
            changed_cells=0,
            terminal_state="NOT_FINISHED",
            terminal_polarity=0,
            level_advanced=False,
            prediction_error=0.0,
            future_delta=0.0,
        )
        self.assertEqual(sampler.pending_reset, ((), (0, 10)))

    def test_prediction_violation_can_continue_at_more_informative_child(self) -> None:
        sampler = sampling.DecisionPointSampler("world", seed=2)
        self.assertEqual(
            sampler.discovery_action(level=0, context=10, actions=(1, 2), history=()),
            1,
        )
        sampler.observe_transition(
            before_level=0,
            before_context=10,
            action=1,
            after_level=0,
            after_context=20,
            after_actions=(1, 2),
            history_after=(1,),
            changed_cells=3,
            terminal_state="NOT_FINISHED",
            terminal_polarity=0,
            level_advanced=False,
            prediction_error=0.9,
            future_delta=0.0,
        )
        self.assertIsNone(sampler.pending_reset)
        self.assertEqual(sampler.points[(0, 20)].priority, 4)
        self.assertEqual(
            sampler.discovery_action(level=0, context=20, actions=(1, 2), history=(1,)),
            1,
        )

    def test_success_is_verified_once_then_transferred_to_next_level(self) -> None:
        sampler = sampling.DecisionPointSampler("ez01", seed=3)
        self.assertEqual(
            sampler.discovery_action(level=0, context=10, actions=(1, 2, 3, 4), history=()),
            1,
        )
        sampler.observe_transition(
            before_level=0,
            before_context=10,
            action=1,
            after_level=1,
            after_context=20,
            after_actions=(1, 2, 3, 4),
            history_after=(1,),
            changed_cells=1,
            terminal_state="NOT_FINISHED",
            terminal_polarity=1,
            level_advanced=True,
            prediction_error=1.0,
            future_delta=0.0,
        )
        self.assertIsNotNone(sampler.verification)
        self.assertEqual(sampler.verification.remaining, 1)

        env = _Env()
        self.assertTrue(sampler.prepare_step(env))
        action = sampler.forced_action(
            level=0,
            context=10,
            actions=(1, 2, 3, 4),
            history=(),
        )
        self.assertEqual(action, 1)
        sampler.observe_transition(
            before_level=0,
            before_context=10,
            action=1,
            after_level=1,
            after_context=20,
            after_actions=(1, 2, 3, 4),
            history_after=(1,),
            changed_cells=1,
            terminal_state="NOT_FINISHED",
            terminal_polarity=1,
            level_advanced=True,
            prediction_error=0.0,
            future_delta=0.0,
        )
        self.assertIsNone(sampler.verification)

        self.assertEqual(sampler.transfer_action, 1)
        self.assertEqual(sampler.transfer_from_level, 0)
        self.assertEqual(
            sampler.discovery_action(
                level=1,
                context=20,
                actions=(1, 2, 3, 4),
                history=(1,),
            ),
            1,
        )
        self.assertEqual(sampler.current.kind, "TRANSFER")

    def test_replay_anchor_actions_are_forced_before_probe(self) -> None:
        sampler = sampling.DecisionPointSampler("world", seed=4)
        point = sampler.register_point(
            level=2,
            context=99,
            anchor=(4, 5),
            actions=(1, 2),
            priority=3,
        )
        sampler._schedule_point(point)
        env = _Env()
        self.assertTrue(sampler.prepare_step(env))
        self.assertEqual(
            sampler.forced_action(level=0, context=11, actions=(4, 7), history=()),
            4,
        )
        sampler.observe_transition(
            before_level=0,
            before_context=11,
            action=4,
            after_level=1,
            after_context=22,
            after_actions=(5, 7),
            history_after=(4,),
            changed_cells=1,
            terminal_state="NOT_FINISHED",
            terminal_polarity=1,
            level_advanced=True,
            prediction_error=0.0,
            future_delta=0.0,
        )
        self.assertEqual(
            sampler.forced_action(level=1, context=22, actions=(5, 7), history=(4,)),
            5,
        )

    def test_frontier_is_bounded(self) -> None:
        sampler = sampling.DecisionPointSampler("world", seed=5, max_points=8)
        for index in range(40):
            sampler.register_point(
                level=0,
                context=index,
                anchor=tuple(range(index % 4)),
                actions=(1, 2),
                priority=1,
            )
        self.assertLessEqual(len(sampler.points), 8)

    def test_non_discovery_sampling_modes_delegate_to_existing_actor(self) -> None:
        prior = os.environ.get(sampling._SAMPLING_MODE_ENV)
        try:
            for mode in ("VERIFY", "ALTERNATIVE", "TRANSFER"):
                os.environ[sampling._SAMPLING_MODE_ENV] = mode
                self.assertFalse(sampling._decision_mode_enabled())
            os.environ[sampling._SAMPLING_MODE_ENV] = "DISCOVERY"
            self.assertTrue(sampling._decision_mode_enabled())
        finally:
            if prior is None:
                os.environ.pop(sampling._SAMPLING_MODE_ENV, None)
            else:
                os.environ[sampling._SAMPLING_MODE_ENV] = prior


if __name__ == "__main__":
    unittest.main()
