from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import v8
from v8 import behavior_recovery as behavior
from v8 import decision_point_sampling_v821 as sampling
from v8 import learning_performance_repair_v824 as v824
from v8 import runtime_repair_v822 as v822
from v8 import sampling_progress_control_v829 as repair
from v8.model import MemoryUid, stable_u64
from v8.publication import PlannedAction


class SamplingProgressControlV829Tests(unittest.TestCase):
    def setUp(self):
        repair._reset_sampling_state_v829()
        repair._CONTROL_STATE.game_id = "ez01"
        repair._CONTROL_STATE.level = 0
        repair._CONTROL_STATE.context = None
        repair._CONTROL_STATE.selection_source = "UNKNOWN"
        repair._CONTROL_STATE.planned_actions = frozenset()

    def test_final_delegates(self):
        self.assertIs(sampling._BASE_ACTOR_WORKER, repair._discovery_actor_v829)
        self.assertIs(v824._BASE_PLAN_CHAIN, repair._plan_chain_v829)
        self.assertIs(behavior._ORIGINAL_SCORE_ACTIONS, repair._score_actions_v829)
        self.assertIs(v822._BASE_ENV_STEP, repair._env_step_v829)
        self.assertIs(v822._BASE_ENV_RESET, repair._env_reset_v829)

    def test_untested_actions_suppress_planner(self):
        base = Mock(return_value=(object(),))
        with patch.object(repair, "_BASE_PLAN_CHAIN", base):
            self.assertEqual(repair._plan_chain_v829(object(), 10, (1, 2, 3, 4)), ())
        base.assert_not_called()

    def test_discovery_exhausts_actions_in_order(self):
        with patch.object(repair, "_BASE_SCORE_ACTIONS", Mock()) as base:
            self.assertEqual(tuple(r.action_id for r in repair._score_actions_v829(object(), 10, (1, 2, 3, 4))), (1,))
            repair._TESTED[("ez01", 0, 10)] = {1}
            self.assertEqual(tuple(r.action_id for r in repair._score_actions_v829(object(), 10, (1, 2, 3, 4))), (2,))
        base.assert_not_called()

    def test_previous_level_action_is_first_next_level_probe(self):
        repair._CONTROL_STATE.level = 1
        repair._TRANSFER_ACTION[("ez01", 1)] = 4
        rows = repair._score_actions_v829(object(), 20, (1, 2, 3, 4))
        self.assertEqual(tuple(r.action_id for r in rows), (4,))
        self.assertEqual(repair._CONTROL_STATE.selection_source, "TRANSFER_PROBE")

    def test_known_progress_action_beats_planner(self):
        repair._PROGRESS_ACTION[("ez01", 0, 10)] = 3
        base = Mock(return_value=(object(),))
        with patch.object(repair, "_BASE_PLAN_CHAIN", base):
            self.assertEqual(repair._plan_chain_v829(object(), 10, (1, 2, 3, 4)), ())
        base.assert_not_called()
        self.assertEqual(tuple(r.action_id for r in repair._score_actions_v829(object(), 10, (1, 2, 3, 4))), (3,))

    def test_cross_context_m7_cannot_control_discovery(self):
        uid, outcome = MemoryUid(1, 2), MemoryUid(3, 4)
        repair._TESTED[("ez01", 0, 10)] = {1, 2}
        view = SimpleNamespace(_strategy_by_context={})
        with patch.object(repair, "_BASE_PLAN_CHAIN", Mock(return_value=(PlannedAction(1, outcome, uid, 1.0),))):
            self.assertEqual(repair._plan_chain_v829(view, 10, (1, 2)), ())

    def test_more_stagnant_plan_cannot_repeat(self):
        context = 10
        first, second, outcome = MemoryUid(1, 1), MemoryUid(2, 2), MemoryUid(3, 3)
        bucket = stable_u64(context, person=b"v8-context")
        view = SimpleNamespace(_strategy_by_context={bucket: (SimpleNamespace(strategy_uid=first), SimpleNamespace(strategy_uid=second))})
        repair._TESTED[("ez01", 0, context)] = {1, 2}
        repair._NO_PROGRESS[("ez01", 0, context, 1)] = 5
        repair._NO_PROGRESS[("ez01", 0, context, 2)] = 0
        plans = (PlannedAction(1, outcome, first, 2.0), PlannedAction(2, outcome, second, 1.0))
        with patch.object(repair, "_BASE_PLAN_CHAIN", Mock(return_value=plans)):
            rows = repair._plan_chain_v829(view, context, (1, 2))
        self.assertEqual(tuple(r.action_id for r in rows), (2,))

    def test_planner_action_is_recorded_without_false_progress(self):
        repair._CONTROL_STATE.context = 10
        repair._CONTROL_STATE.selection_source = "PLANNER"
        env = SimpleNamespace(last_levels_completed=0, last_outcome_state="NOT_FINISHED")
        with patch.object(repair, "_BASE_ENV_STEP", lambda target, action: "frame"):
            self.assertEqual(repair._env_step_v829(env, 2), "frame")
        self.assertIn(2, repair._TESTED[("ez01", 0, 10)])
        self.assertEqual(repair._NO_PROGRESS[("ez01", 0, 10, 2)], 1)
        self.assertNotIn(("ez01", 0, 10), repair._PROGRESS_ACTION)

    def test_level_progress_becomes_replayable_and_transferable(self):
        repair._CONTROL_STATE.context = 10
        repair._CONTROL_STATE.selection_source = "DISCOVERY"
        env = SimpleNamespace(last_levels_completed=0, last_outcome_state="NOT_FINISHED")
        def step(target, action):
            target.last_levels_completed = 1
            return "frame"
        with patch.object(repair, "_BASE_ENV_STEP", step):
            repair._env_step_v829(env, 4)
        self.assertEqual(repair._PROGRESS_ACTION[("ez01", 0, 10)], 4)
        self.assertEqual(repair._TRANSFER_ACTION[("ez01", 1)], 4)
        self.assertEqual(repair._PROGRESS[("ez01", 0, 10, 4)], 1)

    def test_reset_keeps_learned_progress_but_resets_level(self):
        repair._PROGRESS_ACTION[("ez01", 0, 10)] = 1
        repair._CONTROL_STATE.level = 4
        env = SimpleNamespace(last_levels_completed=4)
        def reset(target, *args, **kwargs):
            target.last_levels_completed = 0
            return "reset"
        with patch.object(repair, "_BASE_ENV_RESET", reset):
            self.assertEqual(repair._env_reset_v829(env), "reset")
        self.assertEqual(repair._CONTROL_STATE.level, 0)
        self.assertEqual(repair._PROGRESS_ACTION[("ez01", 0, 10)], 1)


if __name__ == "__main__":
    unittest.main()
