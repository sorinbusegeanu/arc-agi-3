from __future__ import annotations

import unittest

import v8  # installs the current runtime stack including v8.59
from v7.game_sets import resolve_game_selector
from v8 import actor as actor_module
from v8.cli import _actor_jobs
from v8.mixed_environment_v859 import (
    ARC_GAME_IDS,
    GENERIC_GAME_IDS,
    MIX_GAME_IDS,
    MIX_SPECS,
    _choose_action,
    is_generic_game,
    is_mix_selector,
    run_generic_actor_job,
)


class _Stop:
    def is_set(self):
        return False


class _Score:
    def __init__(self, action_id, support_count=0, score=0.0):
        self.action_id = int(action_id)
        self.support_count = int(support_count)
        self.score = float(score)


class _Plan:
    def __init__(self, action_id):
        self.action_id = int(action_id)


class _View:
    def __init__(self, plans=(), scores=()):
        self._plans = tuple(plans)
        self._scores = tuple(scores)

    def plan_candidates(self, context, actions):
        del context, actions
        return self._plans

    def score_actions(self, context, actions):
        del context
        if self._scores:
            return self._scores
        return tuple(_Score(action) for action in actions)

    def outcome_distribution(self, context, action):
        del context, action
        return {}


class _Runtime:
    def __init__(self, view=None):
        self.read_view = view or _View()
        self._stop = _Stop()
        self.watermark = 0
        self.events = []

    def make_experience(self, **kwargs):
        return dict(kwargs)

    def submit(self, event):
        self.events.append(event)
        self.watermark += 1


class MixedEnvironmentV859Tests(unittest.TestCase):
    def test_mix_selector_is_exactly_five_games_from_three_environment_categories(self):
        self.assertTrue(is_mix_selector("mix"))
        self.assertEqual(resolve_game_selector("mix", None), MIX_GAME_IDS)
        self.assertEqual(len(MIX_GAME_IDS), 5)
        self.assertEqual(tuple(spec.environment_id for spec in MIX_SPECS), MIX_GAME_IDS)
        self.assertEqual({spec.environment_family for spec in MIX_SPECS}, {"arc", "gym"})
        self.assertEqual(ARC_GAME_IDS, {"ez01", "ez02", "ic01"})
        self.assertEqual(GENERIC_GAME_IDS, {"FrozenLake-v1", "ArcAgi/Chess-v0"})

    def test_non_mix_selector_delegates_to_existing_arc_selector(self):
        self.assertEqual(resolve_game_selector("ez01,ez02", None), ("ez01", "ez02"))

    def test_five_actors_give_one_lane_to_each_mix_entry(self):
        jobs = _actor_jobs(
            MIX_GAME_IDS,
            actors=5,
            steps_per_game=100,
            seed=0,
            env_root=None,
            epsilon=0.1,
        )
        self.assertEqual(len(jobs), 5)
        self.assertEqual(tuple(job.game_id for job in jobs), MIX_GAME_IDS)
        self.assertTrue(all(job.steps == 100 for job in jobs))

    def test_actor_count_below_five_still_preserves_all_five_environments(self):
        jobs = _actor_jobs(
            MIX_GAME_IDS,
            actors=2,
            steps_per_game=25,
            seed=0,
            env_root=None,
            epsilon=0.1,
        )
        self.assertEqual(tuple(job.game_id for job in jobs), MIX_GAME_IDS)
        self.assertEqual(sum(job.steps for job in jobs), 125)

    def test_generic_classifier_does_not_capture_arc_games(self):
        for game in ARC_GAME_IDS:
            self.assertFalse(is_generic_game(game))
        for game in GENERIC_GAME_IDS:
            self.assertTrue(is_generic_game(game))

    def test_action_choice_never_executes_non_available_planner_action(self):
        view = _View(plans=(_Plan(999999),), scores=(_Score(3, 5, 10.0), _Score(7, 1, 1.0)))
        action, planned = _choose_action(view, 10, (3, 7), __import__("random").Random(1), 0.0)
        self.assertEqual(action, 3)
        self.assertFalse(planned)

    def test_action_choice_can_use_executable_shared_memory_plan(self):
        view = _View(plans=(_Plan(7),))
        action, planned = _choose_action(view, 10, (3, 7), __import__("random").Random(1), 0.0)
        self.assertEqual(action, 7)
        self.assertTrue(planned)

    def test_frozenlake_generic_actor_submits_environment_scoped_experience(self):
        runtime = _Runtime()
        job = actor_module.ActorJob(101, "FrozenLake-v1", 8, 123, epsilon=1.0)
        result = run_generic_actor_job(runtime, job)
        self.assertGreater(result.steps, 0)
        self.assertTrue(runtime.events)
        event = runtime.events[0]
        self.assertEqual(event["producer_id"], 101)
        self.assertNotEqual(event["source_game_hash"], 0)
        self.assertIn(event["terminal_polarity"], {-1, 0, 1})
        self.assertGreater(event["carrier_signature"], 0)

    def test_chess_generic_actor_uses_variable_legal_action_subset(self):
        runtime = _Runtime()
        job = actor_module.ActorJob(102, "ArcAgi/Chess-v0", 2, 456, epsilon=1.0)
        result = run_generic_actor_job(runtime, job)
        self.assertEqual(result.steps, 2)
        self.assertEqual(len(runtime.events), 2)
        for event in runtime.events:
            self.assertGreaterEqual(event["action_id"], 0)
            self.assertLess(event["action_id"], 64 * 64 * 5)
            self.assertGreater(event["changed_cells"], 0)

    def test_frozenlake_and_chess_have_distinct_grounded_provenance(self):
        frozen = _Runtime()
        chess = _Runtime()
        run_generic_actor_job(frozen, actor_module.ActorJob(201, "FrozenLake-v1", 1, 1, epsilon=1.0))
        run_generic_actor_job(chess, actor_module.ActorJob(202, "ArcAgi/Chess-v0", 1, 1, epsilon=1.0))
        self.assertNotEqual(frozen.events[0]["source_game_hash"], chess.events[0]["source_game_hash"])
        self.assertNotEqual(frozen.events[0]["carrier_signature"], chess.events[0]["carrier_signature"])

    def test_runtime_stack_installs_mixed_run_actor_authority(self):
        self.assertEqual(actor_module.run_actor_jobs.__module__, "v8.mixed_environment_v859")


if __name__ == "__main__":
    unittest.main()
