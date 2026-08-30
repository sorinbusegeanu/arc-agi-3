from __future__ import annotations

import threading
import unittest
from unittest.mock import patch

import v8
from v8 import actor as actor_module
from v8 import cli as base_cli
from v8 import mixed_environment_v859 as mixed
from v8.cli import _actor_jobs
from v8.cli_v819 import main as adaptive_cli_main
from v8.mixed_environment_v859 import (
    ARC_GAME_IDS,
    GENERIC_GAME_IDS,
    MIX_GAME_IDS,
    MIX_SPECS,
    MIX_TRANSFER_EXPERIMENT_SCOPE,
    _choose_action,
    is_generic_game,
    is_mix_selector,
    resolve_mixed_game_selector,
    run_generic_actor_job,
    run_mixed_actor_jobs,
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
        self.invalidations = 0

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

    def invalidate_strategy_cache(self):
        self.invalidations += 1


class _Runtime:
    def __init__(self, view=None, watermark=0):
        self.read_view = view or _View()
        self._stop = _Stop()
        self.watermark = int(watermark)
        self.events = []

    def make_experience(self, **kwargs):
        return dict(kwargs)

    def submit(self, event):
        self.events.append(event)
        self.watermark += 1


class MixedEnvironmentV859Tests(unittest.TestCase):
    def test_mix_selector_has_five_games_across_four_environment_categories(self):
        self.assertTrue(is_mix_selector("mix"))
        self.assertEqual(resolve_mixed_game_selector("mix"), MIX_GAME_IDS)
        self.assertIsNone(resolve_mixed_game_selector("ic01,gp03"))
        self.assertEqual(len(MIX_GAME_IDS), 5)
        self.assertEqual(tuple(spec.environment_id for spec in MIX_SPECS), MIX_GAME_IDS)
        self.assertEqual(
            {spec.environment_family for spec in MIX_SPECS},
            {"arc", "gym", "chess", "sudoku"},
        )
        self.assertNotIn("ez01", MIX_GAME_IDS)
        self.assertNotIn("ez02", MIX_GAME_IDS)
        self.assertNotIn("ic01", MIX_GAME_IDS)
        self.assertNotIn("ic02", MIX_GAME_IDS)
        self.assertIn("gp03", MIX_GAME_IDS)
        self.assertIn("ArcAgi/Sudoku-v0", MIX_GAME_IDS)
        self.assertEqual(ARC_GAME_IDS, {"gp03", "tp02"})
        self.assertEqual(
            GENERIC_GAME_IDS,
            {"FrozenLake-v1", "ArcAgi/Chess-v0", "ArcAgi/Sudoku-v0"},
        )
        self.assertEqual(MIX_TRANSFER_EXPERIMENT_SCOPE, "arc-only")

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

    def test_actor_count_below_five_still_preserves_all_environments(self):
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

    def test_restored_generic_actor_uses_fresh_producer_sequence(self):
        runtime = _Runtime(watermark=50_000)
        job = actor_module.ActorJob(101, "FrozenLake-v1", 2, 123, epsilon=1.0)
        run_generic_actor_job(runtime, job)
        self.assertTrue(runtime.events)
        self.assertGreater(runtime.events[0]["producer_sequence"], 50_000)

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

    def test_sudoku_generic_actor_submits_placement_experience(self):
        runtime = _Runtime()
        job = actor_module.ActorJob(103, "ArcAgi/Sudoku-v0", 3, 789, epsilon=1.0)
        result = run_generic_actor_job(runtime, job)
        self.assertGreater(result.steps, 0)
        self.assertTrue(runtime.events)
        for event in runtime.events:
            self.assertGreaterEqual(event["action_id"], 0)
            self.assertLess(event["action_id"], 729)
            self.assertEqual(event["changed_cells"], 1)
            self.assertGreater(event["carrier_signature"], 0)

    def test_generic_environments_have_distinct_grounded_provenance(self):
        frozen = _Runtime()
        chess = _Runtime()
        sudoku = _Runtime()
        run_generic_actor_job(frozen, actor_module.ActorJob(201, "FrozenLake-v1", 1, 1, epsilon=1.0))
        run_generic_actor_job(chess, actor_module.ActorJob(202, "ArcAgi/Chess-v0", 1, 1, epsilon=1.0))
        run_generic_actor_job(sudoku, actor_module.ActorJob(203, "ArcAgi/Sudoku-v0", 1, 1, epsilon=1.0))
        hashes = {
            frozen.events[0]["source_game_hash"],
            chess.events[0]["source_game_hash"],
            sudoku.events[0]["source_game_hash"],
        }
        carriers = {
            frozen.events[0]["carrier_signature"],
            chess.events[0]["carrier_signature"],
            sudoku.events[0]["carrier_signature"],
        }
        self.assertEqual(len(hashes), 3)
        self.assertEqual(len(carriers), 3)

    def test_generic_sampling_starts_before_arc_batch_finishes(self):
        generic_started = threading.Event()
        arc_can_finish = threading.Event()

        class _Pause:
            def __init__(self):
                self._event = threading.Event()
                self._event.set()

            def is_set(self):
                return self._event.is_set()

            def clear(self):
                self._event.clear()

        class _Peers:
            def __init__(self):
                self._pause = _Pause()

        runtime = _Runtime()
        runtime.peers = _Peers()

        arc_job = actor_module.ActorJob(1, "gp03", 1, 1)
        generic_job = actor_module.ActorJob(2, "ArcAgi/Sudoku-v0", 1, 2)

        def fake_arc(_runtime, jobs, **_kwargs):
            self.assertEqual(tuple(job.game_id for job in jobs), ("gp03",))
            runtime.peers._pause.clear()
            self.assertTrue(generic_started.wait(1.0))
            arc_can_finish.set()
            return (actor_module.ActorResult(1, "gp03", 1, 0, 0, 0, 0),)

        def fake_generic(_runtime, job, **_kwargs):
            self.assertEqual(job.game_id, "ArcAgi/Sudoku-v0")
            generic_started.set()
            self.assertTrue(arc_can_finish.wait(1.0))
            return actor_module.ActorResult(2, job.game_id, 1, 0, 0, 0, 0)

        with (
            patch.object(mixed, "run_arc_actor_jobs", side_effect=fake_arc),
            patch.object(mixed, "run_generic_actor_job", side_effect=fake_generic),
        ):
            results = run_mixed_actor_jobs(runtime, (arc_job, generic_job))
        self.assertEqual({row.actor_id for row in results}, {1, 2})

    def test_mix_cli_patches_only_cli_dispatch_and_restores_it_after_run(self):
        original_actor_dispatch = base_cli.run_actor_jobs
        original_experiment_dispatch = base_cli.run_automatic_transfer_experiments

        def inspect_main(argv):
            from v7.game_sets import resolve_game_selector

            self.assertEqual(resolve_game_selector("mix", None), MIX_GAME_IDS)
            self.assertIs(base_cli.run_actor_jobs, run_mixed_actor_jobs)
            self.assertEqual(argv[:3], ["continuous-run", "--games", "mix"])
            return 0

        with patch.object(base_cli, "main", side_effect=inspect_main):
            self.assertEqual(
                adaptive_cli_main(["continuous-run", "--games", "mix", "--actors", "5"]),
                0,
            )
        self.assertIs(base_cli.run_actor_jobs, original_actor_dispatch)
        self.assertIs(base_cli.run_automatic_transfer_experiments, original_experiment_dispatch)

    def test_arc_actor_public_authority_is_not_replaced_by_mix_feature(self):
        self.assertIsNot(actor_module.run_actor_jobs, run_mixed_actor_jobs)
        self.assertNotEqual(actor_module.run_actor_jobs.__module__, "v8.mixed_environment_v859")


if __name__ == "__main__":
    unittest.main()
