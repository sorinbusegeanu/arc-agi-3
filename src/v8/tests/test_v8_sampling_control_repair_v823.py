from __future__ import annotations

import os
import threading
import types
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import v8
from v8 import actor as actor_module
from v8 import adaptive_learning_allocation_v819 as v819
from v8 import adaptive_learning_allocation_v819_performance_fix as perf
from v8 import adaptive_learning_allocation_v819_performance_fixups as perf_fixups
from v8 import adaptive_learning_allocation_v819_solve_fix as solve_fix
from v8 import cli_v819
from v8 import decision_point_sampling_v821 as sampling
from v8 import learning_fixes_v088 as learning
from v8 import progressive_level_learning_v820 as progressive
from v8 import runtime_repair_v822 as v822
from v8 import sampling_control_repair_v823 as repair
from v8 import trajectory_optimizer_v814 as optimizer
from v8 import trajectory_optimizer_v818 as v818
from v8.model import MemoryLevel, MemoryType, MemoryUid
from v8.publication import LiveReadView, PlannedAction


def _source(*, game="world", level=1, actions=(1, 2), prefix=()):
    anchor = optimizer.ReplayAnchor(game, 0, tuple(prefix), None)
    target = optimizer.TrajectoryTarget(level, "LEVEL")
    return optimizer.SuccessfulTrajectory(
        optimizer._trajectory_id(anchor, target, tuple(actions)),
        anchor,
        target,
        tuple(actions),
        MemoryUid.zero(),
        MemoryUid.zero(),
        0,
    )


class _Queue:
    def __init__(self) -> None:
        self.rows = []

    def put(self, row) -> None:
        self.rows.append(row)


class PlannerAuthorityTests(unittest.TestCase):
    def test_v822_probe_suppression_is_no_longer_live(self) -> None:
        self.assertIs(LiveReadView.plan_candidates, v822._BASE_PLAN_CANDIDATES)

    def test_success_verification_is_reduced_to_one_repeat(self) -> None:
        self.assertEqual(sampling._VERIFICATION_REPEATS, 1)
        sampler = sampling.DecisionPointSampler("ez01")
        point = sampler.register_point(level=0, context=11, anchor=(), actions=(1, 2))
        sampler.current = sampling.Intervention("DISCOVERY", point.key, 1, ())
        sampler.observe_transition(
            before_level=0,
            before_context=11,
            action=1,
            after_level=1,
            after_context=22,
            after_actions=(1, 2),
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


class AdaptivePoolTests(unittest.TestCase):
    def test_requested_pool_caps_worker_processes(self) -> None:
        prior = os.environ.get(repair._ACTOR_POOL_ENV)
        try:
            os.environ[repair._ACTOR_POOL_ENV] = "8"
            self.assertEqual(repair.requested_actor_pool(36), 8)
            os.environ[repair._ACTOR_POOL_ENV] = "20"
            self.assertEqual(repair.requested_actor_pool(36), 20)
            os.environ[repair._ACTOR_POOL_ENV] = "100"
            self.assertEqual(repair.requested_actor_pool(36), 36)
        finally:
            if prior is None:
                os.environ.pop(repair._ACTOR_POOL_ENV, None)
            else:
                os.environ[repair._ACTOR_POOL_ENV] = prior

    def test_large_game_set_gets_bounded_initial_breadth_lease(self) -> None:
        self.assertEqual(
            repair.initial_unsolved_lease_steps(
                available=360000,
                base_steps=10000,
                initial_probe=True,
                worker_count=8,
                game_count=36,
            ),
            2048,
        )
        self.assertEqual(
            repair.initial_unsolved_lease_steps(
                available=360000,
                base_steps=10000,
                initial_probe=False,
                worker_count=8,
                game_count=36,
            ),
            10000,
        )

    def test_adaptive_runner_uses_v823_pool_helpers_and_preserves_explicit_dispatcher(self) -> None:
        root = perf._adaptive_run_actor_jobs_perf.__code__
        codes = [root]
        codes.extend(value for value in root.co_consts if isinstance(value, types.CodeType))
        self.assertIn("_v823_requested_actor_pool", set(root.co_names))
        self.assertTrue(
            any("_v823_initial_unsolved_lease_steps" in set(code.co_names) for code in codes)
        )
        self.assertIn("best_win_steps", root.co_consts)
        self.assertIn("last_win_steps", root.co_consts)
        self.assertIs(progressive._BASE_RUN_ACTOR_JOBS, perf_fixups._run_actor_jobs_v819)

    def test_cli_actor_batch_reports_pool_but_retains_all_game_jobs(self) -> None:
        batch = cli_v819._ActorJobBatch(tuple(range(36)), 8)
        self.assertEqual(len(batch), 8)
        self.assertEqual(tuple(batch), tuple(range(36)))
        self.assertEqual(cli_v819._requested_actor_pool(["continuous-run"]), 8)
        self.assertEqual(
            cli_v819._requested_actor_pool(["continuous-run", "--actors", "20"]),
            20,
        )


class AdaptiveWinMetricTests(unittest.TestCase):
    def test_result_adapter_carries_process_local_win_metrics(self) -> None:
        prior_best = learning._BEST_WIN_STEPS
        prior_last = learning._LAST_WIN_STEPS
        target = _Queue()
        lease = v819.ActorLease(
            1,
            1,
            "ic01",
            100,
            7,
            None,
            0.10,
            1000,
            v819.SamplingMode.DISCOVERY,
            MemoryUid.zero(),
        )
        result = actor_module.ActorResult(1, "ic01", 42, 1, 0, 5, 2)
        try:
            learning._BEST_WIN_STEPS = 17
            learning._LAST_WIN_STEPS = 19
            repair._ResultAdapterV823(target, 1, lease).put(result)
        finally:
            learning._BEST_WIN_STEPS = prior_best
            learning._LAST_WIN_STEPS = prior_last

        self.assertEqual(len(target.rows), 1)
        event = target.rows[0]
        self.assertIsInstance(event, v819._LeaseResult)
        self.assertEqual(event.result.best_win_steps, 17)
        self.assertEqual(event.result.last_win_steps, 19)
        self.assertEqual(event.result.wins, 1)

    def test_completed_win_metrics_survive_progress_synthesis(self) -> None:
        self.assertIs(v819._adaptive_progress_rows, repair._adaptive_progress_rows_v823)
        jobs = (actor_module.ActorJob(1, "ic01", 10000, 0),)
        completed = {
            "ic01": {
                "steps": 42,
                "wins": 1,
                "failures": 0,
                "levels_completed": 5,
                "replans": 0,
                "planned_steps": 0,
                "first_win_step": 42,
                "best_win_steps": 17,
                "last_win_steps": 19,
                "resets": 2,
            }
        }
        row = v819._adaptive_progress_rows(
            actor_module,
            jobs,
            completed,
            {},
            {},
        )[0]
        self.assertEqual(row.wins, 1)
        self.assertEqual(row.best_win_steps, 17)
        self.assertEqual(row.last_win_steps, 19)


class PreWinValidationBudgetTests(unittest.TestCase):
    def test_large_unsolved_set_defers_without_starting_replay_validation(self) -> None:
        service = SimpleNamespace(_v819_lock=threading.RLock())
        row = _source(game="ez01", level=1)
        base = Mock(return_value=True)
        with (
            patch.object(progressive, "_is_runtime_unsolved_partial", return_value=True),
            patch.object(solve_fix, "_defer_pre_win_source", return_value=True) as defer,
            patch.object(repair, "_unsolved_game_count", return_value=36),
            patch.object(repair, "_BASE_PROGRESSIVE_SUBMIT", base),
        ):
            self.assertTrue(repair._bounded_progressive_submit(service, row))
        defer.assert_called_once_with(service, row)
        base.assert_not_called()

    def test_small_set_admits_only_first_best_source_per_level(self) -> None:
        service = SimpleNamespace(
            _v819_lock=threading.RLock(),
            _v819_pre_win_sources={},
        )
        first = _source(game="ez01", level=1, actions=(1, 2, 3))
        second = _source(game="ez01", level=1, actions=(1,))
        base = Mock(return_value=True)

        def defer(_service, trajectory):
            _service._v819_pre_win_sources.setdefault("ez01", {})[1] = trajectory
            return True

        with (
            patch.object(progressive, "_is_runtime_unsolved_partial", return_value=True),
            patch.object(solve_fix, "_defer_pre_win_source", side_effect=defer),
            patch.object(repair, "_unsolved_game_count", return_value=4),
            patch.object(repair, "_BASE_PROGRESSIVE_SUBMIT", base),
        ):
            self.assertTrue(repair._bounded_progressive_submit(service, first))
            self.assertTrue(repair._bounded_progressive_submit(service, second))
        self.assertEqual(base.call_count, 1)


class SeedlessActivationSafetyTests(unittest.TestCase):
    def test_cross_prefix_first_action_match_is_not_enough(self) -> None:
        outcome = MemoryUid.from_key(MemoryLevel.M6, MemoryType.OUTCOME, (1, 2, 3))
        parent_strategy = MemoryUid.from_key(MemoryLevel.M7, MemoryType.STRATEGY, (4, 5, 6))
        other_strategy = MemoryUid.from_key(MemoryLevel.M7, MemoryType.STRATEGY, (7, 8, 9))
        anchor = optimizer.ReplayAnchor("world", 0, (9, 9), None)
        target = optimizer.TrajectoryTarget(2, "LEVEL")
        row = optimizer.ValidatedTrajectory(
            "row",
            anchor,
            target,
            (1, 2),
            MemoryUid.from_key(MemoryLevel.M7, MemoryType.STRATEGY, (10, 11, 12)),
            outcome,
            parent_strategy,
            5,
            "DIRECT_ACTION",
            2,
            2,
        )
        view = SimpleNamespace(_v814_variants=(row,), _v814_attempted_variants=set())
        plan = PlannedAction(1, outcome, other_strategy, 1.0, False)
        prior_source = optimizer._CAPTURE_SOURCE_ID
        prior_history = list(optimizer._ACTOR_ACTION_HISTORY)
        try:
            optimizer._CAPTURE_SOURCE_ID = "world"
            optimizer._ACTOR_ACTION_HISTORY[:] = [8, 8]
            with patch.object(optimizer, "_refresh_view_variants", lambda _view: None):
                self.assertIsNone(repair._target_compatible_variant_v823(view, (plan,), (1, 2)))

            matching_plan = PlannedAction(7, outcome, parent_strategy, 1.0, False)
            with patch.object(optimizer, "_refresh_view_variants", lambda _view: None):
                self.assertIs(
                    repair._target_compatible_variant_v823(view, (matching_plan,), (1, 2)),
                    row,
                )
        finally:
            optimizer._CAPTURE_SOURCE_ID = prior_source
            optimizer._ACTOR_ACTION_HISTORY[:] = prior_history


if __name__ == "__main__":
    unittest.main()
