from __future__ import annotations

import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import v8
from v8 import adaptive_learning_allocation_v819 as v819
from v8 import normalized_memory_v086_fixups as normalized
from v8 import optimizer_budget_control_v830 as repair
from v8 import sampling_progress_control_v829 as sampling
from v8 import trajectory_optimizer_v814 as optimizer
from v8 import trajectory_optimizer_v818 as v818
from v8.model import MemoryUid, stable_u64


def source_candidate(*, game: str, level: int, cost: int):
    actions = tuple(1 for _ in range(max(1, int(cost))))
    anchor = optimizer.ReplayAnchor(game, 0, (), None)
    target = optimizer.TrajectoryTarget(level, "LEVEL")
    source = optimizer.SuccessfulTrajectory(
        optimizer._trajectory_id(anchor, target, actions),
        anchor,
        target,
        actions,
        MemoryUid.zero(),
        MemoryUid.zero(),
        0,
    )
    return optimizer.TrajectoryCandidate(
        optimizer._candidate_id(source, "DELETE_ACTION", actions[:-1] or actions),
        source,
        "DELETE_ACTION",
        actions[:-1] or actions,
        0,
        1 if len(actions) > 1 else 0,
    )


class PotentialAwareBudgetTests(unittest.TestCase):
    def test_high_headroom_receives_larger_budget_and_stall_window(self):
        coordinator = v819.AdaptiveLearningCoordinator(
            config=v819.AdaptiveLearningConfig(
                optimization_validation_budget=2048,
                max_validations_without_improvement=256,
            )
        )
        repair._register_candidate(
            coordinator,
            source_candidate(game="low", level=1, cost=2),
        )
        repair._register_candidate(
            coordinator,
            source_candidate(game="high", level=1, cost=401),
        )

        low_budget, low_stall, low_potential = repair._budget_limits(
            coordinator, "low", 1
        )
        high_budget, high_stall, high_potential = repair._budget_limits(
            coordinator, "high", 1
        )

        self.assertEqual(low_potential, 1)
        self.assertEqual(high_potential, 400)
        self.assertLess(low_budget, high_budget)
        self.assertLess(low_stall, high_stall)
        self.assertEqual(high_budget, 2048)

    def test_cost_one_level_is_immediately_minimal(self):
        coordinator = v819.AdaptiveLearningCoordinator()
        repair._register_candidate(
            coordinator,
            source_candidate(game="minimal", level=1, cost=1),
        )
        self.assertEqual(repair._budget_limits(coordinator, "minimal", 1), (0, 0, 0))

    def test_route_precheck_does_not_consume_validation_budget(self):
        coordinator = v819.AdaptiveLearningCoordinator()
        candidate = source_candidate(game="g", level=1, cost=20)
        repair._register_candidate(coordinator, candidate)

        prior, present = repair._context_set(
            repair._BUDGET_CONTEXT,
            mode="precheck",
            key=("g", 1),
        )
        try:
            self.assertTrue(
                coordinator.reserve_optimization(game_id="g", level=1, attempts=2)
            )
        finally:
            repair._context_restore(repair._BUDGET_CONTEXT, prior, present)

        self.assertEqual(
            coordinator._record("g", 1).consumed_optimization_budget,
            0,
        )

        prior, present = repair._context_set(
            repair._BUDGET_CONTEXT,
            mode="consume",
            key=("g", 1),
        )
        try:
            self.assertTrue(
                coordinator.reserve_optimization(game_id="g", level=1, attempts=2)
            )
        finally:
            repair._context_restore(repair._BUDGET_CONTEXT, prior, present)

        self.assertEqual(
            coordinator._record("g", 1).consumed_optimization_budget,
            2,
        )

    def test_actual_direct_validation_consumes_budget_and_logs_failure(self):
        events = []
        coordinator = v819.AdaptiveLearningCoordinator(event_sink=events.append)
        candidate = source_candidate(game="ic01", level=2, cost=200)

        with TemporaryDirectory() as root:
            service = optimizer.TrajectoryOptimizationService(
                Path(root),
                validator=lambda _candidate: None,
            )
            service._v819_runtime = SimpleNamespace(
                _v819_adaptive_learning=coordinator
            )
            validator = SimpleNamespace(
                validate=lambda _candidate: SimpleNamespace(
                    success=False,
                    attempts=2,
                    successes=0,
                )
            )
            result = repair._validate_tracked_v830(
                service,
                validator,
                candidate,
            )

        self.assertIsNotNone(result)
        self.assertEqual(
            coordinator._record("ic01", 2).consumed_optimization_budget,
            2,
        )
        snapshot = repair.optimizer_budget_snapshot(coordinator, "ic01")
        self.assertEqual(snapshot[0]["validations"], 2)
        self.assertTrue(any("status=NO_PROGRESS" in row for row in events))

    def test_improvement_is_logged_and_resets_no_progress(self):
        events = []
        coordinator = v819.AdaptiveLearningCoordinator(event_sink=events.append)
        repair._register_candidate(
            coordinator,
            source_candidate(game="ic01", level=2, cost=200),
        )
        stats = repair._stats_for(coordinator, "ic01", 2)
        stats.validations_since_improvement = 40

        prior, present = repair._context_set(
            repair._PROCESS_CONTEXT,
            source_cost=200,
            key=("ic01", 2),
        )
        try:
            coordinator.record_optimizer_validation(
                game_id="ic01",
                level=2,
                attempts=2,
                successes=2,
                saved_actions=150,
                improved=True,
                generation=10,
            )
        finally:
            repair._context_restore(repair._PROCESS_CONTEXT, prior, present)

        self.assertEqual(stats.best_cost, 50)
        self.assertEqual(stats.validations_since_improvement, 0)
        self.assertTrue(any("status=IMPROVED" in row for row in events))
        self.assertTrue(any("saved=150" in row for row in events))


class ValidatorPriorityTests(unittest.TestCase):
    def test_waiting_validator_slots_prefer_games_with_more_headroom(self):
        coordinator = v819.AdaptiveLearningCoordinator()
        repair._register_candidate(
            coordinator,
            source_candidate(game="low", level=1, cost=2),
        )
        repair._register_candidate(
            coordinator,
            source_candidate(game="high", level=1, cost=401),
        )
        service = SimpleNamespace(
            _v819_runtime=SimpleNamespace(_v819_adaptive_learning=coordinator),
            _v818_validator_lock=threading.RLock(),
            _v818_waiting_games={"low", "high"},
        )
        calls = []
        with patch.object(
            v818,
            "_ensure_validator",
            side_effect=lambda _service, game: calls.append(str(game)),
        ):
            repair._start_waiting_validators_v830(service)
        self.assertEqual(calls, ["high", "low"])


class EasyGameIsolationTests(unittest.TestCase):
    def setUp(self):
        sampling._reset_sampling_state_v829()
        sampling._CONTROL_STATE.game_id = "ez02"
        sampling._CONTROL_STATE.level = 0
        sampling._CONTROL_STATE.context = None
        sampling._CONTROL_STATE.selection_source = "UNKNOWN"
        sampling._CONTROL_STATE.planned_actions = frozenset()

    def tearDown(self):
        for name in (
            "game_id",
            "level",
            "context",
            "selection_source",
            "planned_actions",
        ):
            try:
                delattr(sampling._CONTROL_STATE, name)
            except AttributeError:
                pass

    def test_ez01_progress_action_cannot_control_ez02_discovery(self):
        sampling._PROGRESS_ACTION[("ez01", 0, 10)] = 4
        rows = sampling._score_actions_v829(object(), 10, (1, 2, 3, 4))
        self.assertEqual(tuple(row.action_id for row in rows), (1,))
        self.assertEqual(sampling._CONTROL_STATE.selection_source, "DISCOVERY")

    def test_grounded_control_context_is_game_local(self):
        observable = 123456789
        ez01 = normalized._grounded_context(
            stable_u64("ez01", person=b"v8-game"),
            observable,
        )
        ez02 = normalized._grounded_context(
            stable_u64("ez02", person=b"v8-game"),
            observable,
        )
        self.assertNotEqual(ez01, ez02)


if __name__ == "__main__":
    unittest.main()
