from __future__ import annotations

import os
import unittest
from unittest.mock import patch

import v8
from v8.adaptive_learning_allocation_v819 import (
    AdaptiveLearningConfig,
    AdaptiveLearningCoordinator,
    FrontierCandidate,
    FrontierScope,
    FrontierSource,
    GameLearningState,
    M7StrategyFrontier,
    SamplingMode,
)
from v8.model import MemoryLevel, MemoryType, MemoryUid, ValidationState


def uid(value: int) -> MemoryUid:
    return MemoryUid.from_key(MemoryLevel.M7, MemoryType.STRATEGY, (value, value + 1))


def scope(game: str = "g", level: int = 1, outcome: int = 1) -> FrontierScope:
    target = MemoryUid.from_key(MemoryLevel.M6, MemoryType.OUTCOME, (outcome, 2, 3))
    return FrontierScope(game, level, 1234, target.hi, target.lo)


def candidate(
    value: int,
    *,
    cost: int,
    attempts: int,
    successes: int,
    source: FrontierSource = FrontierSource.SAMPLER,
) -> FrontierCandidate:
    return FrontierCandidate(
        uid(value),
        f"trajectory-{value}",
        value * 100,
        cost,
        attempts,
        successes,
        int(ValidationState.TESTED),
        source,
        10,
    )


class UnifiedFrontierTests(unittest.TestCase):
    def test_pareto_frontier_keeps_cost_reliability_tradeoffs(self) -> None:
        frontier = M7StrategyFrontier()
        target = scope()
        cheap = candidate(1, cost=10, attempts=2, successes=1)
        reliable = candidate(2, cost=12, attempts=4, successes=4)
        dominated = candidate(3, cost=13, attempts=4, successes=1)

        self.assertTrue(frontier.add(target, cheap)[0])
        self.assertTrue(frontier.add(target, reliable)[0])
        self.assertFalse(frontier.add(target, dominated)[0])
        self.assertEqual(
            {row.trajectory_id for row in frontier.candidates(target)},
            {cheap.trajectory_id, reliable.trajectory_id},
        )

    def test_cheaper_equally_reliable_candidate_replaces_prior(self) -> None:
        frontier = M7StrategyFrontier()
        target = scope()
        old = candidate(1, cost=20, attempts=2, successes=2)
        new = candidate(2, cost=12, attempts=2, successes=2)
        frontier.add(target, old)
        changed, version = frontier.add(target, new)
        self.assertTrue(changed)
        self.assertEqual(version, 2)
        self.assertEqual(frontier.candidates(target), (new,))


class LearningStateTests(unittest.TestCase):
    def test_partial_level_success_keeps_game_discovery_priority(self) -> None:
        coordinator = AdaptiveLearningCoordinator()
        target = scope("partial", level=2)
        coordinator.observe_frontier_candidate(
            target,
            candidate(1, cost=20, attempts=2, successes=2),
            terminal_state="LEVEL",
            generation=10,
        )
        self.assertEqual(coordinator.game_state("partial"), GameLearningState.UNSOLVED)
        self.assertAlmostEqual(coordinator.sampling_weight("partial"), 1.0)
        self.assertEqual(coordinator.choose_mode("partial"), SamplingMode.DISCOVERY)

    def test_validated_win_optimizes_then_stabilizes_and_reactivates(self) -> None:
        config = AdaptiveLearningConfig(stabilization_generations=5)
        coordinator = AdaptiveLearningCoordinator(config=config)
        target = scope("won", level=5)
        coordinator.observe_frontier_candidate(
            target,
            candidate(1, cost=20, attempts=2, successes=2),
            terminal_state="WIN",
            generation=10,
        )
        self.assertEqual(
            coordinator.game_state("won"), GameLearningState.SOLVED_OPTIMIZING
        )
        self.assertAlmostEqual(coordinator.sampling_weight("won"), 0.20)

        coordinator.mark_optimizer_idle(generation=12)
        self.assertEqual(
            coordinator.game_state("won"), GameLearningState.SOLVED_OPTIMIZING
        )
        coordinator.stabilize(generation=15)
        self.assertEqual(coordinator.game_state("won"), GameLearningState.SOLVED_STABLE)
        self.assertAlmostEqual(coordinator.sampling_weight("won"), 0.075)

        coordinator.observe_frontier_candidate(
            target,
            candidate(2, cost=15, attempts=2, successes=2),
            terminal_state="WIN",
            generation=20,
        )
        self.assertEqual(
            coordinator.game_state("won"), GameLearningState.SOLVED_OPTIMIZING
        )

    def test_optimization_budget_and_no_improvement_gate(self) -> None:
        config = AdaptiveLearningConfig(
            optimization_validation_budget=4,
            max_validations_without_improvement=3,
        )
        coordinator = AdaptiveLearningCoordinator(config=config)
        self.assertTrue(coordinator.reserve_optimization(game_id="g", level=1, attempts=2))
        coordinator.record_optimizer_validation(
            game_id="g",
            level=1,
            attempts=2,
            successes=0,
            saved_actions=0,
            improved=False,
            generation=1,
        )
        self.assertTrue(coordinator.reserve_optimization(game_id="g", level=1, attempts=2))
        coordinator.record_optimizer_validation(
            game_id="g",
            level=1,
            attempts=2,
            successes=0,
            saved_actions=0,
            improved=False,
            generation=2,
        )
        self.assertFalse(coordinator.reserve_optimization(game_id="g", level=1, attempts=1))


class AllocationTests(unittest.TestCase):
    def test_weighted_allocator_redirects_credits_to_unsolved_game(self) -> None:
        config = AdaptiveLearningConfig(stabilization_generations=1)
        coordinator = AdaptiveLearningCoordinator(config=config)
        target = scope("stable", level=5)
        coordinator.observe_frontier_candidate(
            target,
            candidate(1, cost=10, attempts=2, successes=2),
            terminal_state="WIN",
            generation=1,
        )
        coordinator.mark_optimizer_idle(generation=2)
        coordinator.register_games(("stable", "unsolved"))

        for _ in range(120):
            game = coordinator.choose_game(("stable", "unsolved"))
            coordinator.record_lease(game, coordinator.choose_mode(game), 100)

        rows = {row.game_id: row for row in coordinator.telemetry()}
        self.assertGreater(rows["unsolved"].sample_steps, rows["stable"].sample_steps * 8)
        self.assertEqual(rows["stable"].state, GameLearningState.SOLVED_STABLE.value)
        self.assertEqual(rows["unsolved"].state, GameLearningState.UNSOLVED.value)

    def test_known_long_frontier_expands_lease(self) -> None:
        coordinator = AdaptiveLearningCoordinator(
            config=AdaptiveLearningConfig(lease_steps=512)
        )
        target = scope("long", level=5)
        coordinator.observe_frontier_candidate(
            target,
            candidate(1, cost=3000, attempts=2, successes=2),
            terminal_state="WIN",
            generation=1,
        )
        self.assertGreaterEqual(coordinator.recommended_lease_steps("long", 10000), 3750)

    def test_solved_sampling_modes_include_verify_alternative_transfer(self) -> None:
        coordinator = AdaptiveLearningCoordinator()
        target = scope("won", level=5)
        coordinator.observe_frontier_candidate(
            target,
            candidate(1, cost=20, attempts=2, successes=2),
            terminal_state="WIN",
            generation=1,
        )
        modes = {coordinator.choose_mode("won") for _ in range(8)}
        self.assertEqual(
            modes,
            {SamplingMode.VERIFY, SamplingMode.ALTERNATIVE, SamplingMode.TRANSFER},
        )


class PersistenceTests(unittest.TestCase):
    def test_learning_state_and_frontier_persist_but_run_allocation_does_not(self) -> None:
        coordinator = AdaptiveLearningCoordinator()
        target = scope("persist", level=5)
        coordinator.observe_frontier_candidate(
            target,
            candidate(1, cost=25, attempts=2, successes=2),
            terminal_state="WIN",
            generation=4,
        )
        coordinator.record_lease("persist", SamplingMode.VERIFY, 500)
        payload = coordinator.state_dict()

        restored = AdaptiveLearningCoordinator()
        restored.load_state(payload)
        self.assertEqual(
            restored.game_state("persist"), GameLearningState.SOLVED_OPTIMIZING
        )
        self.assertIsNotNone(restored.frontier.best_for_game("persist"))
        telemetry = {row.game_id: row for row in restored.telemetry()}
        self.assertEqual(telemetry["persist"].sample_steps, 0)

    def test_state_dict_does_not_call_live_sampling_weight_path(self) -> None:
        coordinator = AdaptiveLearningCoordinator()
        coordinator.register_games(("persist",))

        with patch.object(
            coordinator,
            "sampling_weight",
            side_effect=AssertionError("live lifecycle scan must not run in snapshot"),
        ):
            payload = coordinator.state_dict()

        self.assertEqual(payload["sampling_weight"]["persist"], 1.0)


class SamplingModePolicyTests(unittest.TestCase):
    def test_alternative_mode_bypasses_optimized_sidecar_and_excludes_winner(self) -> None:
        from v8 import adaptive_learning_allocation_v819 as v819
        from v8 import trajectory_optimizer_v814 as optimizer

        expected = (object(),)
        captured = {}

        def base(_view, _context, _actions, **kwargs):
            captured.update(kwargs)
            return expected

        exclude = uid(44)
        old_mode = os.environ.get(v819._SAMPLING_MODE_ENV)
        old_exclude = os.environ.get(v819._ALTERNATIVE_EXCLUDE_ENV)
        try:
            os.environ[v819._SAMPLING_MODE_ENV] = SamplingMode.ALTERNATIVE.value
            os.environ[v819._ALTERNATIVE_EXCLUDE_ENV] = f"{exclude.hi}:{exclude.lo}"
            with patch.object(optimizer, "_BASE_PLAN_CANDIDATES", base):
                result = v819._plan_candidates_v819(object(), 1, (1, 2, 3))
            self.assertIs(result, expected)
            self.assertIn(exclude, captured["excluded_strategies"])
        finally:
            if old_mode is None:
                os.environ.pop(v819._SAMPLING_MODE_ENV, None)
            else:
                os.environ[v819._SAMPLING_MODE_ENV] = old_mode
            if old_exclude is None:
                os.environ.pop(v819._ALTERNATIVE_EXCLUDE_ENV, None)
            else:
                os.environ[v819._ALTERNATIVE_EXCLUDE_ENV] = old_exclude


if __name__ == "__main__":
    unittest.main()
