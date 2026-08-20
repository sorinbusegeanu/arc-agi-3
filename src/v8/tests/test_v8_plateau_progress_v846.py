from __future__ import annotations

import unittest

import v8
from v8 import actor as actor_module
from v8 import adaptive_learning_allocation_v819 as v819
from v8 import plateau_progress_v846 as v846
from v8.diagnostics import game_summary
from v8.model import MemoryLevel, MemoryType, MemoryUid, ValidationState


def _uid(value: int) -> MemoryUid:
    return MemoryUid.from_key(
        MemoryLevel.M7,
        MemoryType.STRATEGY,
        (int(value), int(value) + 1),
    )


def _scope(game: str = "g", level: int = 5) -> v819.FrontierScope:
    outcome = MemoryUid.from_key(
        MemoryLevel.M6,
        MemoryType.OUTCOME,
        (int(level), 2, 3),
    )
    return v819.FrontierScope(
        str(game),
        int(level),
        1234,
        int(outcome.hi),
        int(outcome.lo),
    )


def _candidate(
    value: int,
    *,
    cost: int,
    attempts: int,
    successes: int,
) -> v819.FrontierCandidate:
    return v819.FrontierCandidate(
        _uid(value),
        f"trajectory-{value}",
        int(value) * 100,
        int(cost),
        int(attempts),
        int(successes),
        int(ValidationState.TESTED),
        v819.FrontierSource.SAMPLER,
        10,
    )


class AdaptiveProgressDepthTests(unittest.TestCase):
    def setUp(self) -> None:
        v846._reset_progress_depth_v846()

    def test_completed_lease_retains_real_episode_depth(self) -> None:
        progress = actor_module.ActorProgress(
            actor_id=1,
            game_id="g",
            steps=50,
            wins=0,
            failures=0,
            levels_completed=17,
            max_level_reached=3,
        )
        event = v819._LeaseProgress(1, 1, progress)

        # The adaptive scheduler reads event.row before discarding completed lease
        # progress. That read must preserve the deepest real episode depth.
        self.assertIs(event.row, progress)

        job = actor_module.ActorJob(1, "g", 100, 0)
        rows = v819._adaptive_progress_rows(
            actor_module,
            (job,),
            {
                "g": {
                    "steps": 50,
                    "wins": 0,
                    "failures": 0,
                    "levels_completed": 17,
                    "replans": 0,
                    "planned_steps": 0,
                    "first_win_step": 0,
                }
            },
            {},
            {},
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].max_level_reached, 3)
        self.assertEqual(game_summary(rows), (0.0, 60.0, 0, 1))

    def test_depth_is_high_water_across_adaptive_leases(self) -> None:
        first = actor_module.ActorProgress(
            actor_id=1,
            game_id="g",
            steps=20,
            wins=0,
            failures=0,
            levels_completed=2,
            max_level_reached=4,
        )
        later = actor_module.ActorProgress(
            actor_id=2,
            game_id="g",
            steps=20,
            wins=0,
            failures=0,
            levels_completed=1,
            max_level_reached=1,
        )
        _ = v819._LeaseProgress(1, 1, first).row
        _ = v819._LeaseProgress(2, 2, later).row

        job = actor_module.ActorJob(1, "g", 100, 0)
        rows = v819._adaptive_progress_rows(
            actor_module,
            (job,),
            {"g": {"steps": 40, "wins": 0, "failures": 0, "levels_completed": 3}},
            {},
            {},
        )
        self.assertEqual(rows[0].max_level_reached, 4)
        self.assertEqual(game_summary(rows), (0.0, 80.0, 0, 1))


class FrontierOptimizationPlateauTests(unittest.TestCase):
    def test_same_cost_pareto_churn_does_not_reactivate_stable_game(self) -> None:
        coordinator = v819.AdaptiveLearningCoordinator(
            config=v819.AdaptiveLearningConfig(stabilization_generations=1)
        )
        target = _scope("won")
        coordinator.observe_frontier_candidate(
            target,
            _candidate(1, cost=10, attempts=2, successes=1),
            terminal_state="WIN",
            generation=1,
        )
        coordinator.mark_optimizer_idle(generation=2)
        self.assertEqual(
            coordinator.game_state("won"),
            v819.GameLearningState.SOLVED_STABLE,
        )

        record = coordinator._record("won", 5)
        prior_improvement_generation = int(record.last_frontier_improvement_generation)
        prior_validations = int(record.validations_since_improvement)
        prior_competence = float(coordinator._signals["won"].competence_improvement)
        prior_novelty = float(coordinator._signals["won"].novelty)
        prior_version = int(record.frontier_version)

        changed = coordinator.observe_frontier_candidate(
            target,
            _candidate(2, cost=12, attempts=4, successes=4),
            terminal_state="WIN",
            generation=20,
        )

        self.assertTrue(changed)
        self.assertGreater(record.frontier_version, prior_version)
        self.assertEqual(
            coordinator.game_state("won"),
            v819.GameLearningState.SOLVED_STABLE,
        )
        self.assertEqual(
            record.last_frontier_improvement_generation,
            prior_improvement_generation,
        )
        self.assertEqual(record.validations_since_improvement, prior_validations)
        self.assertEqual(record.optimizer_exhausted_version, record.frontier_version)
        self.assertEqual(
            coordinator._signals["won"].competence_improvement,
            prior_competence,
        )
        self.assertEqual(coordinator._signals["won"].novelty, prior_novelty)

    def test_cheaper_frontier_candidate_reactivates_optimization(self) -> None:
        coordinator = v819.AdaptiveLearningCoordinator(
            config=v819.AdaptiveLearningConfig(stabilization_generations=1)
        )
        target = _scope("won")
        coordinator.observe_frontier_candidate(
            target,
            _candidate(1, cost=10, attempts=2, successes=2),
            terminal_state="WIN",
            generation=1,
        )
        coordinator.mark_optimizer_idle(generation=2)
        self.assertEqual(
            coordinator.game_state("won"),
            v819.GameLearningState.SOLVED_STABLE,
        )

        coordinator.observe_frontier_candidate(
            target,
            _candidate(2, cost=8, attempts=2, successes=2),
            terminal_state="WIN",
            generation=20,
        )

        record = coordinator._record("won", 5)
        self.assertEqual(
            coordinator.game_state("won"),
            v819.GameLearningState.SOLVED_OPTIMIZING,
        )
        self.assertEqual(record.last_frontier_improvement_generation, 20)
        self.assertEqual(record.validations_since_improvement, 0)
        self.assertEqual(record.optimizer_exhausted_version, -1)
        self.assertEqual(coordinator.frontier.winner(target).cost, 8)


if __name__ == "__main__":
    unittest.main()
