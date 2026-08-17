from __future__ import annotations

import unittest

import v8
from v8.adaptive_learning_allocation_v819 import (
    AdaptiveLearningConfig,
    AdaptiveLearningCoordinator,
    SamplingMode,
)
from v8.adaptive_learning_allocation_v819_inflight_fix import (
    _begin_inflight_tracking,
    _clear_inflight_tracking,
    _inflight_steps_for,
)


class InflightAllocationRegressionTests(unittest.TestCase):
    def test_initial_concurrent_assignments_spread_across_unsolved_games(self) -> None:
        coordinator = AdaptiveLearningCoordinator(
            config=AdaptiveLearningConfig(lease_steps=4096)
        )
        games = tuple(f"g{index:02d}" for index in range(36))
        coordinator.register_games(games)
        remaining = 36 * 10_000
        selected: list[str] = []

        _begin_inflight_tracking(coordinator)
        try:
            for _ in range(36):
                game = coordinator.choose_game(games)
                steps = coordinator.recommended_lease_steps(game, remaining)
                selected.append(game)
                remaining -= steps

            self.assertEqual(len(selected), 36)
            self.assertEqual(set(selected), set(games))
            self.assertTrue(
                all(_inflight_steps_for(coordinator, game) == 4096 for game in games)
            )
        finally:
            _clear_inflight_tracking(coordinator)

    def test_completed_lease_releases_reservation_and_records_actual_steps(self) -> None:
        coordinator = AdaptiveLearningCoordinator(
            config=AdaptiveLearningConfig(lease_steps=4096)
        )
        coordinator.register_games(("a", "b"))

        _begin_inflight_tracking(coordinator)
        try:
            game = coordinator.choose_game(("a", "b"))
            reserved = coordinator.recommended_lease_steps(game, 10_000)
            self.assertEqual(_inflight_steps_for(coordinator, game), reserved)

            coordinator.record_lease(game, SamplingMode.DISCOVERY, reserved)

            self.assertEqual(_inflight_steps_for(coordinator, game), 0)
            telemetry = {row.game_id: row for row in coordinator.telemetry()}
            self.assertEqual(telemetry[game].sample_steps, reserved)
        finally:
            _clear_inflight_tracking(coordinator)

    def test_short_lease_completion_releases_planned_reservation(self) -> None:
        coordinator = AdaptiveLearningCoordinator(
            config=AdaptiveLearningConfig(lease_steps=4096)
        )
        coordinator.register_games(("a",))

        _begin_inflight_tracking(coordinator)
        try:
            game = coordinator.choose_game(("a",))
            reserved = coordinator.recommended_lease_steps(game, 10_000)
            self.assertEqual(reserved, 4096)

            coordinator.record_lease(game, SamplingMode.DISCOVERY, 1024)

            self.assertEqual(_inflight_steps_for(coordinator, game), 0)
            telemetry = {row.game_id: row for row in coordinator.telemetry()}
            self.assertEqual(telemetry[game].sample_steps, 1024)
        finally:
            _clear_inflight_tracking(coordinator)

    def test_lease_sizing_remains_side_effect_free_outside_actor_run(self) -> None:
        coordinator = AdaptiveLearningCoordinator(
            config=AdaptiveLearningConfig(lease_steps=4096)
        )
        coordinator.register_games(("a", "b"))

        game = coordinator.choose_game(("a", "b"))
        steps = coordinator.recommended_lease_steps(game, 10_000)

        self.assertEqual(steps, 4096)
        self.assertEqual(_inflight_steps_for(coordinator, game), 0)


if __name__ == "__main__":
    unittest.main()
