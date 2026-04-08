from __future__ import annotations

from rl_v1.metrics.metric_keys import (
    GAME_WIN_RATE,
    LEVEL_COMPLETION_RATE,
    MEAN_LEVELS_REACHED,
    MEAN_STEPS_PER_COMPLETED_LEVEL,
)


class MultiLevelMetricAccumulator:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.games_started = 0
        self.games_won = 0
        self.levels_entered = 0
        self.levels_completed = 0
        self.sum_levels_reached = 0
        self.completed_level_steps = 0
        self.completed_level_count = 0

    def on_game_start(self) -> None:
        self.games_started += 1

    def on_level_enter(self, level_index: int) -> None:
        _ = level_index
        self.levels_entered += 1

    def on_level_complete(self, level_index: int, steps_in_level: int) -> None:
        _ = level_index
        self.levels_completed += 1
        self.completed_level_steps += int(steps_in_level)
        self.completed_level_count += 1

    def on_game_end(self, won: bool, deepest_level_reached: int) -> None:
        if won:
            self.games_won += 1
        self.sum_levels_reached += int(deepest_level_reached)

    def compute(self) -> dict:
        return {
            GAME_WIN_RATE: self.games_won / self.games_started if self.games_started else 0.0,
            LEVEL_COMPLETION_RATE: self.levels_completed / self.levels_entered if self.levels_entered else 0.0,
            MEAN_LEVELS_REACHED: self.sum_levels_reached / self.games_started if self.games_started else 0.0,
            MEAN_STEPS_PER_COMPLETED_LEVEL: self.completed_level_steps / self.completed_level_count if self.completed_level_count else 0.0,
        }
