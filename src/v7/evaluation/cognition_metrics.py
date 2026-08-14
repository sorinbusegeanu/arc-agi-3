from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True, slots=True)
class CognitionMetricsSnapshot:
    epochs_observed: int
    solved_game_count_by_epoch: tuple[int, ...]
    repeat_solution_rate: float | None
    solution_retention_rate: float | None
    mean_successful_trajectory_length: float | None
    mean_steps_to_rediscover_solved_level: float | None
    cross_game_transfer_success_rate: float | None
    failure_repetition_rate: float | None
    successful_trajectory_count: int
    repeated_solution_opportunities: int
    retention_opportunities: int
    rediscovery_count: int
    failure_count: int
    repeated_failure_count: int
    selection_modes: dict[str, int]
    development_stages: dict[str, int]

    def as_dict(self) -> dict[str, object]:
        return {
            "epochs_observed": self.epochs_observed,
            "solved_game_count_by_epoch": list(self.solved_game_count_by_epoch),
            "repeat_solution_rate": self.repeat_solution_rate,
            "solution_retention_rate": self.solution_retention_rate,
            "mean_successful_trajectory_length": self.mean_successful_trajectory_length,
            "mean_steps_to_rediscover_solved_level": self.mean_steps_to_rediscover_solved_level,
            "cross_game_transfer_success_rate": self.cross_game_transfer_success_rate,
            "failure_repetition_rate": self.failure_repetition_rate,
            "successful_trajectory_count": self.successful_trajectory_count,
            "repeated_solution_opportunities": self.repeated_solution_opportunities,
            "retention_opportunities": self.retention_opportunities,
            "rediscovery_count": self.rediscovery_count,
            "failure_count": self.failure_count,
            "repeated_failure_count": self.repeated_failure_count,
            "selection_modes": dict(self.selection_modes),
            "development_stages": dict(self.development_stages),
        }


class CognitionMetricsAccumulator:
    """Bounded experiment-level metrics for solution retention and cognition reuse.

    The accumulator stores only aggregate counters and solved-level identities;
    it never retains full episode histories.
    """

    def __init__(self) -> None:
        self._solved_by_epoch: list[set[str]] = []
        self._ever_solved_games: set[str] = set()
        self._repeat_opportunities = 0
        self._repeat_successes = 0
        self._retention_opportunities = 0
        self._retained = 0
        self._seen_success_levels: set[tuple[str, str]] = set()
        self._rediscovery_steps_total = 0
        self._rediscovery_count = 0
        self._successful_trajectory_steps_total = 0
        self._successful_trajectory_count = 0
        self._failure_counts: Counter[tuple[str, int, int]] = Counter()
        self._failure_count = 0
        self._repeated_failure_count = 0
        self._selection_modes: Counter[str] = Counter()
        self._development_stages: Counter[str] = Counter()

    def observe_epoch(self, epoch: int, batches: Iterable[object]) -> None:
        batch_rows = tuple(batches)
        solved = {
            str(batch.game_id)
            for batch in batch_rows
            if int(getattr(batch, "wins", 0) or 0) > 0
        }

        while len(self._solved_by_epoch) < int(epoch):
            self._solved_by_epoch.append(set())
        if len(self._solved_by_epoch) == int(epoch):
            self._solved_by_epoch.append(set(solved))
        else:
            self._solved_by_epoch[int(epoch)] = set(solved)

        if int(epoch) > 0:
            prior_solved = set().union(*self._solved_by_epoch[: int(epoch)])
            self._repeat_opportunities += len(prior_solved)
            self._repeat_successes += len(prior_solved & solved)
            previous = self._solved_by_epoch[int(epoch) - 1]
            self._retention_opportunities += len(previous)
            self._retained += len(previous & solved)
        self._ever_solved_games.update(solved)

        for batch in batch_rows:
            game = str(batch.game_id)
            for trajectory in getattr(batch, "trajectories", ()) or ():
                if not bool(getattr(trajectory, "success", False)):
                    continue
                steps = max(0, int(getattr(trajectory, "steps_to_success", 0) or 0))
                self._successful_trajectory_steps_total += steps
                self._successful_trajectory_count += 1
                key = (game, str(getattr(trajectory, "level_key", "")))
                if key in self._seen_success_levels:
                    self._rediscovery_steps_total += steps
                    self._rediscovery_count += 1
                else:
                    self._seen_success_levels.add(key)

            for row in getattr(batch, "evidence", ()) or ():
                mode = str(getattr(row, "selection_mode", "unknown") or "unknown")
                stage = str(getattr(row, "development_stage", "unknown") or "unknown")
                self._selection_modes[mode] += 1
                self._development_stages[stage] += 1
                if int(getattr(row, "terminal_polarity", 0) or 0) >= 0:
                    continue
                failure_key = (
                    game,
                    int(getattr(row, "context_signature", 0) or 0),
                    int(getattr(row, "action_id", 0) or 0),
                )
                if self._failure_counts[failure_key] > 0:
                    self._repeated_failure_count += 1
                self._failure_counts[failure_key] += 1
                self._failure_count += 1

    def snapshot(
        self,
        *,
        transfer_trials: int = 0,
        transfer_successes: int = 0,
    ) -> CognitionMetricsSnapshot:
        repeat = (
            self._repeat_successes / self._repeat_opportunities
            if self._repeat_opportunities > 0
            else None
        )
        retention = (
            self._retained / self._retention_opportunities
            if self._retention_opportunities > 0
            else None
        )
        mean_length = (
            self._successful_trajectory_steps_total / self._successful_trajectory_count
            if self._successful_trajectory_count > 0
            else None
        )
        rediscovery = (
            self._rediscovery_steps_total / self._rediscovery_count
            if self._rediscovery_count > 0
            else None
        )
        transfer = (
            int(transfer_successes) / int(transfer_trials)
            if int(transfer_trials) > 0
            else None
        )
        failure_repetition = (
            self._repeated_failure_count / self._failure_count
            if self._failure_count > 0
            else None
        )
        return CognitionMetricsSnapshot(
            epochs_observed=len(self._solved_by_epoch),
            solved_game_count_by_epoch=tuple(
                len(values) for values in self._solved_by_epoch
            ),
            repeat_solution_rate=repeat,
            solution_retention_rate=retention,
            mean_successful_trajectory_length=mean_length,
            mean_steps_to_rediscover_solved_level=rediscovery,
            cross_game_transfer_success_rate=transfer,
            failure_repetition_rate=failure_repetition,
            successful_trajectory_count=self._successful_trajectory_count,
            repeated_solution_opportunities=self._repeat_opportunities,
            retention_opportunities=self._retention_opportunities,
            rediscovery_count=self._rediscovery_count,
            failure_count=self._failure_count,
            repeated_failure_count=self._repeated_failure_count,
            selection_modes=dict(sorted(self._selection_modes.items())),
            development_stages=dict(sorted(self._development_stages.items())),
        )


def cross_game_transfer_counts(connection) -> tuple[int, int]:
    row = connection.execute(
        "SELECT COUNT(*), COALESCE(SUM(CASE WHEN success != 0 THEN 1 ELSE 0 END), 0) "
        "FROM transfer_trials WHERE source_game IS NOT NULL AND target_game IS NOT NULL "
        "AND source_game != target_game"
    ).fetchone()
    if row is None:
        return 0, 0
    return int(row[0] or 0), int(row[1] or 0)
