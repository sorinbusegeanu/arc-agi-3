from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Sequence

from v7.environment.arc_adapter import ArcGridEnvironment
from v7.environment.runner import ArcGameRunConfig, ArcGameRunResult, run_arc_game


@dataclass(frozen=True, slots=True)
class V7ExperimentConfig:
    games: tuple[str, ...]
    steps_per_game: int = 1000
    epochs: int = 1
    seed: int = 0
    env_root: str | None = None
    commit_every: int = 250
    epsilon: float = 0.10
    op_mode: str = "normal"

    def __post_init__(self) -> None:
        if not self.games:
            raise ValueError("at least one game is required")
        if self.steps_per_game < 1 or self.epochs < 1:
            raise ValueError("steps_per_game and epochs must be positive")


@dataclass(frozen=True, slots=True)
class V7ExperimentResult:
    epochs: int
    games: int
    total_steps: int
    final_generation: int
    final_memories: int
    wins: int
    failures: int
    levels_completed: int
    transfer_trials: int
    runs: tuple[ArcGameRunResult, ...]


def run_experiment(
    root: str | Path,
    config: V7ExperimentConfig,
    *,
    env_factory: Callable[..., ArcGridEnvironment] = ArcGridEnvironment,
) -> V7ExperimentResult:
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    results: list[ArcGameRunResult] = []
    for epoch in range(config.epochs):
        for game_index, game_id in enumerate(config.games):
            result = run_arc_game(
                root_path,
                ArcGameRunConfig(
                    game_id=game_id,
                    steps=config.steps_per_game,
                    seed=config.seed + epoch * len(config.games) + game_index,
                    env_root=config.env_root,
                    commit_every=config.commit_every,
                    epsilon=config.epsilon,
                    restore=bool(results),
                    op_mode=config.op_mode,
                ),
                env_factory=env_factory,
            )
            results.append(result)
    final = results[-1]
    summary = V7ExperimentResult(
        epochs=config.epochs,
        games=len(config.games),
        total_steps=sum(item.steps for item in results),
        final_generation=final.generation,
        final_memories=final.memories,
        wins=sum(item.wins for item in results),
        failures=sum(item.failures for item in results),
        levels_completed=sum(item.levels_completed for item in results),
        transfer_trials=sum(item.transfer_trials for item in results),
        runs=tuple(results),
    )
    (root_path / "experiment_summary.json").write_text(json.dumps(asdict(summary), indent=2, sort_keys=True), encoding="utf-8")
    return summary


def parse_games(values: Sequence[str]) -> tuple[str, ...]:
    result: list[str] = []
    for value in values:
        result.extend(part.strip() for part in str(value).split(",") if part.strip())
    return tuple(dict.fromkeys(result))
