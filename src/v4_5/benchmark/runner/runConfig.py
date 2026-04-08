from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BenchmarkRunnerConfig:
    use_multiprocessing: bool = False
    max_workers: int = 1
    per_game_timeout_seconds: float = 60.0
    fail_fast: bool = False
    explicit_game_ids: tuple[str, ...] | None = None
    output_root: str | None = None
    runtime_mode: str = "offline"
    solver_version: str = "v4.5"
    seed_base: int = 0
    max_steps: int | None = None
    max_levels: int | None = None
    debug: bool = False

    def __post_init__(self) -> None:
        if int(self.max_workers) < 1:
            raise ValueError("max_workers must be >= 1")
