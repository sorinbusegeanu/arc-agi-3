from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SingleGameRunConfig:
    game_id: str
    runtime_mode: str
    solver_version: str
    output_dir: str | None
    seed: int
    advisor: str
    max_steps: int | None = None
    max_levels: int | None = None
    video: bool = False
    render_terminal: bool = False
    debug: bool = False


@dataclass(frozen=True)
class MultiGameRunConfig:
    game_ids: tuple[str, ...]
    runtime_mode: str
    solver_version: str
    output_dir: str | None
    seed: int
    advisor: str
    max_workers: int
    per_game_timeout_seconds: float
    max_steps: int | None = None
    max_levels: int | None = None
    video: bool = False
    render_terminal: bool = False
    debug: bool = False


@dataclass(frozen=True)
class BenchmarkCliConfig:
    runtime_mode: str
    solver_version: str
    output_dir: str | None
    seed: int
    advisor: str
    active_only: bool
    game_ids: tuple[str, ...] | None
    use_multiprocessing: bool
    max_workers: int
    per_game_timeout_seconds: float
    video: bool = False
    render_terminal: bool = False
    debug: bool = False
