from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class BenchmarkRunRequest:
    run_label: str
    solver_version: str
    runtime_mode: str
    game_ids: tuple[str, ...] | None = None
    notes: str | None = None
    output_csv: bool = True
    use_multiprocessing: bool = False
    max_workers: int = 1
    per_game_timeout_seconds: float = 60.0
    fail_fast: bool = False
    explicit_game_ids: tuple[str, ...] | None = None
    output_root: str | None = None
    seed_base: int = 0
    video: bool = False
    render_terminal: bool = False
    debug: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PerGameBenchmarkRequest:
    game_id: str
    title: str
    family: str
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class NormalizedLevelResult:
    game_id: str
    level_index: int
    attempted: bool
    solved: bool
    steps_executed: int
    terminal_status: str
    failure_reason: str | None
    solution_action_count: int | None
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class NormalizedGameResult:
    game_id: str
    attempted: bool
    levels_seen: int
    levels_solved: int
    total_steps_executed: int
    solved_levels_total_steps: int
    unsolved_levels_total_steps: int
    terminal_success: bool
    terminal_failure: bool
    status: str
    failure_reason: str | None
    level_results: tuple[NormalizedLevelResult, ...] = field(default_factory=tuple)
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "level_results": [item.to_dict() for item in self.level_results],
        }


@dataclass(frozen=True)
class BenchmarkRunSummary:
    run_id: str
    run_label: str
    started_at: str
    finished_at: str
    solver_version: str
    runtime_mode: str
    requested_game_ids: tuple[str, ...]
    executed_game_ids: tuple[str, ...]
    skipped_game_ids: tuple[str, ...]
    game_results: tuple[NormalizedGameResult, ...]
    summary_path: str | None = None
    csv_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "run_label": self.run_label,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "solver_version": self.solver_version,
            "runtime_mode": self.runtime_mode,
            "requested_game_ids": list(self.requested_game_ids),
            "executed_game_ids": list(self.executed_game_ids),
            "skipped_game_ids": list(self.skipped_game_ids),
            "game_results": [item.to_dict() for item in self.game_results],
            "summary_path": self.summary_path,
            "csv_path": self.csv_path,
        }
