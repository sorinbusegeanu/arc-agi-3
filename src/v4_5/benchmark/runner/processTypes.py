from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class BenchmarkProcessRequest:
    run_id: str
    run_label: str
    game_id: str
    solver_version: str
    runtime_mode: str
    output_dir: str
    timeout_seconds: float
    seed: int
    max_steps: int | None = None
    max_levels: int | None = None
    artifacts_dir: str | None = None
    is_benchmark_run: bool = True
    capture_video: bool = False
    render_terminal: bool = False
    debug: bool = False
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BenchmarkProcessResult:
    run_id: str
    game_id: str
    status: str
    result_json_path: str
    started_at: str
    finished_at: str
    worker_pid: int | None
    error_message: str | None = None
    traceback_text: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BenchmarkProcessFailure:
    run_id: str
    game_id: str
    status: str
    error_message: str | None
    traceback_text: str | None
    result_json_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BenchmarkMergeResult:
    run_id: str
    merged_game_ids: tuple[str, ...] = ()
    invalid_game_ids: tuple[str, ...] = ()
    failed_game_ids: tuple[str, ...] = ()
    timed_out_game_ids: tuple[str, ...] = ()
    missing_result_files: tuple[str, ...] = ()
    merge_failed_game_ids: tuple[str, ...] = ()
    inserted_game_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
