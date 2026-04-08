from __future__ import annotations

from dataclasses import dataclass

from v4_5.bootstrap.runtimeFactory import build_runtime_bundle
from v4_5.benchmark.catalog.gameCatalog import GameCatalog
from v4_5.benchmark.db.store import BenchmarkStore, build_run_id, default_output_dir, utc_now_text
from v4_5.benchmark.reporting.summaryBuilder import build_summary_payload, refresh_best_results, write_game_summary_csv, write_summary_json
from v4_5.benchmark.runner.benchmarkTypes import BenchmarkRunRequest, BenchmarkRunSummary, NormalizedGameResult
from v4_5.benchmark.runner.mergeWorkerResults import merge_worker_results
from v4_5.benchmark.runner.processManager import build_worker_requests, run_requests_in_parallel, validate_worker_outputs
from v4_5.benchmark.runner.runConfig import BenchmarkRunnerConfig
from v4_5.benchmark.runner.seedUtils import deterministic_game_seed
from v4_5.benchmark.runner.resultNormalizer import normalize_runner_output


@dataclass
class BenchmarkRunner:
    store: BenchmarkStore
    catalog: GameCatalog | None = None
    game_runner: object | None = None
    output_dir: str | None = None
    debug: bool = False

    def __post_init__(self) -> None:
        if self.catalog is None:
            self.catalog = GameCatalog(self.store)
        self.catalog.initialize_catalog_if_empty()
        if self.game_runner is None:
            from v4_5.cli.outputPaths import debug_log_path
            self.game_runner = build_runtime_bundle(debug=bool(self.debug), debug_log_path=str(debug_log_path()) if self.debug else None).game_runner

    def _select_game_ids(self, request: BenchmarkRunRequest) -> tuple[tuple[str, ...], tuple[str, ...]]:
        requested_source = request.explicit_game_ids if request.explicit_game_ids is not None else request.game_ids
        if requested_source is not None:
            requested = tuple(requested_source)
            return requested, ()
        active_games = self.catalog.list_active_benchmark_games()
        return tuple(row["game_id"] for row in active_games), ()

    def run(self, request: BenchmarkRunRequest) -> BenchmarkRunSummary:
        started_at = utc_now_text()
        run_id = build_run_id()
        requested_game_ids, skipped_game_ids = self._select_game_ids(request)
        self.store.insert_benchmark_run(
            {
                "run_id": run_id,
                "run_label": request.run_label,
                "started_at": started_at,
                "finished_at": None,
                "solver_version": request.solver_version,
                "runtime_mode": request.runtime_mode,
                "notes": request.notes,
            }
        )
        config = BenchmarkRunnerConfig(
            use_multiprocessing=bool(request.use_multiprocessing),
            max_workers=int(request.max_workers),
            per_game_timeout_seconds=float(request.per_game_timeout_seconds),
            fail_fast=bool(request.fail_fast),
            explicit_game_ids=request.explicit_game_ids,
            output_root=request.output_root,
            runtime_mode=request.runtime_mode,
            solver_version=request.solver_version,
            seed_base=int(request.seed_base),
            debug=bool(request.debug),
        )
        normalized_results: list[NormalizedGameResult] = []
        executed_game_ids: list[str] = []
        process_summary = {
            "use_multiprocessing": bool(config.use_multiprocessing),
            "max_workers": int(config.max_workers),
            "completed_game_count": 0,
            "failed_game_count": 0,
            "timed_out_game_count": 0,
            "merge_failed_game_count": 0,
            "worker_status_summary": [],
        }
        if not config.use_multiprocessing:
            for game_id in requested_game_ids:
                created_at = utc_now_text()
                try:
                    if bool(request.video):
                        serial_video_dir = str((__import__("pathlib").Path(request.output_root or str(self.output_dir or default_output_dir())) / run_id / game_id).resolve())
                        raw_result = self.game_runner.run_game(
                            game_id,
                            max_steps=None,
                            max_levels=None,
                            capture_video=True,
                            video_dir=serial_video_dir,
                            render_terminal=bool(request.render_terminal),
                        )
                    else:
                        raw_result = self.game_runner.run_game(game_id, render_terminal=bool(request.render_terminal))
                    normalized = normalize_runner_output(raw_result, created_at=created_at)
                except Exception as exc:
                    normalized = NormalizedGameResult(
                        game_id=game_id,
                        attempted=False,
                        levels_seen=0,
                        levels_solved=0,
                        total_steps_executed=0,
                        solved_levels_total_steps=0,
                        unsolved_levels_total_steps=0,
                        terminal_success=False,
                        terminal_failure=False,
                        status="runner_error",
                        failure_reason=str(exc),
                        level_results=(),
                        created_at=created_at,
                    )
                self.store.insert_game_result(
                    {
                        "run_id": run_id,
                        "game_id": normalized.game_id,
                        "attempted": normalized.attempted,
                        "levels_seen": normalized.levels_seen,
                        "levels_solved": normalized.levels_solved,
                        "total_steps_executed": normalized.total_steps_executed,
                        "solved_levels_total_steps": normalized.solved_levels_total_steps,
                        "unsolved_levels_total_steps": normalized.unsolved_levels_total_steps,
                        "terminal_success": normalized.terminal_success,
                        "terminal_failure": normalized.terminal_failure,
                        "status": normalized.status,
                        "failure_reason": normalized.failure_reason,
                        "created_at": normalized.created_at,
                    }
                )
                self.store.insert_level_results(
                    {
                        "run_id": run_id,
                        "game_id": normalized.game_id,
                        "level_index": level.level_index,
                        "attempted": level.attempted,
                        "solved": level.solved,
                        "steps_executed": level.steps_executed,
                        "terminal_status": level.terminal_status,
                        "failure_reason": level.failure_reason,
                        "solution_action_count": level.solution_action_count,
                        "created_at": level.created_at,
                    }
                    for level in normalized.level_results
                )
                normalized_results.append(normalized)
                executed_game_ids.append(game_id)
        else:
            output_root = request.output_root or str(self.output_dir or default_output_dir())
            worker_requests = build_worker_requests(
                run_id=run_id,
                run_label=request.run_label,
                solver_version=request.solver_version,
                runtime_mode=request.runtime_mode,
                output_dir=output_root,
                timeout_seconds=config.per_game_timeout_seconds,
                game_ids=requested_game_ids,
                seed_builder=lambda game_id: deterministic_game_seed(run_id=run_id, game_id=game_id, seed_base=config.seed_base),
                capture_video=bool(request.video),
                render_terminal=bool(request.render_terminal),
                debug=bool(request.debug),
                notes=request.notes,
            )
            process_results = validate_worker_outputs(
                run_requests_in_parallel(
                    worker_requests,
                    max_workers=config.max_workers,
                    timeout_seconds=config.per_game_timeout_seconds,
                )
            )
            merge_result = merge_worker_results(self.store, run_id, process_results)
            executed_game_ids.extend(merge_result.merged_game_ids)
            rows = self.store.fetch_all("SELECT * FROM benchmark_game_results WHERE run_id = ? ORDER BY game_id", (run_id,))
            for row in rows:
                level_rows = self.store.fetch_all(
                    "SELECT * FROM benchmark_level_results WHERE run_id = ? AND game_id = ? ORDER BY level_index",
                    (run_id, row["game_id"]),
                )
                normalized_results.append(
                    NormalizedGameResult(
                        game_id=row["game_id"],
                        attempted=bool(row["attempted"]),
                        levels_seen=int(row["levels_seen"]),
                        levels_solved=int(row["levels_solved"]),
                        total_steps_executed=int(row["total_steps_executed"]),
                        solved_levels_total_steps=int(row["solved_levels_total_steps"]),
                        unsolved_levels_total_steps=int(row["unsolved_levels_total_steps"]),
                        terminal_success=bool(row["terminal_success"]),
                        terminal_failure=bool(row["terminal_failure"]),
                        status=str(row["status"]),
                        failure_reason=row.get("failure_reason"),
                        level_results=tuple(),
                        created_at=str(row["created_at"]),
                    )
                )
            process_summary = {
                "use_multiprocessing": True,
                "max_workers": config.max_workers,
                "completed_game_count": len(merge_result.merged_game_ids),
                "failed_game_count": len(merge_result.failed_game_ids),
                "timed_out_game_count": len(merge_result.timed_out_game_ids),
                "merge_failed_game_count": len(merge_result.merge_failed_game_ids),
                "worker_status_summary": [item.to_dict() for item in process_results],
            }
        finished_at = utc_now_text()
        self.store.finalize_benchmark_run(run_id, finished_at)
        if not config.use_multiprocessing:
            refresh_best_results(self.store, run_id)
        payload = build_summary_payload(self.store, run_id, extra_summary=process_summary)
        summary_path = write_summary_json(payload, output_dir=self.output_dir or default_output_dir(), run_id=run_id)
        csv_path = None
        if request.output_csv:
            csv_path = write_game_summary_csv(payload, output_dir=self.output_dir or default_output_dir(), run_id=run_id)
        return BenchmarkRunSummary(
            run_id=run_id,
            run_label=request.run_label,
            started_at=started_at,
            finished_at=finished_at,
            solver_version=request.solver_version,
            runtime_mode=request.runtime_mode,
            requested_game_ids=requested_game_ids,
            executed_game_ids=tuple(executed_game_ids),
            skipped_game_ids=skipped_game_ids,
            game_results=tuple(normalized_results),
            summary_path=str(summary_path),
            csv_path=None if csv_path is None else str(csv_path),
        )
