from __future__ import annotations

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FutureTimeoutError

from v4_5.benchmark.runner.processTypes import BenchmarkProcessRequest, BenchmarkProcessResult
from v4_5.benchmark.runner.processWorker import benchmark_worker_entrypoint
from v4_5.benchmark.runner.resultFileStore import validate_result_payload, load_result_file


def build_worker_requests(
    *,
    run_id: str,
    run_label: str,
    solver_version: str,
    runtime_mode: str,
    output_dir: str,
    timeout_seconds: float,
    game_ids: tuple[str, ...],
    seed_builder,
    max_steps: int | None = None,
    max_levels: int | None = None,
    artifacts_dir: str | None = None,
    is_benchmark_run: bool = True,
    capture_video: bool = False,
    render_terminal: bool = False,
    debug: bool = False,
    notes: str | None = None,
) -> tuple[BenchmarkProcessRequest, ...]:
    return tuple(
        BenchmarkProcessRequest(
            run_id=run_id,
            run_label=run_label,
            game_id=game_id,
            solver_version=solver_version,
            runtime_mode=runtime_mode,
            output_dir=output_dir,
            timeout_seconds=timeout_seconds,
            seed=seed_builder(game_id),
            max_steps=max_steps,
            max_levels=max_levels,
            artifacts_dir=artifacts_dir,
            is_benchmark_run=bool(is_benchmark_run),
            capture_video=bool(capture_video),
            render_terminal=bool(render_terminal),
            debug=bool(debug),
            notes=notes,
        )
        for game_id in game_ids
    )


def run_requests_in_parallel(
    requests: tuple[BenchmarkProcessRequest, ...],
    *,
    max_workers: int,
    timeout_seconds: float,
    worker_fn=benchmark_worker_entrypoint,
) -> tuple[BenchmarkProcessResult, ...]:
    if not requests:
        return ()
    ctx = mp.get_context("spawn")
    results: list[BenchmarkProcessResult] = []
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
        future_map = {executor.submit(worker_fn, request): request for request in requests}
        for future, request in sorted(future_map.items(), key=lambda item: item[1].game_id):
            try:
                result = future.result(timeout=timeout_seconds)
            except FutureTimeoutError:
                future.cancel()
                results.append(
                    BenchmarkProcessResult(
                        run_id=request.run_id,
                        game_id=request.game_id,
                        status="timed_out",
                        result_json_path="",
                        started_at="",
                        finished_at="",
                        worker_pid=None,
                        error_message="timed out",
                        traceback_text=None,
                    )
                )
            except Exception as exc:
                results.append(
                    BenchmarkProcessResult(
                        run_id=request.run_id,
                        game_id=request.game_id,
                        status="failed",
                        result_json_path="",
                        started_at="",
                        finished_at="",
                        worker_pid=None,
                        error_message=str(exc),
                        traceback_text=None,
                    )
                )
            else:
                results.append(result)
    return tuple(sorted(results, key=lambda item: item.game_id))


def validate_worker_outputs(results: tuple[BenchmarkProcessResult, ...]) -> tuple[BenchmarkProcessResult, ...]:
    validated = []
    for result in results:
        if result.result_json_path:
            try:
                payload = load_result_file(result.result_json_path)
            except Exception:
                validated.append(BenchmarkProcessResult(**{**result.to_dict(), "status": "missing_result_file"}))
                continue
            if not validate_result_payload(payload):
                validated.append(BenchmarkProcessResult(**{**result.to_dict(), "status": "invalid_result_file"}))
                continue
        validated.append(result)
    return tuple(sorted(validated, key=lambda item: item.game_id))
