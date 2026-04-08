from __future__ import annotations

import logging
import os
import traceback
from pathlib import Path
from time import sleep

from v4_5.bootstrap.runtimeFactory import build_runtime_bundle
from v4_5.benchmark.db.store import utc_now_text
from v4_5.benchmark.runner.processTypes import BenchmarkProcessRequest, BenchmarkProcessResult
from v4_5.benchmark.runner.resultFileStore import atomic_write_result_file
from v4_5.benchmark.runner.resultNormalizer import normalize_runner_output
from v4_5.cli.outputPaths import debug_log_path


def benchmark_worker_entrypoint(request: BenchmarkProcessRequest) -> BenchmarkProcessResult:
    started_at = utc_now_text()
    game_output_dir = Path(request.output_dir) / request.game_id
    game_output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = str(game_output_dir / f"{request.run_id}__{request.game_id}.json")
    previous_disable = logging.root.manager.disable
    try:
        if not request.debug:
            logging.disable(logging.CRITICAL)
        raw = _run_single_game(request)
        normalized = normalize_runner_output(raw, created_at=started_at)
        finished_at = utc_now_text()
        payload = {
            "schema_version": "v4.5.benchmark.process",
            "run_id": request.run_id,
            "game_id": request.game_id,
            "status": "completed",
            "worker_pid": os.getpid(),
            "started_at": started_at,
            "finished_at": finished_at,
            "game_result": normalized.to_dict(),
            "level_results": [item.to_dict() for item in normalized.level_results],
            "video_path": raw.get("video_path"),
            "error_message": None,
            "traceback_text": None,
        }
        final_path = atomic_write_result_file(output_dir=game_output_dir, run_id=request.run_id, game_id=request.game_id, payload=payload)
        return BenchmarkProcessResult(
            run_id=request.run_id,
            game_id=request.game_id,
            status="completed",
            result_json_path=str(final_path),
            started_at=started_at,
            finished_at=finished_at,
            worker_pid=os.getpid(),
            error_message=None,
            traceback_text=None,
        )
    except Exception as exc:
        finished_at = utc_now_text()
        tb = traceback.format_exc()
        payload = {
            "schema_version": "v4.5.benchmark.process",
            "run_id": request.run_id,
            "game_id": request.game_id,
            "status": "failed",
            "worker_pid": os.getpid(),
            "started_at": started_at,
            "finished_at": finished_at,
            "game_result": None,
            "level_results": [],
            "video_path": None,
            "error_message": str(exc),
            "traceback_text": tb,
        }
        final_path = atomic_write_result_file(output_dir=game_output_dir, run_id=request.run_id, game_id=request.game_id, payload=payload)
        return BenchmarkProcessResult(
            run_id=request.run_id,
            game_id=request.game_id,
            status="failed",
            result_json_path=str(final_path),
            started_at=started_at,
            finished_at=finished_at,
            worker_pid=os.getpid(),
            error_message=str(exc),
            traceback_text=tb,
        )
    finally:
        logging.disable(previous_disable)


def _run_single_game(request: BenchmarkProcessRequest) -> dict:
    if request.runtime_mode == "mock_success":
        result = {
            "game_id": request.game_id,
            "attempted": True,
            "stop_reason": "terminal_win",
            "steps_executed": 4,
            "failure_reason": None,
            "levels_completed_start": 0,
            "levels_completed_end": 1,
            "win_levels": 1,
            "step_records": [
                {
                    "action_executed": True,
                    "pre_levels_completed": 0,
                    "post_levels_completed": 1,
                    "levels_completed_delta": 1,
                    "terminal_status": "success",
                }
            ],
        }
        if request.capture_video:
            result["video_path"] = str((Path(request.output_dir) / request.game_id / "episode.mp4").resolve())
        return result
    if request.runtime_mode == "mock_failure":
        raise RuntimeError(f"mock failure for {request.game_id}")
    if request.runtime_mode == "mock_sleep":
        sleep(max(0.0, float(request.timeout_seconds) + 0.2))
        return {
            "game_id": request.game_id,
            "attempted": True,
            "stop_reason": "timeout_like",
            "steps_executed": 0,
            "failure_reason": "slept",
            "levels_completed_start": 0,
            "levels_completed_end": 0,
            "win_levels": 1,
            "step_records": [],
        }
    bundle = build_runtime_bundle(
        debug=bool(request.debug),
        debug_log_path=str(debug_log_path()) if request.debug else None,
    )
    return bundle.game_runner.run_game(
        request.game_id,
        max_steps=request.max_steps,
        max_levels=request.max_levels,
        seed=request.seed,
        capture_video=bool(request.capture_video),
        video_dir=str((Path(request.artifacts_dir or request.output_dir) / request.game_id).resolve()),
        render_terminal=bool(request.render_terminal),
    )
