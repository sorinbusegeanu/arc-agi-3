from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from v4_5.benchmark.db.store import build_run_id
from v4_5.benchmark.runner.processManager import build_worker_requests, run_requests_in_parallel, validate_worker_outputs
from v4_5.benchmark.runner.resultFileStore import load_result_file
from v4_5.benchmark.runner.seedUtils import deterministic_game_seed
from v4_5.cli.outputPaths import resolve_run_output_dir
from v4_5.cli.types import MultiGameRunConfig


def run_multiple_games(config: MultiGameRunConfig) -> dict[str, Any]:
    run_id = build_run_id()
    output_dir = resolve_run_output_dir(config.output_dir)
    requests = build_worker_requests(
        run_id=run_id,
        run_label="run_games",
        solver_version=config.solver_version,
        runtime_mode=config.runtime_mode,
        output_dir=str(output_dir),
        timeout_seconds=config.per_game_timeout_seconds,
        game_ids=config.game_ids,
        seed_builder=lambda game_id: deterministic_game_seed(run_id=run_id, game_id=game_id, seed_base=config.seed),
        capture_video=bool(config.video),
        render_terminal=bool(config.render_terminal),
        debug=bool(config.debug),
        notes=None,
        max_steps=config.max_steps,
        max_levels=config.max_levels,
        artifacts_dir=str(output_dir),
        is_benchmark_run=False,
    )
    process_results = validate_worker_outputs(
        run_requests_in_parallel(
            requests,
            max_workers=config.max_workers,
            timeout_seconds=config.per_game_timeout_seconds,
        )
    )
    results = []
    for item in process_results:
        payload = None
        if item.result_json_path:
            try:
                payload = load_result_file(item.result_json_path)
            except Exception:
                payload = None
        results.append({"process_result": item.to_dict(), "payload": payload})
    summary = {
        "run_id": run_id,
        "games": list(config.game_ids),
        "completed": [item.game_id for item in process_results if item.status == "completed"],
        "failed": [item.game_id for item in process_results if item.status == "failed"],
        "timed_out": [item.game_id for item in process_results if item.status == "timed_out"],
        "results": results,
    }
    summary_path = output_dir / f"{run_id}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {"summary": summary, "summary_path": str(summary_path)}
