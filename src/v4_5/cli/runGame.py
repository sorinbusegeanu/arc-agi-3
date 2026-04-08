from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from v4_5.benchmark.db.store import utc_now_text
from v4_5.benchmark.runner.resultNormalizer import normalize_runner_output
from v4_5.bootstrap.runtimeFactory import build_runtime_bundle
from v4_5.cli.outputPaths import debug_log_path, resolve_run_output_dir
from v4_5.cli.types import SingleGameRunConfig


def run_single_game(config: SingleGameRunConfig) -> dict[str, Any]:
    output_dir = resolve_run_output_dir(config.output_dir)
    bundle = build_runtime_bundle(
        advisor_name=config.advisor,
        debug=bool(config.debug),
        debug_log_path=str(debug_log_path()) if config.debug else None,
    )
    game_dir = output_dir / config.game_id
    game_dir.mkdir(parents=True, exist_ok=True)
    raw = bundle.game_runner.run_game(
        config.game_id,
        max_steps=config.max_steps,
        max_levels=config.max_levels,
        seed=config.seed,
        capture_video=bool(config.video),
        video_dir=str(game_dir),
        render_terminal=bool(config.render_terminal),
    )
    normalized = normalize_runner_output(raw, created_at=utc_now_text())
    payload = {
        "game_id": config.game_id,
        "runtime_mode": config.runtime_mode,
        "solver_version": config.solver_version,
        "normalized_result": normalized.to_dict(),
        "video_path": raw.get("video_path"),
    }
    path = game_dir / "result.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {"result": payload, "result_path": str(path)}
