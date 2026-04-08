from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def result_file_name(*, run_id: str, game_id: str) -> str:
    return f"{run_id}.json"


def result_file_path(*, output_dir: str | Path, run_id: str, game_id: str) -> Path:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    return root / result_file_name(run_id=run_id, game_id=game_id)


def atomic_write_result_file(*, output_dir: str | Path, run_id: str, game_id: str, payload: dict[str, Any]) -> Path:
    final_path = result_file_path(output_dir=output_dir, run_id=run_id, game_id=game_id)
    tmp_path = final_path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    tmp_path.replace(final_path)
    return final_path


def load_result_file(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_result_payload(payload: dict[str, Any]) -> bool:
    required = (
        "schema_version",
        "run_id",
        "game_id",
        "status",
        "worker_pid",
        "started_at",
        "finished_at",
        "game_result",
        "level_results",
        "error_message",
        "traceback_text",
    )
    return all(key in payload for key in required)
