from __future__ import annotations

"""v8.68 make best-trajectory export agree with verified-success metrics."""

import json
from pathlib import Path


_INSTALLED = False
_BASE_SAVE_BEST_TRAJECTORIES = None


def _latest_verified_run(root: str | Path) -> Path | None:
    verified_root = Path(root) / "verified_success"
    if not verified_root.is_dir():
        return None
    candidates: list[tuple[int, str, Path]] = []
    for path in verified_root.glob("run-*"):
        if not path.is_dir():
            continue
        started_ns = 0
        try:
            raw = json.loads((path / "run.json").read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                started_ns = max(0, int(raw.get("started_ns", 0) or 0))
        except (OSError, ValueError, TypeError):
            pass
        if started_ns <= 0:
            try:
                started_ns = int(path.stat().st_mtime_ns)
            except OSError:
                continue
        candidates.append((started_ns, path.name, path))
    if not candidates:
        return None
    return max(candidates, key=lambda item: (item[0], item[1]))[2]


def _verified_win_events(run_root: Path | None) -> tuple[dict[str, object], ...]:
    if run_root is None:
        return ()
    event_root = run_root / "events"
    if not event_root.is_dir():
        return ()
    rows: list[dict[str, object]] = []
    for path in sorted(event_root.glob("*.json")):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            continue
        if not isinstance(raw, dict):
            continue
        try:
            schema_version = int(raw.get("schema_version", 0) or 0)
        except (TypeError, ValueError):
            continue
        if schema_version != 1 or str(raw.get("terminal_state", "")).upper() != "WIN":
            continue
        game_id = str(raw.get("game_id", "")).strip()
        actions = raw.get("actions")
        if not game_id or not isinstance(actions, list):
            continue
        try:
            action_values = [int(value) for value in actions]
            recorded_ns = max(0, int(raw.get("recorded_ns", 0) or 0))
        except (TypeError, ValueError):
            continue
        row = dict(raw)
        row["game_id"] = game_id
        row["actions"] = action_values
        row["recorded_ns"] = recorded_ns
        rows.append(row)
    return tuple(rows)


def _best_verified_wins(root: str | Path) -> dict[str, dict[str, object]]:
    best: dict[str, dict[str, object]] = {}
    for row in _verified_win_events(_latest_verified_run(root)):
        game = str(row["game_id"])
        prior = best.get(game)
        candidate_key = (
            len(row["actions"]),
            int(row.get("recorded_ns", 0) or 0),
            str(row.get("trajectory_id", "")),
        )
        if prior is None:
            best[game] = row
            continue
        prior_key = (
            len(prior["actions"]),
            int(prior.get("recorded_ns", 0) or 0),
            str(prior.get("trajectory_id", "")),
        )
        if candidate_key < prior_key:
            best[game] = row
    return best


def _verified_export_record(row: dict[str, object]) -> dict[str, object]:
    actions = tuple(int(value) for value in row.get("actions", ()))
    return {
        "game_id": str(row["game_id"]),
        "trajectory_id": str(row.get("trajectory_id", "verified")) or "verified",
        "source": "verified",
        "terminal_state": "WIN",
        "total_cost": len(actions),
        "levels": [{"level": 0, "actions": list(actions)}],
        "attempts": 1,
        "successes": 1,
        "reliability": 1.0,
    }


def save_best_trajectories_v868(root: str | Path, output_path: str | Path) -> int:
    from v8 import trajectory_inspection_v819 as inspection

    verified = _best_verified_wins(root)
    if not verified:
        return _BASE_SAVE_BEST_TRAJECTORIES(root, output_path)

    optimizer_root = Path(root) / "trajectory_optimizer"
    optimizer_games = inspection._load_best_successful(
        optimizer_root / "best_successful.json"
    )
    records: list[tuple[str, dict[str, object]]] = []
    for game in sorted(verified):
        record = optimizer_games.get(game)
        if record is None:
            record = _verified_export_record(verified[game])
        records.append((game, record))
    return inspection._save_best_trajectory_records(output_path, tuple(records))


def install_verified_trajectory_export_v868() -> None:
    global _INSTALLED, _BASE_SAVE_BEST_TRAJECTORIES
    if _INSTALLED:
        return
    from v8 import trajectory_inspection_v819 as inspection

    _BASE_SAVE_BEST_TRAJECTORIES = inspection.save_best_trajectories
    inspection.save_best_trajectories = save_best_trajectories_v868
    _INSTALLED = True
