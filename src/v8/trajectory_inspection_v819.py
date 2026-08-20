from __future__ import annotations

import json
import os
import time
from pathlib import Path


_INSTALLED = False
_BASE_WRITE_SUCCESSFUL_TRAJECTORY = None
_BASE_SERVICE_INIT = None
_BASE_INGEST_INBOX_V818 = None
_BASE_RUNTIME_VALIDATION_CALLBACK_V818 = None

_OBSERVED_GAME_ID = ""
_OBSERVED_LEVELS: list[tuple[int, ...]] = []
_OBSERVED_CHAIN_VALID = False
_CAPTURED_SOLUTIONS_FOR_TESTS: list[dict[str, object]] = []


def _reset_observed_capture() -> None:
    global _OBSERVED_GAME_ID, _OBSERVED_LEVELS, _OBSERVED_CHAIN_VALID
    _OBSERVED_GAME_ID = ""
    _OBSERVED_LEVELS = []
    _OBSERVED_CHAIN_VALID = False


def _flatten_levels(levels) -> tuple[int, ...]:
    result: list[int] = []
    for level in levels:
        result.extend(int(value) for value in level)
    return tuple(result)


def _level_payload(levels) -> list[dict[str, object]]:
    return [
        {"level": int(index), "actions": [int(value) for value in actions]}
        for index, actions in enumerate(levels)
    ]


def _observed_solution(row, levels) -> dict[str, object]:
    flat = _flatten_levels(levels)
    return {
        "game_id": str(row.anchor.source_id),
        "trajectory_id": str(row.trajectory_id),
        "source": "observed",
        "terminal_state": "WIN",
        "total_cost": len(flat),
        "levels": _level_payload(levels),
        "attempts": 1,
        "successes": 1,
        "reliability": 1.0,
    }


def _write_complete_observed_solution(row) -> None:
    global _OBSERVED_GAME_ID, _OBSERVED_LEVELS, _OBSERVED_CHAIN_VALID

    game_id = str(row.anchor.source_id)
    prefix = tuple(int(value) for value in row.anchor.prefix_actions)
    actions = tuple(int(value) for value in row.actions)
    terminal_state = str(row.target.terminal_state)
    levels_completed = int(row.target.levels_completed)

    first_level = not prefix and levels_completed <= 1
    if first_level or game_id != _OBSERVED_GAME_ID:
        _OBSERVED_GAME_ID = game_id
        _OBSERVED_LEVELS = []
        _OBSERVED_CHAIN_VALID = bool(first_level)

    expected_prefix = _flatten_levels(_OBSERVED_LEVELS)
    if not _OBSERVED_CHAIN_VALID or prefix != expected_prefix:
        _OBSERVED_LEVELS = []
        _OBSERVED_CHAIN_VALID = False
        if terminal_state == "WIN":
            _reset_observed_capture()
        return

    _OBSERVED_LEVELS.append(actions)
    if terminal_state != "WIN":
        return

    solution = _observed_solution(row, tuple(_OBSERVED_LEVELS))
    root_raw = os.environ.get("ARC_AGI3_V8_TRAJECTORY_ROOT")
    if root_raw:
        from v8 import trajectory_optimizer_v814 as optimizer

        inbox = Path(root_raw) / "solutions_inbox"
        target = inbox / (
            f"{game_id}-{row.trajectory_id}-{os.getpid()}-{time.time_ns()}.json"
        )
        optimizer._atomic_json(target, solution)
    else:
        _CAPTURED_SOLUTIONS_FOR_TESTS.append(solution)
    _reset_observed_capture()


def _write_successful_trajectory_v819(row) -> None:
    _BASE_WRITE_SUCCESSFUL_TRAJECTORY(row)
    _write_complete_observed_solution(row)


def _normalize_levels(raw_levels) -> tuple[tuple[int, ...], ...] | None:
    if not isinstance(raw_levels, list) or not raw_levels:
        return None
    levels: list[tuple[int, ...]] = []
    for index, raw in enumerate(raw_levels):
        if not isinstance(raw, dict):
            return None
        try:
            stored_index = int(raw.get("level", index))
        except (TypeError, ValueError):
            return None
        if stored_index != index:
            return None
        actions = raw.get("actions")
        if not isinstance(actions, list) or not actions:
            return None
        try:
            level = tuple(int(value) for value in actions)
        except (TypeError, ValueError):
            return None
        levels.append(level)
    return tuple(levels)


def _validated_solution_record(raw: object) -> dict[str, object] | None:
    if not isinstance(raw, dict):
        return None
    game_id = str(raw.get("game_id", "")).strip()
    source = str(raw.get("source", ""))
    terminal_state = str(raw.get("terminal_state", ""))
    if not game_id or source not in {"observed", "optimized"} or terminal_state != "WIN":
        return None
    levels = _normalize_levels(raw.get("levels"))
    if levels is None:
        return None
    total_cost = sum(len(level) for level in levels)
    try:
        declared_cost = int(raw.get("total_cost", -1))
    except (TypeError, ValueError):
        return None
    if declared_cost != total_cost:
        return None

    trajectory_id = str(raw.get("trajectory_id", ""))
    variant_id = str(raw.get("variant_id", ""))
    if source == "observed" and not trajectory_id:
        return None
    if source == "optimized" and not variant_id:
        return None

    try:
        attempts = max(1, int(raw.get("attempts", 1)))
        successes = max(0, int(raw.get("successes", 1)))
        reliability = float(raw.get("reliability", successes / attempts))
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    reliability = max(0.0, min(1.0, reliability))

    record: dict[str, object] = {
        "game_id": game_id,
        "source": source,
        "terminal_state": "WIN",
        "total_cost": total_cost,
        "levels": _level_payload(levels),
        "attempts": attempts,
        "successes": successes,
        "reliability": reliability,
    }
    if trajectory_id:
        record["trajectory_id"] = trajectory_id
    if variant_id:
        record["variant_id"] = variant_id
    return record


def _load_best_successful(path: Path) -> dict[str, dict[str, object]]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}
    if not isinstance(raw, dict):
        return {}
    try:
        version = int(raw.get("version", 0))
    except (TypeError, ValueError):
        return {}
    if version != 1:
        return {}
    games = raw.get("games", {})
    if not isinstance(games, dict):
        return {}
    result: dict[str, dict[str, object]] = {}
    for game_id, item in games.items():
        record = _validated_solution_record(item)
        if record is not None and record["game_id"] == str(game_id):
            result[str(game_id)] = record
    return result


def _persist_best_successful(service) -> None:
    from v8 import trajectory_optimizer_v814 as optimizer

    with service._v819_solution_lock:
        payload = {
            "version": 1,
            "games": {
                game: dict(record)
                for game, record in sorted(service._v819_best_successful.items())
            },
        }
    optimizer._atomic_json(service.best_successful_path, payload)


def _is_better_solution(candidate: dict[str, object], prior: dict[str, object] | None) -> bool:
    if prior is None:
        return True
    candidate_cost = int(candidate["total_cost"])
    prior_cost = int(prior["total_cost"])
    if candidate_cost != prior_cost:
        return candidate_cost < prior_cost
    candidate_reliability = float(candidate.get("reliability", 0.0))
    prior_reliability = float(prior.get("reliability", 0.0))
    if abs(candidate_reliability - prior_reliability) > 1e-12:
        return candidate_reliability > prior_reliability
    if str(candidate.get("source", "")) != str(prior.get("source", "")):
        return str(candidate.get("source", "")) == "optimized"
    return False


def _consider_best_solution(service, raw: object) -> bool:
    record = _validated_solution_record(raw)
    if record is None:
        return False
    game_id = str(record["game_id"])
    with service._v819_solution_lock:
        prior = service._v819_best_successful.get(game_id)
        if not _is_better_solution(record, prior):
            return False
        service._v819_best_successful[game_id] = record
    _persist_best_successful(service)
    return True


def _service_init_v819(self, *args, **kwargs) -> None:
    _BASE_SERVICE_INIT(self, *args, **kwargs)
    import threading

    self.solutions_inbox = self.root / "solutions_inbox"
    self.best_successful_path = self.root / "best_successful.json"
    self._v819_solution_lock = threading.RLock()
    self._v819_best_successful = _load_best_successful(self.best_successful_path)
    self.solutions_inbox.mkdir(parents=True, exist_ok=True)
    if not self.best_successful_path.exists():
        _persist_best_successful(self)

    base_callback = self.on_validation
    if base_callback is not None:
        self.on_validation = lambda candidate, result, validated: _service_on_validation_v819(
            self,
            base_callback,
            candidate,
            result,
            validated,
        )


def _ingest_solution_inbox(service) -> None:
    service.solutions_inbox.mkdir(parents=True, exist_ok=True)
    for path in sorted(service.solutions_inbox.glob("*.json"))[:128]:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if _validated_solution_record(raw) is None:
                raise ValueError("invalid complete-solution record")
            _consider_best_solution(service, raw)
        except BaseException as exc:
            service._log(
                "solution_inbox_error",
                path=str(path),
                error=f"{type(exc).__name__}: {exc}",
            )
        finally:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass


def _ingest_inbox_v819(service) -> None:
    _BASE_INGEST_INBOX_V818(service)
    _ingest_solution_inbox(service)


def _optimized_levels(service, candidate, result) -> tuple[tuple[int, ...], ...] | None:
    if str(candidate.source.target.terminal_state) != "WIN":
        return None
    count = int(candidate.source.target.levels_completed)
    if count <= 0:
        return None
    full = tuple(int(value) for value in result.prefix_actions) + tuple(
        int(value) for value in candidate.actions
    )
    with service._v818_validator_lock:
        by_level = dict(service._v818_best_prefixes.get(str(candidate.source.anchor.source_id), {}))
    cumulative: list[tuple[int, ...]] = []
    for boundary in range(count + 1):
        row = by_level.get(boundary)
        if row is None:
            return None
        cumulative.append(tuple(int(value) for value in row))
    if cumulative[0] != () or cumulative[-1] != full:
        return None
    levels: list[tuple[int, ...]] = []
    for previous, current in zip(cumulative, cumulative[1:]):
        if len(current) <= len(previous) or current[: len(previous)] != previous:
            return None
        levels.append(current[len(previous) :])
    return tuple(levels)


def _publish_optimized_solution(service, candidate, result, validated) -> bool:
    if validated is None or str(candidate.source.target.terminal_state) != "WIN":
        return False
    levels = _optimized_levels(service, candidate, result)
    if levels is None:
        return False
    attempts = max(1, int(getattr(validated, "attempts", getattr(result, "attempts", 1))))
    successes = max(0, int(getattr(validated, "successes", getattr(result, "successes", 1))))
    record = {
        "game_id": str(candidate.source.anchor.source_id),
        "variant_id": str(validated.variant_id),
        "source": "optimized",
        "terminal_state": "WIN",
        "total_cost": sum(len(level) for level in levels),
        "levels": _level_payload(levels),
        "attempts": attempts,
        "successes": successes,
        "reliability": successes / attempts,
    }
    return _consider_best_solution(service, record)


def _service_on_validation_v819(service, base_callback, candidate, result, validated) -> None:
    _publish_optimized_solution(service, candidate, result, validated)
    base_callback(candidate, result, validated)


def _runtime_validation_callback_v819(runtime, candidate, result, validated) -> None:
    service = getattr(runtime, "_v814_trajectory_optimizer", None)
    if service is not None:
        _publish_optimized_solution(service, candidate, result, validated)
    _BASE_RUNTIME_VALIDATION_CALLBACK_V818(runtime, candidate, result, validated)


def show_best_trajectory(root: str | Path, game_id: str) -> int:
    from v8.action_targeting_v810 import native_action_id

    game = str(game_id)
    path = Path(root) / "trajectory_optimizer" / "best_successful.json"
    games = _load_best_successful(path)
    record = games.get(game)
    if record is None:
        print(f"game={game} no successful trajectory found", flush=True)
        return 1

    reliability = float(record.get("reliability", 0.0))
    print(
        f"game={game} cost={int(record['total_cost'])} "
        f"source={record['source']} reliability={reliability:.3f}",
        flush=True,
    )
    levels = _normalize_levels(record.get("levels")) or ()
    for index, actions in enumerate(levels):
        formatted = ",".join(f"A{int(native_action_id(action))}" for action in actions)
        print(f"L{index}: {formatted}", flush=True)
    return 0


def _format_best_trajectory_lines(
    game_id: str,
    record: dict[str, object],
) -> tuple[str, ...]:
    from v8.action_targeting_v810 import native_action_id

    reliability = float(record.get("reliability", 0.0))
    lines = [
        f"game={str(game_id)} cost={int(record['total_cost'])} "
        f"source={record['source']} reliability={reliability:.3f}"
    ]
    levels = _normalize_levels(record.get("levels")) or ()
    for index, actions in enumerate(levels):
        formatted = ",".join(f"A{int(native_action_id(action))}" for action in actions)
        lines.append(f"L{index}: {formatted}")
    return tuple(lines)


def _save_best_trajectory_records(
    output_path: str | Path,
    records: tuple[tuple[str, dict[str, object]], ...],
) -> int:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    sections = [
        "\n".join(_format_best_trajectory_lines(game, record))
        for game, record in records
    ]
    payload = "\n\n".join(sections)
    if payload:
        payload += "\n"
    temporary = target.with_name(
        f".{target.name}.{os.getpid()}.{time.time_ns()}.tmp"
    )
    try:
        temporary.write_text(payload, encoding="utf-8")
        os.replace(temporary, target)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    print(f"saved best trajectories games={len(records)} path={target}", flush=True)
    return 0


def save_best_trajectories(root: str | Path, output_path: str | Path) -> int:
    optimizer_root = Path(root) / "trajectory_optimizer"
    games = _load_best_successful(optimizer_root / "best_successful.json")
    records = tuple((game, games[game]) for game in sorted(games))
    return _save_best_trajectory_records(output_path, records)


def install_trajectory_inspection_v819() -> None:
    global _INSTALLED
    global _BASE_WRITE_SUCCESSFUL_TRAJECTORY, _BASE_SERVICE_INIT
    global _BASE_INGEST_INBOX_V818, _BASE_RUNTIME_VALIDATION_CALLBACK_V818
    if _INSTALLED:
        return

    from v8 import trajectory_optimizer_v814 as optimizer
    from v8 import trajectory_optimizer_v818 as v818

    _BASE_WRITE_SUCCESSFUL_TRAJECTORY = optimizer._write_successful_trajectory
    _BASE_SERVICE_INIT = optimizer.TrajectoryOptimizationService.__init__
    _BASE_INGEST_INBOX_V818 = v818._ingest_inbox_v818
    _BASE_RUNTIME_VALIDATION_CALLBACK_V818 = v818._runtime_validation_callback_v818

    optimizer._write_successful_trajectory = _write_successful_trajectory_v819
    optimizer.TrajectoryOptimizationService.__init__ = _service_init_v819
    optimizer.TrajectoryOptimizationService._ingest_inbox = _ingest_inbox_v819
    v818._ingest_inbox_v818 = _ingest_inbox_v819
    v818._runtime_validation_callback_v818 = _runtime_validation_callback_v819
    _INSTALLED = True
