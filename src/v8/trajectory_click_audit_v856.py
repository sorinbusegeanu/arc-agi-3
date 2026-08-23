from __future__ import annotations

"""v8.56 click trajectory audit and inspection.

Successful trajectories must preserve the executable action token for replay while
also exposing the concrete click coordinates that were executed. Exact-coordinate
ACTION6 tokens are already reversible; structural click tokens are location-
independent and therefore need an observational audit captured at execution time.

This layer is observational only. It does not change action identity, optimizer
selection, validation, replay, or canonical memory mutation.
"""

import os
import time
from pathlib import Path


_INSTALLED = False
_BASE_ENV_STEP = None
_BASE_RESET_CAPTURE = None
_BASE_RESET_OBSERVED_CAPTURE = None
_BASE_VALIDATED_SOLUTION_RECORD = None

_ACTION_AUDIT_HISTORY: list[dict[str, object]] = []
_OBSERVED_LEVEL_AUDITS: list[tuple[dict[str, object], ...]] = []


def _fallback_action_audit(token: int) -> dict[str, object]:
    from v8 import action_targeting_v810 as targeting

    value = int(token)
    native = int(targeting.native_action_id(value))
    row: dict[str, object] = {
        "action_token": value,
        "native_action": native,
    }
    payload = targeting._legacy_coordinate_payload(value)
    if payload is not None:
        row.update(
            {
                "x": int(payload["x"]),
                "y": int(payload["y"]),
                "target_kind": "exact_coordinate",
            }
        )
    elif targeting.is_structural_click_token(value):
        row["target_kind"] = "structural_unresolved"
    elif native == 6:
        row["target_kind"] = "unresolved_click"
    return row


def describe_executed_action(env, token: int) -> dict[str, object]:
    """Describe the click target that the active ACTION6 wrapper will execute."""
    from v8 import action_targeting_v810 as targeting

    value = int(token)
    row = _fallback_action_audit(value)
    native = int(row["native_action"])
    if native != 6:
        return row

    # Exact-coordinate clicks carry their own reversible coordinates.
    if "x" in row and "y" in row:
        return row

    target = None
    if targeting.is_structural_click_token(value):
        target = getattr(env, "_v810_click_targets", {}).get(value)
        if target is None:
            targets = targeting.structural_click_targets(
                env._last_grid,
                last_changed=getattr(env, "_v810_last_changed", ()),
            )
            target = next((item for item in targets if int(item.token) == value), None)
    elif value == 6:
        targets = targeting.structural_click_targets(
            env._last_grid,
            last_changed=getattr(env, "_v810_last_changed", ()),
        )
        target = targets[0] if targets else None

    if target is not None:
        row.update(
            {
                "x": int(target.x),
                "y": int(target.y),
                "target_kind": str(target.kind),
            }
        )
    return row


def _env_step_v856(self, action):
    from v8 import trajectory_optimizer_v814 as optimizer

    capture = bool(getattr(optimizer, "_CAPTURE_ACTIVE", False))
    audit = describe_executed_action(self, int(action))
    appended = False
    if capture:
        _ACTION_AUDIT_HISTORY.append(dict(audit))
        appended = True
    self._v856_last_action_audit = dict(audit)

    try:
        result = _BASE_ENV_STEP(self, action)
    except BaseException:
        if appended and _ACTION_AUDIT_HISTORY:
            _ACTION_AUDIT_HISTORY.pop()
        raise

    if capture:
        state = str(getattr(self, "last_outcome_state", ""))
        reset_boundary = bool(getattr(self, "last_step_was_reset_boundary", False))
        # v8.14/v8.19 trajectory publication runs inside _BASE_ENV_STEP. Clear only
        # after it has consumed the audit for the terminal/resetting action.
        if state in {"WIN", "GAME_OVER"} or reset_boundary:
            _ACTION_AUDIT_HISTORY.clear()
    return result


def _reset_capture_v856(job=None) -> None:
    _ACTION_AUDIT_HISTORY.clear()
    _OBSERVED_LEVEL_AUDITS.clear()
    _BASE_RESET_CAPTURE(job)


def _reset_observed_capture_v856() -> None:
    _OBSERVED_LEVEL_AUDITS.clear()
    _BASE_RESET_OBSERVED_CAPTURE()


def _audit_for_segment(row) -> tuple[dict[str, object], ...]:
    prefix = tuple(int(value) for value in row.anchor.prefix_actions)
    actions = tuple(int(value) for value in row.actions)
    start = len(prefix)
    stop = start + len(actions)
    if stop <= len(_ACTION_AUDIT_HISTORY):
        candidate = tuple(dict(item) for item in _ACTION_AUDIT_HISTORY[start:stop])
        if tuple(int(item.get("action_token", -1)) for item in candidate) == actions:
            return candidate
    return tuple(_fallback_action_audit(action) for action in actions)


def _level_payload_with_audit(
    levels: tuple[tuple[int, ...], ...],
    audits: tuple[tuple[dict[str, object], ...], ...],
) -> list[dict[str, object]]:
    payload: list[dict[str, object]] = []
    for index, actions in enumerate(levels):
        level: dict[str, object] = {
            "level": int(index),
            "actions": [int(value) for value in actions],
        }
        if index < len(audits) and len(audits[index]) == len(actions):
            level["action_audit"] = [dict(item) for item in audits[index]]
        payload.append(level)
    return payload


def _write_complete_observed_solution_v856(row) -> None:
    from v8 import trajectory_inspection_v819 as inspection
    from v8 import trajectory_optimizer_v814 as optimizer

    game_id = str(row.anchor.source_id)
    prefix = tuple(int(value) for value in row.anchor.prefix_actions)
    actions = tuple(int(value) for value in row.actions)
    terminal_state = str(row.target.terminal_state)
    levels_completed = int(row.target.levels_completed)

    first_level = not prefix and levels_completed <= 1
    if first_level or game_id != inspection._OBSERVED_GAME_ID:
        inspection._OBSERVED_GAME_ID = game_id
        inspection._OBSERVED_LEVELS = []
        inspection._OBSERVED_CHAIN_VALID = bool(first_level)
        _OBSERVED_LEVEL_AUDITS.clear()

    expected_prefix = inspection._flatten_levels(inspection._OBSERVED_LEVELS)
    if not inspection._OBSERVED_CHAIN_VALID or prefix != expected_prefix:
        inspection._OBSERVED_LEVELS = []
        inspection._OBSERVED_CHAIN_VALID = False
        _OBSERVED_LEVEL_AUDITS.clear()
        if terminal_state == "WIN":
            inspection._reset_observed_capture()
        return

    inspection._OBSERVED_LEVELS.append(actions)
    _OBSERVED_LEVEL_AUDITS.append(_audit_for_segment(row))
    if terminal_state != "WIN":
        return

    levels = tuple(inspection._OBSERVED_LEVELS)
    audits = tuple(_OBSERVED_LEVEL_AUDITS)
    flat = inspection._flatten_levels(levels)
    solution = {
        "game_id": game_id,
        "trajectory_id": str(row.trajectory_id),
        "source": "observed",
        "terminal_state": "WIN",
        "total_cost": len(flat),
        "levels": _level_payload_with_audit(levels, audits),
        "attempts": 1,
        "successes": 1,
        "reliability": 1.0,
    }

    root_raw = os.environ.get("ARC_AGI3_V8_TRAJECTORY_ROOT")
    if root_raw:
        inbox = Path(root_raw) / "solutions_inbox"
        target = inbox / (
            f"{game_id}-{row.trajectory_id}-{os.getpid()}-{time.time_ns()}.json"
        )
        optimizer._atomic_json(target, solution)
    else:
        inspection._CAPTURED_SOLUTIONS_FOR_TESTS.append(solution)
    inspection._reset_observed_capture()


def _valid_level_audit(raw_level: object, actions: tuple[int, ...]):
    if not isinstance(raw_level, dict):
        return None
    raw = raw_level.get("action_audit")
    if raw is None:
        return None
    if not isinstance(raw, list) or len(raw) != len(actions):
        return None

    result: list[dict[str, object]] = []
    for action, item in zip(actions, raw, strict=True):
        if not isinstance(item, dict):
            return None
        try:
            token = int(item.get("action_token", action))
            native = int(item.get("native_action", -1))
        except (TypeError, ValueError):
            return None
        if token != int(action):
            return None
        normalized: dict[str, object] = {
            "action_token": token,
            "native_action": native,
        }
        if "x" in item or "y" in item:
            try:
                x = int(item["x"])
                y = int(item["y"])
            except (KeyError, TypeError, ValueError):
                return None
            normalized["x"] = x
            normalized["y"] = y
        if item.get("target_kind") is not None:
            normalized["target_kind"] = str(item["target_kind"])
        result.append(normalized)
    return result


def _validated_solution_record_v856(raw: object):
    record = _BASE_VALIDATED_SOLUTION_RECORD(raw)
    if record is None or not isinstance(raw, dict):
        return record

    raw_levels = raw.get("levels")
    normalized_levels = record.get("levels")
    if not isinstance(raw_levels, list) or not isinstance(normalized_levels, list):
        return record

    for index, level in enumerate(normalized_levels):
        if index >= len(raw_levels) or not isinstance(level, dict):
            break
        actions_raw = level.get("actions")
        if not isinstance(actions_raw, list):
            continue
        actions = tuple(int(value) for value in actions_raw)
        audit = _valid_level_audit(raw_levels[index], actions)
        if audit is not None:
            level["action_audit"] = audit
    return record


def _format_click_detail(token: int, audit: dict[str, object] | None = None) -> str | None:
    from v8 import action_targeting_v810 as targeting

    value = int(token)
    native = int(targeting.native_action_id(value))
    if native != 6:
        return None

    row = audit if isinstance(audit, dict) else None
    if row is not None:
        try:
            same_token = int(row.get("action_token", value)) == value
        except (TypeError, ValueError):
            same_token = False
        if same_token and "x" in row and "y" in row:
            try:
                x = int(row["x"])
                y = int(row["y"])
            except (TypeError, ValueError):
                pass
            else:
                kind = str(row.get("target_kind", ""))
                if kind and kind not in {"exact_coordinate", "unresolved_click"}:
                    return f"A6({x},{y})[{kind}]"
                return f"A6({x},{y})"

    payload = targeting._legacy_coordinate_payload(value)
    if payload is not None:
        return f"A6({int(payload['x'])},{int(payload['y'])})"
    if targeting.is_structural_click_token(value):
        return f"A6[structural=0x{value:x}]"
    return "A6[unresolved]"


def _format_best_trajectory_lines_v856(
    game_id: str,
    record: dict[str, object],
) -> tuple[str, ...]:
    from v8 import action_targeting_v810 as targeting
    from v8 import trajectory_inspection_v819 as inspection

    reliability = float(record.get("reliability", 0.0))
    lines = [
        f"game={str(game_id)} cost={int(record['total_cost'])} "
        f"source={record['source']} reliability={reliability:.3f}"
    ]
    levels = inspection._normalize_levels(record.get("levels")) or ()
    raw_levels = record.get("levels")
    raw_levels = raw_levels if isinstance(raw_levels, list) else []

    for index, actions in enumerate(levels):
        formatted = ",".join(
            f"A{int(targeting.native_action_id(action))}" for action in actions
        )
        lines.append(f"L{index}: {formatted}")

        raw_level = raw_levels[index] if index < len(raw_levels) else None
        audit_rows = raw_level.get("action_audit") if isinstance(raw_level, dict) else None
        audit_rows = audit_rows if isinstance(audit_rows, list) else []
        details = []
        for position, action in enumerate(actions):
            audit = audit_rows[position] if position < len(audit_rows) else None
            detail = _format_click_detail(action, audit if isinstance(audit, dict) else None)
            if detail is not None:
                details.append(f"{position}={detail}")
        if details:
            lines.append(f"L{index} clicks: " + ",".join(details))
    return tuple(lines)


def _show_best_trajectory_v856(root: str | Path, game_id: str) -> int:
    from v8 import trajectory_inspection_v819 as inspection
    from v8 import trajectory_inspection_v819_fixups as fixups

    game = str(game_id)
    record = fixups._best_visible_solution(root, game)
    if record is None:
        print(f"game={game} no successful trajectory found", flush=True)
        return 1
    for line in _format_best_trajectory_lines_v856(game, record):
        print(line, flush=True)
    return 0


def install_trajectory_click_audit_v856() -> None:
    global _INSTALLED
    global _BASE_ENV_STEP, _BASE_RESET_CAPTURE, _BASE_RESET_OBSERVED_CAPTURE
    global _BASE_VALIDATED_SOLUTION_RECORD
    if _INSTALLED:
        return

    from v7.environment.arc_adapter import ArcGridEnvironment
    from v8 import trajectory_inspection_v819 as inspection
    from v8 import trajectory_optimizer_v814 as optimizer

    _BASE_ENV_STEP = ArcGridEnvironment.step
    _BASE_RESET_CAPTURE = optimizer._reset_capture
    _BASE_RESET_OBSERVED_CAPTURE = inspection._reset_observed_capture
    _BASE_VALIDATED_SOLUTION_RECORD = inspection._validated_solution_record

    ArcGridEnvironment.step = _env_step_v856
    optimizer._reset_capture = _reset_capture_v856
    inspection._reset_observed_capture = _reset_observed_capture_v856
    inspection._write_complete_observed_solution = _write_complete_observed_solution_v856
    inspection._validated_solution_record = _validated_solution_record_v856
    inspection._format_best_trajectory_lines = _format_best_trajectory_lines_v856
    inspection.show_best_trajectory = _show_best_trajectory_v856
    _INSTALLED = True
