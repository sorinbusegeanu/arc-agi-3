from __future__ import annotations

"""Small integrity repairs for v8.49 reporting.

Keep productive click-target identity, compute exact-coordinate coverage rather than
structural-target coverage, and let synthetic/restored coordinator measurements
participate in the same data-derived allocation baseline used by live actor events.
"""

import math
import statistics


_INSTALLED = False
_BASE_CLICK_STEP = None
_BASE_RECORD_EPISODE_KIND = None
_BASE_EVENT_PAYLOAD = None
_BASE_MARK_REPORTED = None
_BASE_EMPTY_AGGREGATE = None
_BASE_MERGE_EVENT = None
_BASE_GAME_ROW = None
_BASE_SNAPSHOT = None


def _click_step_v849_fix(self, action):
    self._v849_last_action_token = int(action)
    return _BASE_CLICK_STEP(self, action)


def _ensure_productive_sets(env) -> None:
    if not hasattr(env, "_v849_productive_clicks"):
        env._v849_productive_clicks = set()
    if not hasattr(env, "_v849_reported_productive_clicks"):
        env._v849_reported_productive_clicks = set()


def _record_episode_kind_fix(env, kind: str, *, productive: bool, level_advanced: bool) -> None:
    _BASE_RECORD_EPISODE_KIND(
        env,
        kind,
        productive=productive,
        level_advanced=level_advanced,
    )
    if str(kind) != "click" or not bool(productive):
        return
    _ensure_productive_sets(env)
    token = getattr(env, "_v849_last_action_token", None)
    if token is not None:
        env._v849_productive_clicks.add(int(token))


def _event_payload_fix(env):
    _ensure_productive_sets(env)
    payload = _BASE_EVENT_PAYLOAD(env)
    if payload is None:
        return None
    payload["productive_click_targets"] = sorted(
        int(value)
        for value in env._v849_productive_clicks - env._v849_reported_productive_clicks
    )
    return payload


def _mark_reported_fix(env) -> None:
    _ensure_productive_sets(env)
    _BASE_MARK_REPORTED(env)
    env._v849_reported_productive_clicks.update(env._v849_productive_clicks)


def _empty_aggregate_fix() -> dict[str, object]:
    row = _BASE_EMPTY_AGGREGATE()
    row["productive_click_targets"] = set()
    return row


def _merge_event_fix(target, raw) -> None:
    _BASE_MERGE_EVENT(target, raw)
    game = str(raw.get("game_id", ""))
    if not game:
        return
    row = target.get(game)
    if row is None:
        return
    values = row.setdefault("productive_click_targets", set())
    if isinstance(values, set):
        values.update(int(value) for value in raw.get("productive_click_targets", ()))


def _game_row_fix(
    coordinator,
    game_id: str,
    *,
    refresh_events: bool = True,
) -> dict[str, object]:
    from v8 import action_learning_report_v849 as report

    row = _BASE_GAME_ROW(
        coordinator,
        game_id,
        refresh_events=refresh_events,
    )
    run = report._RUN.get(str(game_id), {})
    exact = run.get("exact_click_targets_tested", set())
    productive = run.get("productive_click_targets", set())
    row["exact_click_targets_tested"] = len(exact) if isinstance(exact, set) else 0
    row["unique_productive_click_targets"] = (
        len(productive) if isinstance(productive, set) else 0
    )
    return row


def _snapshot_fix(coordinator) -> dict[str, object]:
    from v8 import action_learning_report_v849 as report

    payload = _BASE_SNAPSHOT(coordinator)
    rows = payload.get("games", ())
    click_capable = [
        row
        for row in rows
        if isinstance(row, dict) and row.get("action_space_type") in {"click", "mixed"}
    ]
    total_exact = sum(int(row.get("exact_click_targets_tested", 0)) for row in click_capable)
    total_capacity = sum(
        int(report._RUN.get(str(row.get("game_id", "")), {}).get("grid_coordinate_capacity", 0))
        for row in click_capable
    )
    summary = payload.get("summary", {})
    if isinstance(summary, dict):
        summary["click_coverage_pct"] = (
            100.0 * float(total_exact) / float(total_capacity)
            if total_capacity > 0
            else 0.0
        )
    return payload


def _complexity_multiplier_fix(
    coordinator,
    game_id: str,
    *,
    refresh_events: bool = True,
) -> float:
    from v8 import action_learning_report_v849 as report

    if refresh_events:
        report._refresh_events()
    game = str(game_id)
    with coordinator._lock:
        coordinator_spaces = dict(getattr(coordinator, "_v848_action_spaces", {}))
    measured = coordinator_spaces.get(game)
    if measured is None:
        measured = report._probe_game_action_space_v849(
            game,
            refresh_events=False,
        )
    if not measured or not bool(measured[0]):
        return 1.0

    branching = max(1, int(measured[1]))
    references = [
        max(1, int(value[1]))
        for value in coordinator_spaces.values()
        if value is not None and not bool(value[0]) and int(value[1]) > 0
    ]
    if not references:
        for other in report._SPACE.values():
            native = other.get("native_types", set())
            movements = other.get("movement_actions_available", set())
            if (
                isinstance(native, set)
                and 6 not in native
                and isinstance(movements, set)
                and movements
            ):
                references.append(len(movements))
    if not references:
        for other in report._SPACE.values():
            movements = other.get("movement_actions_available", set())
            if isinstance(movements, set) and movements:
                references.append(len(movements))
    reference = max(1.0, float(statistics.median(references))) if references else 1.0
    return math.sqrt(max(1.0, float(branching) / reference))


def install_action_learning_report_v849_fixups() -> None:
    global _INSTALLED
    global _BASE_CLICK_STEP, _BASE_RECORD_EPISODE_KIND
    global _BASE_EVENT_PAYLOAD, _BASE_MARK_REPORTED
    global _BASE_EMPTY_AGGREGATE, _BASE_MERGE_EVENT
    global _BASE_GAME_ROW, _BASE_SNAPSHOT
    if _INSTALLED:
        return

    from v8 import action_learning_report_v849 as report
    from v8 import click_exploration_v848 as click

    _BASE_CLICK_STEP = click._env_step_v848
    _BASE_RECORD_EPISODE_KIND = report._record_episode_kind
    _BASE_EVENT_PAYLOAD = report._event_payload
    _BASE_MARK_REPORTED = report._mark_payload_reported
    _BASE_EMPTY_AGGREGATE = report._empty_aggregate
    _BASE_MERGE_EVENT = report._merge_event
    _BASE_GAME_ROW = report._game_row
    _BASE_SNAPSHOT = report.action_learning_snapshot_v849

    click._env_step_v848 = _click_step_v849_fix
    report._record_episode_kind = _record_episode_kind_fix
    report._event_payload = _event_payload_fix
    report._mark_payload_reported = _mark_reported_fix
    report._empty_aggregate = _empty_aggregate_fix
    report._merge_event = _merge_event_fix
    report._game_row = _game_row_fix
    report.action_learning_snapshot_v849 = _snapshot_fix
    report._click_complexity_multiplier_v849 = _complexity_multiplier_fix
    click._click_complexity_multiplier = _complexity_multiplier_fix
    _INSTALLED = True
