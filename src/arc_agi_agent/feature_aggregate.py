from __future__ import annotations

from typing import Any, Dict, List, Optional


FEATURE_KEYS = [
    "global.event_sig.translation.rate",
    "global.event_sig.paint.rate",
    "global.event_sig.toggle.rate",
    "global.event_sig.gravity.rate",
    "global.event_sig.spawn.rate",
    "global.event_sig.despawn.rate",
    "global.event_sig.swap.rate",
    "global.motion.dy.mode",
    "global.motion.dx.mode",
    "global.object_tracking.spawn.rate",
    "global.object_tracking.despawn.rate",
    "global.object_tracking.swap.rate",
    "global.palette.added.rate",
    "global.palette.removed.rate",
    "global.object_count.delta.avg",
    "global.reward.delta.avg",
    "global.terminal.rate",
    "per_action[<action_id>].event_sig.translation.rate",
    "per_action[<action_id>].event_sig.paint.rate",
    "per_action[<action_id>].event_sig.toggle.rate",
    "per_action[<action_id>].event_sig.gravity.rate",
    "per_action[<action_id>].event_sig.spawn.rate",
    "per_action[<action_id>].event_sig.despawn.rate",
    "per_action[<action_id>].event_sig.swap.rate",
    "per_action[<action_id>].noop.rate",
    "per_action[<action_id>].hotspot.non_noop_rate_top1",
    "per_action[<action_id>].negative_zone.noop_rate_top1",
    "per_action[<action_id>].coord.hotspot.count",
    "per_action[<action_id>].coord.negative_zone.count",
]


def aggregate_features(
    initial_fp_reports: List[Dict[str, Any]],
    simple_report: Optional[Dict[str, Any]],
    full_report: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    features: Dict[str, Any] = {}
    _init_global_features(features)

    _aggregate_fp(initial_fp_reports, features)
    if simple_report:
        _aggregate_simple(simple_report, features)
    if full_report:
        _aggregate_full(full_report, features)

    return features


def _init_global_features(features: Dict[str, Any]) -> None:
    for key in FEATURE_KEYS:
        if "<action_id>" in key:
            continue
        features[key] = 0.0


def _aggregate_fp(reports: List[Dict[str, Any]], features: Dict[str, Any]) -> None:
    if not reports:
        return
    event_counts = _event_signature_counts(reports)
    total = sum(event_counts.values()) or 1
    for sig, count in event_counts.items():
        features[f"global.event_sig.{sig}.rate"] = count / total

    motion_dy, motion_dx = _motion_modes(reports)
    features["global.motion.dy.mode"] = motion_dy
    features["global.motion.dx.mode"] = motion_dx

    palette_added, palette_removed = _palette_changes(reports)
    features["global.palette.added.rate"] = palette_added
    features["global.palette.removed.rate"] = palette_removed

    object_delta = _object_count_delta(reports)
    features["global.object_count.delta.avg"] = object_delta

    reward_delta = _reward_delta(reports)
    features["global.reward.delta.avg"] = reward_delta

    terminal_rate = _terminal_rate(reports)
    features["global.terminal.rate"] = terminal_rate

    spawn_rate, despawn_rate, swap_rate = _tracking_rates(reports)
    features["global.object_tracking.spawn.rate"] = spawn_rate
    features["global.object_tracking.despawn.rate"] = despawn_rate
    features["global.object_tracking.swap.rate"] = swap_rate


def _aggregate_simple(report: Dict[str, Any], features: Dict[str, Any]) -> None:
    action_effects = report.get("action_effect_model", {})
    for action_id, stats in action_effects.items():
        prefix = f"per_action[{action_id}]"
        _set_action_feature(features, f"{prefix}.noop.rate", stats.get("no_effect_rate", 0.0))
        dominant = stats.get("dominant_event_signatures", [])
        for sig, rate in dominant:
            _set_action_feature(features, f"{prefix}.event_sig.{sig}.rate", rate)


def _aggregate_full(report: Dict[str, Any], features: Dict[str, Any]) -> None:
    coord_model = report.get("coord_action_effect_model", {})
    for action_id, stats in coord_model.items():
        prefix = f"per_action[{action_id}]"
        _set_action_feature(features, f"{prefix}.noop.rate", stats.get("no_effect_rate", 0.0))
        _set_action_feature(
            features,
            f"{prefix}.coord.hotspot.count",
            float(len(stats.get("hotspots", []))),
        )
        _set_action_feature(
            features,
            f"{prefix}.coord.negative_zone.count",
            float(len(stats.get("negative_zones", []))),
        )
        if stats.get("hotspots"):
            top = stats["hotspots"][0]
            _set_action_feature(features, f"{prefix}.hotspot.non_noop_rate_top1", float(top[2]))
        if stats.get("negative_zones"):
            top = stats["negative_zones"][0]
            _set_action_feature(features, f"{prefix}.negative_zone.noop_rate_top1", float(top[2]))
        dominant = stats.get("dominant_event_signatures", [])
        for sig, rate in dominant:
            _set_action_feature(features, f"{prefix}.event_sig.{sig}.rate", rate)


def _set_action_feature(features: Dict[str, Any], key: str, value: float) -> None:
    features[key] = float(value)


def _event_signature_counts(reports: List[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for rep in reports:
        diff = rep.get("diff_summary")
        if not diff:
            continue
        for sig in diff.get("event_signatures", []):
            kind = sig.get("kind") if isinstance(sig, dict) else None
            if kind:
                counts[kind] = counts.get(kind, 0) + 1
    return counts


def _motion_modes(reports: List[Dict[str, Any]]) -> tuple[float, float]:
    dy = []
    dx = []
    for rep in reports:
        diff = rep.get("diff_summary")
        if not diff:
            continue
        for delta in diff.get("per_object_deltas", []):
            if delta.get("event") == "moved":
                dy.append(delta.get("dy", 0.0))
                dx.append(delta.get("dx", 0.0))
    return _mode(dy), _mode(dx)


def _palette_changes(reports: List[Dict[str, Any]]) -> tuple[float, float]:
    added = 0
    removed = 0
    count = 0
    for rep in reports:
        diff = rep.get("diff_summary")
        if not diff:
            continue
        count += 1
        for k in diff.get("changed_colors", {}).keys():
            prev_color, next_color = k.split("->") if "->" in k else ("", "")
            if prev_color == "0" and next_color:
                added += 1
            if next_color == "0" and prev_color:
                removed += 1
    if count == 0:
        return 0.0, 0.0
    return added / count, removed / count


def _object_count_delta(reports: List[Dict[str, Any]]) -> float:
    delta_total = 0
    count = 0
    for rep in reports:
        diff = rep.get("diff_summary")
        if not diff:
            continue
        count += 1
        appeared = sum(1 for d in diff.get("per_object_deltas", []) if d.get("event") == "appeared")
        disappeared = sum(1 for d in diff.get("per_object_deltas", []) if d.get("event") == "disappeared")
        delta_total += (appeared - disappeared)
    return delta_total / count if count else 0.0


def _reward_delta(reports: List[Dict[str, Any]]) -> float:
    deltas = []
    for rep in reports:
        delta = _extract_reward_delta(rep)
        if delta is not None:
            deltas.append(delta)
    return float(sum(deltas) / len(deltas)) if deltas else 0.0


def _terminal_rate(reports: List[Dict[str, Any]]) -> float:
    if not reports:
        return 0.0
    terminal = 0
    count = 0
    for rep in reports:
        flag = _extract_terminal_flag(rep)
        if flag is None:
            continue
        count += 1
        terminal += int(flag)
    return terminal / count if count else 0.0


def _tracking_rates(reports: List[Dict[str, Any]]) -> tuple[float, float, float]:
    spawn = 0
    despawn = 0
    swap = 0
    count = 0
    for rep in reports:
        diff = rep.get("diff_summary")
        if not diff:
            continue
        count += 1
        for sig in diff.get("event_signatures", []):
            kind = sig.get("kind") if isinstance(sig, dict) else None
            if kind == "spawn":
                spawn += 1
            if kind == "despawn":
                despawn += 1
            if kind == "swap":
                swap += 1
    if count == 0:
        return 0.0, 0.0, 0.0
    return spawn / count, despawn / count, swap / count


def _mode(values: List[float]) -> float:
    if not values:
        return 0.0
    counts: Dict[float, int] = {}
    for val in values:
        counts[val] = counts.get(val, 0) + 1
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def _extract_reward_delta(rep: Dict[str, Any]) -> Optional[float]:
    for path in (
        ("reward_delta",),
        ("reward_change",),
        ("meta", "reward_delta"),
        ("meta", "reward_change"),
        ("debug", "reward_delta"),
        ("debug", "reward_change"),
    ):
        val = _get_nested(rep, path)
        if isinstance(val, (int, float)):
            return float(val)

    reward = _get_nested(rep, ("reward",))
    prev_reward = _get_nested(rep, ("prev_reward",))
    if isinstance(reward, (int, float)) and isinstance(prev_reward, (int, float)):
        return float(reward) - float(prev_reward)
    return None


def _extract_terminal_flag(rep: Dict[str, Any]) -> Optional[bool]:
    for path in (
        ("terminal",),
        ("done",),
        ("meta", "terminal"),
        ("meta", "done"),
        ("debug", "terminal"),
        ("debug", "done"),
    ):
        val = _get_nested(rep, path)
        if isinstance(val, bool):
            return val
        if isinstance(val, (int, float)):
            return bool(val)

    game_state = _get_nested(rep, ("meta", "state"))
    if isinstance(game_state, str):
        return game_state.upper() in {"WIN", "GAME_OVER", "TERMINAL"}
    return None


def _get_nested(rep: Dict[str, Any], path: tuple[str, ...]) -> Any:
    cursor: Any = rep
    for key in path:
        if not isinstance(cursor, dict):
            return None
        cursor = cursor.get(key)
    return cursor
