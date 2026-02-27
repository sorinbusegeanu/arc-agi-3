from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from .goal_detector_config import GoalDetectorConfig
from .goal_detector_types import GoalDetectorReport, GoalHints, ProgressEstimate, SignalEntry
from .goal_proxies import compute_proxies
from .goal_signal_extract import extract_meta


def estimate(
    fp_reports: List[Dict[str, Any]],
    trace_path: Optional[str] = None,
    cfg: Optional[GoalDetectorConfig] = None,
    ctx: Optional[Dict[str, Any]] = None,
) -> GoalDetectorReport:
    if not fp_reports:
        raise ValueError("fp_reports is required and must be non-empty")

    cfg = cfg or GoalDetectorConfig()
    reports_window = _select_window(fp_reports, cfg)
    trace_entries = _load_trace(trace_path)
    trace_by_step = {entry.get("step_idx"): entry for entry in trace_entries if isinstance(entry, dict)}

    meta_series = []
    proxy_series = []
    warnings: List[str] = []

    initial_targets = None
    initial_components = None
    for rep in reports_window:
        step_idx = rep.get("state_summary", {}).get("step_idx")
        trace_entry = trace_by_step.get(step_idx)
        meta = extract_meta(rep, trace_entry)
        meta_series.append(meta)

        proxies = compute_proxies(
            rep,
            min_target_color_rarity=cfg.min_target_color_rarity,
            initial_target_counts=initial_targets,
            initial_component_count=initial_components,
        )
        if initial_targets is None:
            initial_targets = proxies.get("target_counts", {})
        if initial_components is None:
            initial_components = proxies.get("component_count")
        proxy_series.append(proxies)

    progress_estimate, signals, goal_hints, warnings = _compute_progress(
        meta_series, proxy_series, cfg, warnings
    )

    run_summary = {
        "window_length": len(reports_window),
        "warnings": warnings,
        "signal_families": list(signals.keys()),
    }
    if ctx:
        run_summary["ctx"] = ctx

    return GoalDetectorReport(
        progress_estimate=progress_estimate,
        signals=signals,
        goal_hints=goal_hints,
        run_summary=run_summary,
    )


def _select_window(fp_reports: List[Dict[str, Any]], cfg: GoalDetectorConfig) -> List[Dict[str, Any]]:
    if len(fp_reports) <= cfg.max_window_steps:
        return fp_reports
    return fp_reports[-cfg.max_window_steps :]


def _load_trace(path: Optional[str]) -> List[Dict[str, Any]]:
    if not path:
        return []
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries


def _compute_progress(
    meta_series: List[Dict[str, Any]],
    proxy_series: List[Dict[str, Any]],
    cfg: GoalDetectorConfig,
    warnings: List[str],
) -> Tuple[ProgressEstimate, Dict[str, Any], GoalHints, List[str]]:
    if not meta_series:
        warnings.append("no_meta")
        return _fallback_unknown(cfg), {}, GoalHints(likely_goal_type="unknown", stop_condition_predicates=[]), warnings

    reward_values, reward_key = _collect_reward(meta_series)
    terminal_values, terminal_key = _collect_terminal(meta_series)
    counters = _collect_counters(meta_series)

    signals: Dict[str, Any] = {
        "reward_signal": None,
        "terminal_signal": None,
        "counter_signals": [],
        "board_signals": [],
        "object_signals": [],
    }

    progress_scalar = 0.5
    progress_delta = 0.0

    if reward_values:
        progress_scalar, progress_delta = _progress_from_reward(reward_values)
        signals["reward_signal"] = SignalEntry(
            signal_id=reward_key or "reward",
            value_start=reward_values[0],
            value_end=reward_values[-1],
            delta=reward_values[-1] - reward_values[0],
            weight=1.0,
            evidence=[{"key": reward_key}],
        )
    elif terminal_values:
        if any(terminal_values):
            progress_scalar = 1.0
        else:
            progress_scalar, progress_delta = _progress_from_proxies(proxy_series, cfg)
        signals["terminal_signal"] = SignalEntry(
            signal_id=terminal_key or "terminal",
            value_start=float(terminal_values[0]),
            value_end=float(terminal_values[-1]),
            delta=float(terminal_values[-1]) - float(terminal_values[0]),
            weight=1.0,
            evidence=[{"key": terminal_key}],
        )
    else:
        progress_scalar, progress_delta = _progress_from_proxies(proxy_series, cfg)

    board_signals, object_signals = _signals_from_proxies(proxy_series, cfg)
    signals["board_signals"] = board_signals
    signals["object_signals"] = object_signals
    signals["counter_signals"] = _signals_from_counters(counters)

    direction = _direction(progress_delta)
    confidence = _confidence(meta_series, reward_values, terminal_values, board_signals, object_signals, cfg)

    if len(meta_series) < cfg.min_window_steps:
        warnings.append("short_window")
        if confidence > cfg.confidence_low:
            confidence = cfg.confidence_low

    if progress_scalar == 0.5 and progress_delta == 0.0 and not reward_values and not terminal_values:
        warnings.append("uninformative_signals")
        confidence = 0.0
        direction = "unknown"

    goal_hints = _goal_hints(proxy_series, terminal_values, cfg)

    return (
        ProgressEstimate(
            progress_scalar=_clamp01(progress_scalar),
            progress_delta=_clamp(progress_delta, -1.0, 1.0),
            confidence=_clamp01(confidence),
            direction=direction,
        ),
        signals,
        goal_hints,
        warnings,
    )


def _collect_reward(meta_series: List[Dict[str, Any]]) -> Tuple[List[float], Optional[str]]:
    values: List[float] = []
    key = None
    for meta in meta_series:
        val = meta.get("reward_value")
        if isinstance(val, (int, float)):
            values.append(float(val))
            if key is None:
                key = meta.get("reward_key")
        else:
            values.append(values[-1] if values else 0.0)
    if key and key.endswith("_delta"):
        cumulative = []
        total = 0.0
        for v in values:
            total += v
            cumulative.append(total)
        values = cumulative
    if all(v == 0.0 for v in values):
        return [], None
    return values, key


def _collect_terminal(meta_series: List[Dict[str, Any]]) -> Tuple[List[bool], Optional[str]]:
    values: List[bool] = []
    key = None
    for meta in meta_series:
        val = meta.get("terminal")
        values.append(bool(val) if val is not None else False)
        if key is None and meta.get("terminal_key"):
            key = meta.get("terminal_key")
    if any(values):
        return values, key
    return [], None


def _collect_counters(meta_series: List[Dict[str, Any]]) -> Dict[str, List[float]]:
    counters: Dict[str, List[float]] = {}
    for meta in meta_series:
        for key, value in (meta.get("counters") or {}).items():
            counters.setdefault(key, []).append(float(value))
    return counters


def _progress_from_reward(values: List[float]) -> Tuple[float, float]:
    if not values:
        return 0.5, 0.0
    min_v = min(values)
    max_v = max(values)
    if max_v == min_v:
        return (1.0 if values[-1] > values[0] else 0.5), 0.0
    norm = [(v - min_v) / (max_v - min_v) for v in values]
    return norm[-1], norm[-1] - norm[0]


def _progress_from_proxies(proxy_series: List[Dict[str, Any]], cfg: GoalDetectorConfig) -> Tuple[float, float]:
    if not proxy_series:
        return 0.5, 0.0
    start = _proxy_score(proxy_series[0], cfg)
    end = _proxy_score(proxy_series[-1], cfg)
    return end, end - start


def _proxy_score(proxies: Dict[str, Any], cfg: GoalDetectorConfig) -> float:
    total = 0.0
    total += cfg.w_target_depletion * float(proxies.get("target_depletion_ratio", 0.0))
    total += cfg.w_filled_area * float(proxies.get("filled_area_ratio", 0.0))
    total += cfg.w_stability * float(proxies.get("stability_ratio", 0.0))
    total += cfg.w_uniformity * float(proxies.get("uniformity_score", 0.0))
    total += cfg.w_symmetry * float(proxies.get("symmetry_score", 0.0))
    total += cfg.w_component_consolidation * float(proxies.get("component_consolidation", 0.0))
    return _clamp01(total)


def _signals_from_proxies(
    proxy_series: List[Dict[str, Any]],
    cfg: GoalDetectorConfig,
) -> Tuple[List[SignalEntry], List[SignalEntry]]:
    if not proxy_series:
        return [], []
    start = proxy_series[0]
    end = proxy_series[-1]
    board = [
        SignalEntry(
            signal_id="filled_area_ratio",
            value_start=float(start.get("filled_area_ratio", 0.0)),
            value_end=float(end.get("filled_area_ratio", 0.0)),
            delta=float(end.get("filled_area_ratio", 0.0)) - float(start.get("filled_area_ratio", 0.0)),
            weight=cfg.w_filled_area,
        ),
        SignalEntry(
            signal_id="stability_ratio",
            value_start=float(start.get("stability_ratio", 0.0)),
            value_end=float(end.get("stability_ratio", 0.0)),
            delta=float(end.get("stability_ratio", 0.0)) - float(start.get("stability_ratio", 0.0)),
            weight=cfg.w_stability,
        ),
        SignalEntry(
            signal_id="uniformity_score",
            value_start=float(start.get("uniformity_score", 0.0)),
            value_end=float(end.get("uniformity_score", 0.0)),
            delta=float(end.get("uniformity_score", 0.0)) - float(start.get("uniformity_score", 0.0)),
            weight=cfg.w_uniformity,
        ),
        SignalEntry(
            signal_id="symmetry_score",
            value_start=float(start.get("symmetry_score", 0.0)),
            value_end=float(end.get("symmetry_score", 0.0)),
            delta=float(end.get("symmetry_score", 0.0)) - float(start.get("symmetry_score", 0.0)),
            weight=cfg.w_symmetry,
        ),
    ]
    objects = [
        SignalEntry(
            signal_id="target_depletion_ratio",
            value_start=float(start.get("target_depletion_ratio", 0.0)),
            value_end=float(end.get("target_depletion_ratio", 0.0)),
            delta=float(end.get("target_depletion_ratio", 0.0)) - float(start.get("target_depletion_ratio", 0.0)),
            weight=cfg.w_target_depletion,
        ),
        SignalEntry(
            signal_id="component_consolidation",
            value_start=float(start.get("component_consolidation", 0.0)),
            value_end=float(end.get("component_consolidation", 0.0)),
            delta=float(end.get("component_consolidation", 0.0)) - float(start.get("component_consolidation", 0.0)),
            weight=cfg.w_component_consolidation,
        ),
    ]
    return board, objects


def _signals_from_counters(counters: Dict[str, List[float]]) -> List[SignalEntry]:
    signals = []
    for key, values in counters.items():
        if not values:
            continue
        signals.append(
            SignalEntry(
                signal_id=key,
                value_start=values[0],
                value_end=values[-1],
                delta=values[-1] - values[0],
                weight=0.5,
            )
        )
    return signals


def _direction(progress_delta: float) -> str:
    if progress_delta > 0.01:
        return "increasing"
    if progress_delta < -0.01:
        return "decreasing"
    if progress_delta == 0:
        return "flat"
    return "unknown"


def _confidence(
    meta_series: List[Dict[str, Any]],
    reward_values: List[float],
    terminal_values: List[bool],
    board_signals: List[SignalEntry],
    object_signals: List[SignalEntry],
    cfg: GoalDetectorConfig,
) -> float:
    confidence = 0.1
    if reward_values:
        confidence += 0.5
    if terminal_values:
        confidence += 0.3
    agreeing = sum(1 for sig in board_signals + object_signals if sig.delta > 0)
    conflicting = sum(1 for sig in board_signals + object_signals if sig.delta < 0)
    if agreeing > conflicting and agreeing > 0:
        confidence += 0.1
    if conflicting > agreeing and conflicting > 0:
        confidence -= 0.1
    if len(meta_series) < cfg.min_window_steps:
        confidence -= 0.2
    return _clamp01(confidence)


def _goal_hints(
    proxy_series: List[Dict[str, Any]],
    terminal_values: List[bool],
    cfg: GoalDetectorConfig,
) -> GoalHints:
    if not proxy_series:
        return GoalHints(likely_goal_type="unknown", stop_condition_predicates=[])
    end = proxy_series[-1]
    goal_type = "unknown"
    predicates: List[str] = []

    if end.get("target_depletion_ratio", 0.0) > 0.6:
        goal_type = "collect_all"
        predicates.append("targets_remaining==0")
    elif end.get("uniformity_score", 0.0) >= cfg.uniformity_goal_threshold:
        goal_type = "paint_to_match"
        predicates.append(f"grid_uniformity>={cfg.uniformity_goal_threshold}")
    elif end.get("stability_ratio", 0.0) >= cfg.stability_goal_threshold:
        goal_type = "stabilize_state"
        predicates.append(f"stability_ratio>={cfg.stability_goal_threshold}")
    if terminal_values:
        predicates.insert(0, "terminal_flag==true")
        if goal_type == "unknown":
            goal_type = "reach_terminal"

    return GoalHints(likely_goal_type=goal_type, stop_condition_predicates=predicates)


def _fallback_unknown(cfg: GoalDetectorConfig) -> ProgressEstimate:
    return ProgressEstimate(progress_scalar=0.5, progress_delta=0.0, confidence=0.0, direction="unknown")


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
