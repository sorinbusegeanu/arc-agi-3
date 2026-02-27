from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


REWARD_PRIORITY = [
    "reward_delta",
    "reward",
    "score_delta",
    "score",
    "points_delta",
    "points",
]

TERMINAL_PRIORITY = [
    "terminal",
    "done",
    "is_terminal",
]

WIN_STATUS = {"WIN", "WON", "SUCCESS"}
LOSE_STATUS = {"LOSE", "LOST", "GAME_OVER", "FAIL"}


def extract_meta(fp_report: Dict[str, Any], trace_entry: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    meta: Dict[str, Any] = {
        "reward_value": None,
        "reward_key": None,
        "terminal": None,
        "terminal_key": None,
        "status": None,
        "counters": {},
    }

    trace_entry = trace_entry or {}
    info = trace_entry.get("info") if isinstance(trace_entry.get("info"), dict) else {}
    counters = trace_entry.get("counters") if isinstance(trace_entry.get("counters"), dict) else {}

    reward_key, reward_value = _extract_reward(trace_entry, info, counters, fp_report)
    if reward_key is not None:
        meta["reward_key"] = reward_key
        meta["reward_value"] = reward_value

    terminal_key, terminal_value, status = _extract_terminal(trace_entry, info, counters, fp_report)
    if terminal_key is not None:
        meta["terminal_key"] = terminal_key
        meta["terminal"] = terminal_value
    if status is not None:
        meta["status"] = status

    meta["counters"] = _extract_counters(info, counters, fp_report)
    return meta


def _extract_reward(
    trace_entry: Dict[str, Any],
    info: Dict[str, Any],
    counters: Dict[str, Any],
    fp_report: Dict[str, Any],
) -> Tuple[Optional[str], Optional[float]]:
    for key in REWARD_PRIORITY:
        value = _lookup_key(trace_entry, info, counters, fp_report, key)
        if isinstance(value, (int, float)):
            return key, float(value)
    return None, None


def _extract_terminal(
    trace_entry: Dict[str, Any],
    info: Dict[str, Any],
    counters: Dict[str, Any],
    fp_report: Dict[str, Any],
) -> Tuple[Optional[str], Optional[bool], Optional[str]]:
    for key in TERMINAL_PRIORITY:
        value = _lookup_key(trace_entry, info, counters, fp_report, key)
        if isinstance(value, bool):
            return key, value, None
        if isinstance(value, (int, float)):
            return key, bool(value), None

    status = _lookup_key(trace_entry, info, counters, fp_report, "status")
    if isinstance(status, str):
        upper = status.upper()
        if upper in WIN_STATUS:
            return "status", True, upper

    state = _lookup_key(trace_entry, info, counters, fp_report, "state")
    if isinstance(state, str):
        upper = state.upper()
        if upper in WIN_STATUS:
            return "state", True, upper

    if isinstance(status, str):
        upper = status.upper()
        if upper in LOSE_STATUS:
            return "status", True, upper
    return None, None, None


def _extract_counters(info: Dict[str, Any], counters: Dict[str, Any], fp_report: Dict[str, Any]) -> Dict[str, float]:
    extracted: Dict[str, float] = {}
    for source in (info, counters, fp_report):
        if not isinstance(source, dict):
            continue
        for key, value in source.items():
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                extracted[key] = float(value)
    return extracted


def _lookup_key(
    trace_entry: Dict[str, Any],
    info: Dict[str, Any],
    counters: Dict[str, Any],
    fp_report: Dict[str, Any],
    key: str,
) -> Any:
    for source in (trace_entry, info, counters, fp_report):
        if isinstance(source, dict) and key in source:
            return source[key]
    return None
