from __future__ import annotations

from typing import Any, Iterable

from v3_1.execution.env_factory import normalize_action_lookup


def canonicalize_action_name(action: Any) -> str:
    if isinstance(action, dict):
        for key in ("name", "action_name", "action_id", "id"):
            if action.get(key) is not None:
                action = action.get(key)
                break
    row = normalize_action_lookup(action)
    name = str(row.get("action_name") or action or "").strip()
    return name.upper() if name else "UNKNOWN"


def env_action_name(action_name: str) -> str:
    return canonicalize_action_name(action_name).lower()


def normalize_allowed_actions(raw_actions: Iterable[Any]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in raw_actions:
        if isinstance(item, dict):
            if item.get("name") is not None:
                candidate = item.get("name")
            elif item.get("action_name") is not None:
                candidate = item.get("action_name")
            elif item.get("id") is not None:
                candidate = item.get("id")
            else:
                candidate = item.get("action_id")
        else:
            candidate = item
        name = canonicalize_action_name(candidate)
        if name == "UNKNOWN" or name in seen:
            continue
        seen.add(name)
        ordered.append(name)
    return ordered


def extract_available_actions(env_adapter: Any) -> list[str]:
    raw_actions = []
    if hasattr(env_adapter, "env") and hasattr(env_adapter.env, "available_actions"):
        try:
            raw_actions = env_adapter.env.available_actions()
        except Exception:
            raw_actions = []
    if not raw_actions and hasattr(env_adapter, "available_actions"):
        raw_actions = env_adapter.available_actions()
    return normalize_allowed_actions(raw_actions)


def validate_action_sequence(actions: Iterable[Any], *, allowed_actions: Iterable[str], max_length: int) -> list[str]:
    allowed = set(normalize_allowed_actions(allowed_actions))
    normalized: list[str] = []
    for idx, item in enumerate(actions):
        if idx >= max_length:
            break
        name = canonicalize_action_name(item)
        if name not in allowed:
            raise ValueError(f"unknown action: {item}")
        normalized.append(name)
    if not normalized:
        raise ValueError("action sequence is empty")
    return normalized
