from __future__ import annotations

from typing import Any

from v3_1.execution.env_factory import normalize_action_lookup


LETTER_TO_ACTION = {
    "U": "UP",
    "D": "DOWN",
    "L": "LEFT",
    "R": "RIGHT",
}

ACTION_TO_ENV_NAME = {
    "UP": "up",
    "DOWN": "down",
    "LEFT": "left",
    "RIGHT": "right",
}


def canonicalize_action_name(action: Any) -> str:
    if isinstance(action, dict):
        row = normalize_action_lookup(action)
        return str(row.get("action_name") or "").strip().upper()
    token = str(action).strip().upper()
    if token in LETTER_TO_ACTION:
        return LETTER_TO_ACTION[token]
    if token in ACTION_TO_ENV_NAME:
        return token
    row = normalize_action_lookup(token)
    name = str(row.get("action_name") or "").strip().upper()
    return name if name else token


def parse_action_letters(text: str) -> list[str]:
    compact = str(text or "").replace(",", "").replace(" ", "").strip().upper()
    if not compact:
        return []
    actions: list[str] = []
    for ch in compact:
        if ch not in LETTER_TO_ACTION:
            raise ValueError(f"unsupported action letter: {ch}")
        actions.append(LETTER_TO_ACTION[ch])
    return actions


def action_sequence_to_letters(actions: list[str]) -> str:
    reverse = {value: key for key, value in LETTER_TO_ACTION.items()}
    letters: list[str] = []
    for action in actions:
        name = canonicalize_action_name(action)
        if name not in reverse:
            raise ValueError(f"cannot encode action as LRUD letter: {action}")
        letters.append(reverse[name])
    return "".join(letters)


def extract_available_actions(adapter: Any, info: dict[str, Any] | None = None) -> list[str]:
    if hasattr(adapter, "available_actions"):
        rows = adapter.available_actions()
        if isinstance(rows, list) and rows:
            out: list[str] = []
            for row in rows:
                normalized = canonicalize_action_name(row)
                if normalized:
                    out.append(normalized)
            return out
    if isinstance(info, dict):
        raw_actions = info.get("available_actions")
        if isinstance(raw_actions, list):
            out = []
            for item in raw_actions:
                normalized = canonicalize_action_name(item)
                if normalized:
                    out.append(normalized)
            return out
    return ["UP", "DOWN", "LEFT", "RIGHT", "INTERACT", "CLICK_AT", "UNDO"]


def resolve_env_action(action_name: str, action_rows: list[dict[str, Any]]) -> Any:
    target = canonicalize_action_name(action_name)
    for row in action_rows:
        if canonicalize_action_name(row) != target:
            continue
        if row.get("raw") is not None:
            return row["raw"]
        if row.get("id") is not None:
            return row["id"]
        if row.get("name") is not None:
            return row["name"]
    return ACTION_TO_ENV_NAME.get(target, target.lower())
