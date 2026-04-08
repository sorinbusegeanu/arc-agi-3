from __future__ import annotations

import json
import re
from typing import Any


_VAR_PATTERN = re.compile(r"\[([A-Za-z0-9_-]+)\]")


def build_prompt(template: str, *, memory: dict[str, Any] | None = None) -> str:
    rendered = str(template)
    available = normalize_prompt_memory(memory or {})
    required = extract_prompt_variables(rendered)
    missing = sorted(name for name in required if name not in available)
    if missing:
        raise ValueError(f"unsupported prompt template variable(s): {', '.join(missing)}")
    for key in sorted(required, key=len, reverse=True):
        rendered = rendered.replace(f"[{key}]", str(available[key]))
    if not rendered:
        raise ValueError("prompt template rendered to empty text")
    return rendered


def extract_prompt_variables(template: str) -> set[str]:
    return set(_VAR_PATTERN.findall(str(template)))


def normalize_prompt_memory(memory: dict[str, Any]) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for key, value in dict(memory).items():
        if value is None:
            continue
        key_str = str(key)
        value_str = _stringify_prompt_value(value)
        normalized[key_str] = value_str
        normalized.setdefault(key_str.replace("-", "_"), value_str)
        normalized.setdefault(key_str.replace("_", "-"), value_str)
    return normalized


def build_prompt_memory(**kwargs: Any) -> dict[str, Any]:
    memory: dict[str, Any] = {}
    for key, value in kwargs.items():
        if value is None:
            continue
        memory[key] = value
    return memory


def _stringify_prompt_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    return json.dumps(value, indent=2, ensure_ascii=False)
