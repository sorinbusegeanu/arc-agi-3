from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from v5_0.contracts.avatar_types import ProbePlan

_ACTION_BY_CODE: dict[str, str] = {
    "U": "UP",
    "D": "DOWN",
    "L": "LEFT",
    "R": "RIGHT",
}

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.json"


@lru_cache(maxsize=1)
def _load_bootstrap_sequence() -> tuple[str, ...]:
    payload = json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
    raw_value = str(payload.get("bootstrap_sequence", "")).strip().upper()
    if not raw_value:
        raise ValueError("v5_0 config bootstrap_sequence must be non-empty")
    try:
        return tuple(_ACTION_BY_CODE[code] for code in raw_value)
    except KeyError as exc:
        raise ValueError(f"unsupported bootstrap_sequence code: {exc.args[0]}") from exc


DEFAULT_PROBE_SEQUENCE: tuple[str, ...] = _load_bootstrap_sequence()


def build_probe_plan(*, game_id: str, level_id: str) -> ProbePlan:
    return ProbePlan(game_id=game_id, level_id=level_id, action_sequence=DEFAULT_PROBE_SEQUENCE)
