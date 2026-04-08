from __future__ import annotations

import json
from typing import Any


def format_single_game_result(result: dict[str, Any]) -> str:
    return json.dumps(result, indent=2, sort_keys=True)


def format_multi_game_summary(summary: dict[str, Any]) -> str:
    return json.dumps(summary, indent=2, sort_keys=True)


def format_benchmark_summary(summary: dict[str, Any]) -> str:
    return json.dumps(summary, indent=2, sort_keys=True)
