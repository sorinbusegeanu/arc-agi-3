from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RunContext:
    session_id: str
    run_id: str
    game_id: str

