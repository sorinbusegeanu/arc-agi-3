from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GameControlProfile:
    control_category: str
    max_action_set: tuple[str, ...]
    enabled_actions: tuple[str, ...]
    movement_actions: tuple[str, ...]
    click_actions: tuple[str, ...]
    supports_wait: bool
