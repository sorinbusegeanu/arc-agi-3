from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class AgentLogContext:
    game_name: str
    agent_name: str
    round_id: str | None = None
    level_index: str | None = None


@dataclass(frozen=True)
class AgentLogEvent:
    severity: str
    message: str
    context: AgentLogContext
    structured_fields: dict[str, Any] = field(default_factory=dict)
