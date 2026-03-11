from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RankerState:
    ranker_version: str = "ranker:disabled"
    weights: dict[str, float] = field(default_factory=dict)

