from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class UnknownCellFactV4:
    x: int
    y: int
    reason: str
    frontier: bool = False
    certainty: float = 1.0
    evidence_step_index: int | None = None
    evidence_state_key: str | None = None

    def __post_init__(self) -> None:
        if not self.reason:
            raise ValueError("reason must be non-empty")
        if not 0.0 <= float(self.certainty) <= 1.0:
            raise ValueError("certainty must be in [0.0, 1.0]")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
