from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ObservedCellFactV4:
    x: int
    y: int
    value: object
    certainty: float = 1.0
    evidence_step_index: int | None = None
    evidence_state_key: str | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.certainty) <= 1.0:
            raise ValueError("certainty must be in [0.0, 1.0]")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class InferredLocalFactV4:
    fact_id: str
    kind: str
    payload: dict[str, object]
    certainty: float
    evidence_refs: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.fact_id:
            raise ValueError("fact_id must be non-empty")
        if not self.kind:
            raise ValueError("kind must be non-empty")
        if not isinstance(self.evidence_refs, tuple):
            raise ValueError("evidence_refs must be a tuple")
        if not 0.0 < float(self.certainty) <= 1.0:
            raise ValueError("certainty must be in (0.0, 1.0]")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
