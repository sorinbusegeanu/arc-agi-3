from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class HypothesisSnapshotReferenceV4:
    revision: int | None = None
    hypothesis_count: int = 0
    low_count: int = 0
    medium_count: int = 0
    high_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
