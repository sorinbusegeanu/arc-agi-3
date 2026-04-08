from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ResourceValueV4:
    name: str
    current_value: float
    min_safe_value: float = 0.0
    max_value: float | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name must be non-empty")
        if self.max_value is not None and self.current_value > self.max_value:
            raise ValueError("current_value must be <= max_value")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TemporalSnapshotReferenceV4:
    revision: int | None = None
    resource_count: int = 0
    hazard_window_remaining: int | None = None
    safe_horizon_steps: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TemporalResourceStateV4:
    revision: int = 0
    resources: tuple[ResourceValueV4, ...] = ()
    time_cost_per_action: float = 1.0
    hazard_window_remaining: int | None = None
    safe_horizon_steps: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def snapshot_reference(self) -> TemporalSnapshotReferenceV4:
        return TemporalSnapshotReferenceV4(
            revision=self.revision,
            resource_count=len(self.resources),
            hazard_window_remaining=self.hazard_window_remaining,
            safe_horizon_steps=self.safe_horizon_steps,
        )
