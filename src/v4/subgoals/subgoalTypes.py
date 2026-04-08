from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class SubgoalProgressV4:
    current_value: float = 0.0
    target_value: float = 1.0
    is_complete: bool = False

    def __post_init__(self) -> None:
        if float(self.target_value) <= 0.0:
            raise ValueError("target_value must be greater than 0.0")
        if float(self.current_value) < 0.0:
            raise ValueError("current_value must be greater than or equal to 0.0")
        if float(self.current_value) >= float(self.target_value):
            object.__setattr__(self, "is_complete", True)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SubgoalV4:
    subgoal_id: str
    family: str
    kind: str
    description: str
    required_facts: tuple[str, ...]
    dependency_ids: tuple[str, ...]
    progress: SubgoalProgressV4

    def __post_init__(self) -> None:
        if not self.subgoal_id:
            raise ValueError("subgoal_id must be non-empty")
        if not self.family:
            raise ValueError("family must be non-empty")
        if not self.kind:
            raise ValueError("kind must be non-empty")
        if not self.description:
            raise ValueError("description must be non-empty")
        if not isinstance(self.required_facts, tuple):
            raise ValueError("required_facts must be a tuple")
        if not isinstance(self.dependency_ids, tuple):
            raise ValueError("dependency_ids must be a tuple")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
