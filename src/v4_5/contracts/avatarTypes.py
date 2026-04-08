from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class AvatarProbeFrame:
    action: str
    pre_frame: tuple[tuple[int, ...], ...]
    post_frame: tuple[tuple[int, ...], ...]
    pre_levels_completed: int = 0
    post_levels_completed: int = 0


@dataclass(frozen=True)
class AvatarProbeSequenceResult:
    sequence_name: str
    frames: tuple[AvatarProbeFrame, ...]
    actions: tuple[str, ...]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AvatarDetectionResult:
    avatar_bbox: tuple[int, int, int, int] | None = None
    avatar_center: tuple[float, float] | None = None
    support_actions: tuple[str, ...] = ()
    support_step_indices: tuple[int, ...] = ()
    confidence: float = 0.0
    avatar_value_candidates: tuple[int, ...] = ()
    failure_reason: str | None = None
    used_fallback: bool = False
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def avatar_position(self) -> tuple[float, float] | None:
        return self.avatar_center
