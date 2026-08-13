from __future__ import annotations

from dataclasses import dataclass
from typing import NewType

GenerationId = NewType("GenerationId", int)


@dataclass(frozen=True, slots=True)
class GenerationState:
    generation_id: GenerationId
    parent_generation_id: GenerationId | None
    first_global_step: int | None = None
    last_global_step: int | None = None

    def __post_init__(self) -> None:
        if int(self.generation_id) < 0:
            raise ValueError("generation_id must be non-negative")
        if self.parent_generation_id is not None and int(self.parent_generation_id) >= int(self.generation_id):
            raise ValueError("parent generation must precede generation")
        if (
            self.first_global_step is not None
            and self.last_global_step is not None
            and self.last_global_step < self.first_global_step
        ):
            raise ValueError("last_global_step must not precede first_global_step")
