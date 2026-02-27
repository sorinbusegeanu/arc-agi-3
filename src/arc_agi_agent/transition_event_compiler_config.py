from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class TransitionEventCompilerConfig:
    enable_multigrid: bool = False
    hash_meta_whitelist: List[str] = field(default_factory=lambda: ["available_actions", "terminal", "reward"])
    changed_colors_topM: int = 12
