from __future__ import annotations

from typing import Protocol

import numpy as np


class Environment(Protocol):
    """Minimal v6 environment contract over ARC grids and integer actions."""

    def observe(self) -> np.ndarray:
        ...

    def step(self, action: int) -> np.ndarray:
        ...

    def available_actions(self) -> list[int]:
        ...
