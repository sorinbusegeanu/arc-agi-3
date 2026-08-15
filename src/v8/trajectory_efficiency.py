from __future__ import annotations

from dataclasses import dataclass

from v8.model import MemoryUid


@dataclass(frozen=True, slots=True)
class TrajectoryCost:
    strategy_uid: MemoryUid
    outcome_uid: MemoryUid
    interactions: int
    blocked_actions: int = 0
    repeated_states: int = 0
    loops: int = 0

    @property
    def cost(self) -> float:
        return float(
            max(1, self.interactions)
            + self.blocked_actions
            + self.repeated_states
            + self.loops
        )


class TrajectoryEfficiencyEstimator:
    """Compare cost only inside one learned outcome-equivalence class."""

    def rank(self, rows: tuple[TrajectoryCost, ...]) -> tuple[TrajectoryCost, ...]:
        outcomes = {row.outcome_uid for row in rows}
        if len(outcomes) > 1:
            raise ValueError("efficiency comparison requires one outcome-equivalence class")
        return tuple(sorted(rows, key=lambda row: (row.cost, row.strategy_uid)))
