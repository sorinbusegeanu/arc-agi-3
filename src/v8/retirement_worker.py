from __future__ import annotations

from dataclasses import dataclass

from v8.model import CognitiveState, MemoryUid
from v8.pruning import PruningPlanner


@dataclass(frozen=True, slots=True)
class RetirementDecision:
    uid: MemoryUid
    retire: bool


class RetirementWorker:
    """Convert dependency-safe RETIRE_PENDING nodes into RETIRED state proposals."""

    def __init__(self) -> None:
        self.pruning = PruningPlanner()

    def decide(self, nodes, edges) -> tuple[RetirementDecision, ...]:
        return tuple(
            RetirementDecision(row.uid, not row.protected_by_dependencies)
            for row in self.pruning.candidates(nodes, edges)
        )
