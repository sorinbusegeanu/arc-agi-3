from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping

from v9.memory.identity import MemoryUid


class ReasoningOperator(str, Enum):
    RETRIEVE_SUPPORT = "RETRIEVE_SUPPORT"
    RETRIEVE_COUNTEREVIDENCE = "RETRIEVE_COUNTEREVIDENCE"
    EXPAND_CORRESPONDENCE = "EXPAND_CORRESPONDENCE"
    REFINE_CONTEXT = "REFINE_CONTEXT"
    SIMULATE_CONSEQUENCE = "SIMULATE_CONSEQUENCE"
    COMPARE_OUTCOMES = "COMPARE_OUTCOMES"
    COMPARE_STRATEGIES = "COMPARE_STRATEGIES"
    ESTIMATE_FUTURE_OPTIONS = "ESTIMATE_FUTURE_OPTIONS"
    CHECK_TARGET_TRUST = "CHECK_TARGET_TRUST"
    CHECK_TRAJECTORY_EFFICIENCY = "CHECK_TRAJECTORY_EFFICIENCY"
    RESOLVE_CONTRADICTION = "RESOLVE_CONTRADICTION"
    FALLBACK_LOCAL = "FALLBACK_LOCAL"


@dataclass(frozen=True, slots=True)
class ReasoningWorkspace:
    cycle: int = 0
    ambiguity: float = 1.0
    contradiction_count: int = 0
    consulted_memories: tuple[MemoryUid, ...] = ()
    operator_history: tuple[ReasoningOperator, ...] = ()
    signals: Mapping[str, float] = field(default_factory=dict)

    def advance(
        self,
        *,
        operator: ReasoningOperator,
        ambiguity: float | None = None,
        contradiction_count: int | None = None,
        consulted_memories: tuple[MemoryUid, ...] = (),
        signals: Mapping[str, float] | None = None,
    ) -> "ReasoningWorkspace":
        merged = dict(self.signals)
        if signals:
            merged.update({str(key): float(value) for key, value in signals.items()})
        memory_rows = tuple(dict.fromkeys((*self.consulted_memories, *consulted_memories)))
        return ReasoningWorkspace(
            cycle=self.cycle + 1,
            ambiguity=self.ambiguity if ambiguity is None else max(0.0, float(ambiguity)),
            contradiction_count=(
                self.contradiction_count
                if contradiction_count is None
                else max(0, int(contradiction_count))
            ),
            consulted_memories=memory_rows,
            operator_history=(*self.operator_history, operator),
            signals=merged,
        )
