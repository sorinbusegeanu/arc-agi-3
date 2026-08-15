from __future__ import annotations

from dataclasses import dataclass
from math import exp
from typing import Iterable

from v8.isf import SignificanceVector
from v8.model import MemoryUid


@dataclass(frozen=True, slots=True)
class AttentionPriority:
    uid: MemoryUid
    probability: float
    significance: float


class AttentionAllocator:
    def __init__(self, *, temperature: float = 3.0, max_items: int = 128, exploration_floor: float = 0.01) -> None:
        self.temperature = float(temperature)
        self.max_items = int(max_items)
        self.exploration_floor = max(0.0, float(exploration_floor))

    def allocate(self, rows: Iterable[SignificanceVector]) -> tuple[AttentionPriority, ...]:
        selected = tuple(sorted(rows, key=lambda row: (-row.isf, row.uid))[: max(0, self.max_items)])
        if not selected:
            return ()
        logits = [exp(max(-30.0, min(30.0, self.temperature * row.isf))) for row in selected]
        total = sum(logits)
        base = self.exploration_floor
        normalized = [value / total for value in logits]
        if base > 0:
            uniform = 1.0 / len(selected)
            normalized = [(1.0 - min(1.0, base * len(selected))) * value + min(1.0, base * len(selected)) * uniform for value in normalized]
        return tuple(AttentionPriority(row.uid, prob, row.isf) for row, prob in zip(selected, normalized, strict=True))
