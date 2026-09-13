from __future__ import annotations

from dataclasses import dataclass

from .codec import SymbolObservation


@dataclass(frozen=True, slots=True)
class OrderedSymbolStream:
    observations: tuple[SymbolObservation, ...]

    def __post_init__(self) -> None:
        positions = tuple(row.position.value for row in self.observations)
        if positions != tuple(sorted(positions)) or len(positions) != len(set(positions)):
            raise ValueError("symbol positions must be unique and ordered")

