from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from v8.arena import NodeRecord
from v8.model import MemoryLevel, MemoryType, MemoryUid


@dataclass(frozen=True, slots=True)
class WorldModelComponent:
    uid: MemoryUid
    key_parts: tuple[int, ...]
    consequences: tuple[MemoryUid, ...]
    support: int
    game_evidence_count: int


class WorldModelEstimator:
    """Integrate recurring M5 consequence structures into explanatory components."""

    def __init__(self, *, min_consequences: int = 2) -> None:
        self.min_consequences = int(min_consequences)

    def propose(self, rows: tuple[NodeRecord, ...]) -> tuple[WorldModelComponent, ...]:
        grouped: dict[tuple[int, int], list[NodeRecord]] = defaultdict(list)
        for row in rows:
            if int(row.level) != int(MemoryLevel.M5):
                continue
            if int(row.memory_type) != int(MemoryType.CONSEQUENCE) or len(row.key_parts) < 4:
                continue
            outcome_signature = int(row.key_parts[2])
            future_bucket = int(row.key_parts[3])
            grouped[(outcome_signature, future_bucket)].append(row)

        result = []
        for key, members in grouped.items():
            distinct_concepts = {
                (int(row.key_parts[0]), int(row.key_parts[1])) for row in members
            }
            if len(distinct_concepts) < self.min_consequences:
                continue
            mask = 0
            for row in members:
                mask |= int(row.game_mask)
            result.append(
                WorldModelComponent(
                    MemoryUid.from_key(MemoryLevel.M5, MemoryType.WORLD_MODEL, key),
                    key,
                    tuple(sorted(row.uid for row in members)),
                    sum(max(0, int(row.support_count)) for row in members),
                    mask.bit_count(),
                )
            )
        return tuple(result)
