from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from v8.arena import NodeRecord
from v8.model import MemoryLevel, MemoryType, MemoryUid


@dataclass(frozen=True, slots=True)
class RoleCandidate:
    uid: MemoryUid
    key_parts: tuple[int, ...]
    carriers: tuple[MemoryUid, ...]
    game_evidence_count: int


class FunctionalRoleEstimator:
    """Promote recurring relational positions across distinct carrier hypotheses."""

    def __init__(self, *, min_carriers: int = 2) -> None:
        self.min_carriers = int(min_carriers)

    def propose(self, rows: tuple[NodeRecord, ...]) -> tuple[RoleCandidate, ...]:
        grouped: dict[tuple[int, int], list[NodeRecord]] = defaultdict(list)
        for row in rows:
            if int(row.level) != int(MemoryLevel.M3) or int(row.memory_type) != int(MemoryType.CARRIER):
                continue
            if len(row.key_parts) < 3:
                continue
            family, carrier, future_bucket = map(int, row.key_parts[:3])
            grouped[(family, future_bucket)].append(row)

        result: list[RoleCandidate] = []
        for (family, future_bucket), members in grouped.items():
            distinct_carriers = {int(row.key_parts[1]) for row in members}
            if len(distinct_carriers) < self.min_carriers:
                continue
            key = (family, future_bucket)
            mask = 0
            for row in members:
                mask |= int(row.game_mask)
            result.append(
                RoleCandidate(
                    MemoryUid.from_key(MemoryLevel.M3, MemoryType.ROLE, key),
                    key,
                    tuple(row.uid for row in members),
                    mask.bit_count(),
                )
            )
        return tuple(result)
