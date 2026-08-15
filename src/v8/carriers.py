from __future__ import annotations

from dataclasses import dataclass

from v8.arena import NodeRecord
from v8.model import MemoryLevel, MemoryType, MemoryUid


@dataclass(frozen=True, slots=True)
class CarrierHypothesis:
    uid: MemoryUid
    family_signature: int
    carrier_signature: int
    future_bucket: int
    support: int


class CarrierEstimator:
    """Expose persistent M3 carrier hypotheses separately from functional roles."""

    def evaluate(self, rows: tuple[NodeRecord, ...]) -> tuple[CarrierHypothesis, ...]:
        result = []
        for row in rows:
            if int(row.level) != int(MemoryLevel.M3) or int(row.memory_type) != int(MemoryType.CARRIER):
                continue
            if len(row.key_parts) < 3 or int(row.key_parts[1]) == 0:
                continue
            result.append(
                CarrierHypothesis(
                    row.uid,
                    int(row.key_parts[0]),
                    int(row.key_parts[1]),
                    int(row.key_parts[2]),
                    int(row.support_count),
                )
            )
        return tuple(result)
