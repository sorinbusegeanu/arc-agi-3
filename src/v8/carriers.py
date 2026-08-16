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
    carrier_utility: float = 0.0
    compression_gain: float = 0.0
    activatable: bool = False


class CarrierEstimator:
    """Evaluate carrier persistence separately from raw recurrence.

    Promotion creates a structural hypothesis.  Stable activation additionally
    requires positive explanatory/compression evidence stored on that hypothesis.
    """

    def __init__(self, *, min_support: int = 2, utility_threshold: float = 0.05) -> None:
        self.min_support = int(min_support)
        self.utility_threshold = float(utility_threshold)

    def evaluate(self, rows: tuple[NodeRecord, ...]) -> tuple[CarrierHypothesis, ...]:
        result = []
        for row in rows:
            if int(row.level) != int(MemoryLevel.M3) or int(row.memory_type) != int(MemoryType.CARRIER):
                continue
            if len(row.key_parts) < 3 or int(row.key_parts[1]) == 0:
                continue
            support = max(0, int(row.support_count))
            compression_gain = max(0.0, support - 1.0) / max(1.0, float(support))
            explanatory = max(0.0, min(1.0, float(row.explanatory_reach)))
            utility = max(explanatory, compression_gain if explanatory > 0.0 else 0.0)
            result.append(
                CarrierHypothesis(
                    row.uid,
                    int(row.key_parts[0]),
                    int(row.key_parts[1]),
                    int(row.key_parts[2]),
                    support,
                    utility,
                    compression_gain,
                    bool(support >= self.min_support and utility > self.utility_threshold),
                )
            )
        return tuple(result)
