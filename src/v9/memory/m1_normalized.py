from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .identity import MemoryUid, stable_u64
from .m1_grounded import M1GroundedContingency
from .model import MemoryLevel, MemoryType
from .provenance import DerivationProvenance


class NormalizedChannel(str, Enum):
    WORLD = "WORLD"
    SYMBOL = "SYMBOL"
    CROSS_MODAL = "CROSS_MODAL"


@dataclass(frozen=True, slots=True)
class M1NormalizedRelation:
    uid: MemoryUid
    observable_relation: str
    channel: NormalizedChannel
    structural_signature: int
    provenance: DerivationProvenance
    support: float = 1.0
    contradiction: float = 0.0
    temporal_offsets: tuple[int, ...] = ()
    causal_watermark: int = 0

    @classmethod
    def build(
        cls,
        observable_relation: str,
        channel: NormalizedChannel,
        parents: tuple[M1GroundedContingency, ...],
        *,
        support: float = 1.0,
        contradiction: float = 0.0,
        temporal_offsets: tuple[int, ...] = (),
        causal_watermark: int = 0,
    ) -> "M1NormalizedRelation":
        if not parents:
            raise ValueError("normalized relation requires grounded parents")
        if support < 0.0 or contradiction < 0.0:
            raise ValueError("normalized relation evidence must be non-negative")
        signature = stable_u64(observable_relation, channel.value, person=b"v9-m1-normalized")
        uid = MemoryUid.from_key(MemoryLevel.M1, MemoryType.NORMALIZED_RELATION, (signature,))
        parent_ids = tuple(row.uid for row in parents)
        evidence = tuple(uid for row in parents for uid in row.provenance.evidence)
        return cls(
            uid,
            observable_relation,
            channel,
            signature,
            DerivationProvenance(parent_ids, evidence),
            float(support),
            float(contradiction),
            tuple(int(value) for value in temporal_offsets),
            int(causal_watermark),
        )
