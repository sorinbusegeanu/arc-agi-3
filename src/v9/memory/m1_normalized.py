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

    @classmethod
    def build(cls, observable_relation: str, channel: NormalizedChannel, parents: tuple[M1GroundedContingency, ...]) -> "M1NormalizedRelation":
        if not parents:
            raise ValueError("normalized relation requires grounded parents")
        signature = stable_u64(observable_relation, channel.value, person=b"v9-m1-normalized")
        uid = MemoryUid.from_key(MemoryLevel.M1, MemoryType.NORMALIZED_RELATION, (signature,))
        parent_ids = tuple(row.uid for row in parents)
        evidence = tuple(uid for row in parents for uid in row.provenance.evidence)
        watermark = max((int(getattr(row, "created_watermark", 0)) for row in parents), default=0)\n        return cls(uid, observable_relation, channel, signature, DerivationProvenance(parent_ids, evidence), float(len(parents)), 0.0, (), watermark)

