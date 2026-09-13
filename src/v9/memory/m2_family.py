from __future__ import annotations

from dataclasses import dataclass

from .identity import MemoryUid, stable_u64
from .m1_normalized import M1NormalizedRelation
from .model import MemoryLevel, MemoryType
from .provenance import DerivationProvenance


@dataclass(frozen=True, slots=True)
class M2TransformationFamily:
    uid: MemoryUid
    structural_signature: int
    provenance: DerivationProvenance
    recurrence: int
    compression_benefit: float

    @classmethod
    def form(cls, members: tuple[M1NormalizedRelation, ...], *, representation_cost: float = 1.0) -> "M2TransformationFamily":
        if len(members) < 2:
            raise ValueError("M2 requires recurrent support from at least two M1N structures")
        signatures = {row.structural_signature for row in members}
        if len(signatures) != 1:
            raise ValueError("M2 members must share observable structural identity")
        benefit = float(len(members)) - float(representation_cost)
        if benefit <= 0:
            raise ValueError("M2 formation must provide measurable compression")
        signature = next(iter(signatures))
        uid = MemoryUid.from_key(MemoryLevel.M2, MemoryType.FAMILY, (signature,))
        parents = tuple(sorted({row.uid for row in members}))
        evidence = tuple(sorted({uid for row in members for uid in row.provenance.evidence}))
        if len(evidence) < 2:
            raise ValueError("M2 requires at least two distinct grounded evidence roots")
        return cls(uid, signature, DerivationProvenance(parents, evidence), len(members), benefit)
