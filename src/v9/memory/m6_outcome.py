from __future__ import annotations

from dataclasses import dataclass

from .identity import MemoryUid
from .m5_consequence import M5ConsequenceStructure
from .model import MemoryLevel, MemoryType
from .provenance import DerivationProvenance


@dataclass(frozen=True, slots=True)
class M6Outcome:
    uid: MemoryUid
    class_signature: tuple[int, ...]
    members: tuple[MemoryUid, ...]
    provenance: DerivationProvenance
    class_version: int = 1

    @classmethod
    def form(cls, consequences: tuple[M5ConsequenceStructure, ...], *, diameter_bound: int) -> "M6Outcome":
        if not consequences:
            raise ValueError("M6 requires consequence evidence")
        descriptors = tuple(sorted({value for row in consequences for value in row.consequence_descriptor}))
        if descriptors and max(descriptors) - min(descriptors) > int(diameter_bound):
            raise ValueError("outcome candidates exceed declared within-class diameter")
        uid = MemoryUid.from_key(MemoryLevel.M6, MemoryType.OUTCOME, descriptors)
        members = tuple(sorted(row.uid for row in consequences))
        evidence = tuple(uid for row in consequences for uid in row.provenance.evidence)
        return cls(uid, descriptors, members, DerivationProvenance(members, evidence))

