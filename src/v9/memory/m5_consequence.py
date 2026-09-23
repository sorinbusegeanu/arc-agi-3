from __future__ import annotations

from dataclasses import dataclass

from .identity import MemoryUid
from .m4_concept import M4Concept
from .model import MemoryLevel, MemoryType
from .provenance import DerivationProvenance


@dataclass(frozen=True, slots=True)
class M5ConsequenceStructure:
    uid: MemoryUid
    consequence_descriptor: tuple[int, ...]
    provenance: DerivationProvenance
    mature: bool

    @classmethod
    def form(cls, concepts: tuple[M4Concept, ...], descriptor: tuple[int, ...]) -> "M5ConsequenceStructure":
        if not concepts:
            raise ValueError("M5 requires concept evidence")
        normalized_descriptor = tuple(int(v) for v in descriptor)
        parents = tuple(sorted({row.uid for row in concepts}))
        identity = (
            len(normalized_descriptor),
            *normalized_descriptor,
            len(parents),
            *(value for uid in parents for value in (int(uid.hi), int(uid.lo))),
        )
        uid = MemoryUid.from_key(MemoryLevel.M5, MemoryType.CONSEQUENCE, identity)
        evidence = tuple(sorted({uid for row in concepts for uid in row.provenance.evidence}))
        return cls(
            uid,
            normalized_descriptor,
            DerivationProvenance(parents, evidence),
            all(row.validated for row in concepts),
        )

