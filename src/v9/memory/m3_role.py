from __future__ import annotations

from dataclasses import dataclass

from .identity import MemoryUid, stable_u64
from .m2_family import M2TransformationFamily
from .model import MemoryLevel, MemoryType
from .provenance import DerivationProvenance


@dataclass(frozen=True, slots=True)
class M3FunctionalRole:
    uid: MemoryUid
    relational_signature: int
    consequence_signature: int
    provenance: DerivationProvenance
    support_decomposition: tuple[tuple[str, int], ...] = ()

    @classmethod
    def form(cls, families: tuple[M2TransformationFamily, ...], *, consequence_signature: int) -> "M3FunctionalRole":
        if not families:
            raise ValueError("M3 requires M2 family evidence")
        if len({uid for row in families for uid in row.provenance.evidence}) < 2:
            raise ValueError("M3 requires diverse lower-level support")
        relational = stable_u64(*(sorted(row.structural_signature for row in families)), person=b"v9-m3-relational")
        uid = MemoryUid.from_key(MemoryLevel.M3, MemoryType.ROLE, (relational, int(consequence_signature)))
        parents = tuple(sorted(row.uid for row in families))
        evidence = tuple(sorted({uid for row in families for uid in row.provenance.evidence}))
        support: dict[str, int] = {}
        for family in families:
            for key, value in family.support_decomposition:
                support[key] = support.get(key, 0) + int(value)
        return cls(
            uid,
            relational,
            int(consequence_signature),
            DerivationProvenance(parents, evidence),
            tuple(sorted(support.items())),
        )
