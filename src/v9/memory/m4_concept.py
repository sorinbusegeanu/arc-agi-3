from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .identity import MemoryUid
from .m3_role import M3FunctionalRole
from .model import MemoryLevel, MemoryType
from .provenance import DerivationProvenance


class ConceptState(str, Enum):
    CONCEPT_CANDIDATE = "CONCEPT_CANDIDATE"
    TRANSFER_TEST_ELIGIBLE = "TRANSFER_TEST_ELIGIBLE"
    VALIDATED_CONCEPT = "VALIDATED_CONCEPT"
    FAILED_PROBATIONARY_CONCEPT = "FAILED_PROBATIONARY_CONCEPT"


@dataclass(frozen=True, slots=True)
class M4Concept:
    uid: MemoryUid
    invariant_descriptor: tuple[int, ...]
    provenance: DerivationProvenance
    compression_benefit: float
    explanatory_reach: int
    transfer_prior: float
    held_out_targets: tuple[int, ...] = ()
    validated: bool = False
    state: ConceptState = ConceptState.CONCEPT_CANDIDATE

    @classmethod
    def candidate(cls, roles: tuple[M3FunctionalRole, ...], *, compression_benefit: float, explanatory_reach: int, transfer_prior: float, formation_scope: tuple[int, ...]) -> "M4Concept":
        if not roles or compression_benefit <= 0 or explanatory_reach <= 0:
            raise ValueError("concept candidates require roles, compression and explanatory reach")
        structural = tuple(value for row in sorted(roles, key=lambda item: (item.relational_signature, item.consequence_signature)) for value in (row.relational_signature, row.consequence_signature))
        uid = MemoryUid.from_key(MemoryLevel.M4, MemoryType.CONCEPT, structural)
        parents = tuple(sorted(row.uid for row in roles))
        evidence = tuple(uid for row in roles for uid in row.provenance.evidence)
        return cls(uid, structural, DerivationProvenance(parents, evidence, tuple(sorted(set(formation_scope)))), float(compression_benefit), int(explanatory_reach), float(transfer_prior), state=ConceptState.TRANSFER_TEST_ELIGIBLE)

    def with_validation(self, targets: tuple[int, ...]) -> "M4Concept":
        from dataclasses import replace
        held_out = tuple(sorted(set(int(target) for target in targets) - set(self.provenance.formation_scope)))
        return replace(self, held_out_targets=held_out, validated=bool(held_out), state=ConceptState.VALIDATED_CONCEPT if held_out else self.state)
