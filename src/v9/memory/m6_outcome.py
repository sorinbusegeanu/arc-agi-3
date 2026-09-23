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
    equivalence_trials: int = 0
    equivalence_successes: int = 0
    contexts_observed: tuple[int, ...] = ()
    environments_observed: tuple[int, ...] = ()
    primary_valence_sum: int = 0
    preference_trials: int = 0

    @property
    def equivalence_confidence(self) -> float:
        return self.equivalence_successes / self.equivalence_trials if self.equivalence_trials else 0.0

    @property
    def mean_primary_valence(self) -> float:
        return self.primary_valence_sum / self.preference_trials if self.preference_trials else 0.0

    @classmethod
    def form(cls, consequences: tuple[M5ConsequenceStructure, ...], *, diameter_bound: int) -> "M6Outcome":
        if not consequences:
            raise ValueError("M6 requires consequence evidence")
        descriptors = tuple(sorted({value for row in consequences for value in row.consequence_descriptor}))
        if descriptors and max(descriptors) - min(descriptors) > int(diameter_bound):
            raise ValueError("outcome candidates exceed declared within-class diameter")
        uid = MemoryUid.from_key(MemoryLevel.M6, MemoryType.OUTCOME, descriptors)
        members = tuple(sorted({row.uid for row in consequences}))
        evidence = tuple(sorted({uid for row in consequences for uid in row.provenance.evidence}))
        return cls(uid, descriptors, members, DerivationProvenance(members, evidence))

