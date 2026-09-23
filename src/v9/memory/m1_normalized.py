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
    family_signature: int = 0
    heldout_transfer: bool = False
    context_signature: int = 0

    @property
    def temporal_offset_range(self) -> tuple[int, int] | None:
        if not self.temporal_offsets:
            return None
        return min(self.temporal_offsets), max(self.temporal_offsets)

    @staticmethod
    def _context_signature(parents: tuple[M1GroundedContingency, ...]) -> int:
        contexts = tuple(sorted({int(row.grounded_context_signature) for row in parents}))
        if not contexts:
            return 0
        if len(contexts) == 1:
            return contexts[0]
        return stable_u64(*contexts, person=b"v9-m1-context")

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
        structural_key: tuple[object, ...] | None = None,
        family_key: tuple[object, ...] | None = None,
        heldout_transfer: bool = False,
        context_signature: int | None = None,
    ) -> "M1NormalizedRelation":
        if not parents:
            raise ValueError("normalized relation requires grounded parents")
        if support < 0.0 or contradiction < 0.0:
            raise ValueError("normalized relation evidence must be non-negative")
        if structural_key is None:
            signature = stable_u64(observable_relation, channel.value, person=b"v9-m1-normalized")
        else:
            signature = stable_u64(*structural_key, person=b"v9-m1-structure")
        family_signature = int(signature) if family_key is None else stable_u64(*family_key, person=b"v9-m1-family")
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
            int(family_signature),
            bool(heldout_transfer),
            cls._context_signature(parents) if context_signature is None else int(context_signature),
        )

    @classmethod
    def from_provenance(
        cls,
        observable_relation: str,
        channel: NormalizedChannel,
        *,
        parents: tuple[MemoryUid, ...],
        evidence: tuple[MemoryUid, ...],
        structural_key: tuple[object, ...],
        family_signature: int,
        support: float = 1.0,
        contradiction: float = 0.0,
        temporal_offsets: tuple[int, ...] = (),
        causal_watermark: int = 0,
        heldout_transfer: bool = False,
        context_signature: int = 0,
    ) -> "M1NormalizedRelation":
        if not parents or not evidence:
            raise ValueError("normalized relation requires provenance")
        if support < 0.0 or contradiction < 0.0:
            raise ValueError("normalized relation evidence must be non-negative")
        signature = stable_u64(*structural_key, person=b"v9-m1-structure")
        uid = MemoryUid.from_key(MemoryLevel.M1, MemoryType.NORMALIZED_RELATION, (signature,))
        return cls(
            uid,
            str(observable_relation),
            channel,
            int(signature),
            DerivationProvenance(tuple(parents), tuple(evidence)),
            float(support),
            float(contradiction),
            tuple(int(value) for value in temporal_offsets),
            int(causal_watermark),
            int(family_signature),
            bool(heldout_transfer),
            int(context_signature),
        )
