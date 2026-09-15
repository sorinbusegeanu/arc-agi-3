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
    modality_support: tuple[tuple[str, int], ...] = ()
    support_decomposition: tuple[tuple[str, int], ...] = ()

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
        channels: dict[str, int] = {}
        for row in members:
            key = row.channel.value
            channels[key] = channels.get(key, 0) + 1
        decomposition = {"interaction_only": channels.get("WORLD", 0), "symbol_only": channels.get("SYMBOL", 0), "aligned_cross_modal": channels.get("CROSS_MODAL", 0), "heldout_transfer": sum(int(bool(getattr(row, "heldout_transfer", False))) for row in members)}
        return cls(uid, signature, DerivationProvenance(parents, evidence), len(members), benefit, tuple(sorted(channels.items())), tuple(sorted(decomposition.items())))
