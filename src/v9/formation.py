from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable

from v8.model import stable_u64
from v9.memory import M1NRecord, M1NUid, NormalizedChannel


class ProvenanceComposition(str, Enum):
    WORLD_ONLY = "WORLD_ONLY"
    SYMBOL_ONLY = "SYMBOL_ONLY"
    MIXED = "MIXED"


@dataclass(frozen=True, slots=True)
class M2Family:
    uid: int
    structural_signature: int
    members: tuple[M1NUid, ...]
    composition: ProvenanceComposition
    recurrence: int

    @property
    def action_authority(self) -> bool:
        return self.composition is not ProvenanceComposition.SYMBOL_ONLY


@dataclass(frozen=True, slots=True)
class M3Role:
    uid: int
    structural_signature: int
    families: tuple[int, ...]
    composition: ProvenanceComposition


def provenance_composition(channels: Iterable[NormalizedChannel]) -> ProvenanceComposition:
    rows = set(channels)
    has_world = NormalizedChannel.WORLD in rows
    has_symbol = NormalizedChannel.SYMBOL in rows
    has_cross = NormalizedChannel.CROSS_MODAL in rows
    if has_cross or (has_world and has_symbol):
        return ProvenanceComposition.MIXED
    if has_symbol:
        return ProvenanceComposition.SYMBOL_ONLY
    return ProvenanceComposition.WORLD_ONLY


def form_m2(records: Iterable[M1NRecord], *, min_recurrence: int = 2) -> tuple[M2Family, ...]:
    if int(min_recurrence) < 2:
        raise ValueError("M2 recurrence gate must be at least two")
    groups: dict[str, list[M1NRecord]] = {}
    for record in records:
        groups.setdefault(record.primitive.value, []).append(record)
    output: list[M2Family] = []
    for primitive in sorted(groups):
        rows = groups[primitive]
        if len(rows) < int(min_recurrence):
            continue
        composition = provenance_composition(row.channel for row in rows)
        signature = stable_u64(primitive, composition.value, person=b"v9-m2-structure")
        uid = stable_u64(signature, *(sorted(row.uid.value for row in rows)), person=b"v9-m2")
        output.append(M2Family(uid, signature, tuple(sorted((row.uid for row in rows), key=lambda x: x.value)), composition, len(rows)))
    return tuple(output)


def form_m3(families: Iterable[M2Family]) -> tuple[M3Role, ...]:
    groups: dict[int, list[M2Family]] = {}
    for family in families:
        groups.setdefault(int(family.structural_signature), []).append(family)
    output: list[M3Role] = []
    for signature in sorted(groups):
        rows = groups[signature]
        channels: list[NormalizedChannel] = []
        for family in rows:
            if family.composition is ProvenanceComposition.SYMBOL_ONLY:
                channels.append(NormalizedChannel.SYMBOL)
            elif family.composition is ProvenanceComposition.WORLD_ONLY:
                channels.append(NormalizedChannel.WORLD)
            else:
                channels.append(NormalizedChannel.CROSS_MODAL)
        composition = provenance_composition(channels)
        family_ids = tuple(sorted(int(row.uid) for row in rows))
        uid = stable_u64(signature, *family_ids, person=b"v9-m3-role")
        output.append(M3Role(uid, signature, family_ids, composition))
    return tuple(output)
