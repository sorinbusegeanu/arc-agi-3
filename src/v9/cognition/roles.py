from __future__ import annotations

from collections import defaultdict

from v9.memory.m2_family import M2TransformationFamily
from v9.memory.m3_role import M3FunctionalRole


def form_roles(families: tuple[M2TransformationFamily, ...], *, consequence_by_family: dict[int, int]) -> tuple[M3FunctionalRole, ...]:
    groups: dict[int, list[M2TransformationFamily]] = defaultdict(list)
    for family in families:
        consequence = int(consequence_by_family.get(family.uid.lo, family.structural_signature))
        groups[consequence].append(family)
    return tuple(M3FunctionalRole.form(tuple(groups[key]), consequence_signature=key) for key in sorted(groups))

