from __future__ import annotations

from collections import defaultdict

from v9.memory.m1_normalized import M1NormalizedRelation
from v9.memory.m2_family import M2TransformationFamily


def form_families(records: tuple[M1NormalizedRelation, ...], *, minimum_recurrence: int = 2) -> tuple[M2TransformationFamily, ...]:
    if minimum_recurrence < 2:
        raise ValueError("minimum recurrence must be at least two")
    groups: dict[int, list[M1NormalizedRelation]] = defaultdict(list)
    for row in records:
        groups[row.structural_signature].append(row)
    return tuple(
        M2TransformationFamily.form(tuple(groups[key]))
        for key in sorted(groups)
        if len(groups[key]) >= minimum_recurrence
    )

