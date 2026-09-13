from __future__ import annotations

from v9.memory.m5_consequence import M5ConsequenceStructure
from v9.memory.m6_outcome import M6Outcome


def form_outcome(consequences: tuple[M5ConsequenceStructure, ...], *, within_class_diameter: int) -> M6Outcome:
    return M6Outcome.form(consequences, diameter_bound=within_class_diameter)

