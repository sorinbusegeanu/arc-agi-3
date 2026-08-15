from __future__ import annotations

from collections import defaultdict
from typing import Protocol

from v8.arena import NodeRecord
from v8.diagnostics import HYPOTHESIS_IDS, hypothesis_statuses
from v8.model import MemoryLevel


class HypothesisReadView(Protocol):
    def node_records(self, *, level: MemoryLevel | int | None = None) -> tuple[NodeRecord, ...]: ...


def _recurring(rows: tuple[NodeRecord, ...]) -> tuple[NodeRecord, ...]:
    return tuple(row for row in rows if int(row.support_count) >= 2)


def evaluate_live_hypothesis_statuses(read_view: HypothesisReadView) -> dict[str, str]:
    """Return conservative live H01-H15 diagnostics from currently published RAM state.

    This is not the immutable scientific reporting window. Structural evidence may only
    produce PARTIALLY_VALID here. Claims requiring held-out transfer, causal ablation,
    replanning trials, preference evidence, or other evidence not stored in the current
    v8 arenas remain INSUFFICIENT_EVIDENCE.
    """
    levels = {
        level: tuple(read_view.node_records(level=level))
        for level in MemoryLevel
    }
    recurring = {level: _recurring(rows) for level, rows in levels.items()}
    statuses = hypothesis_statuses()

    if recurring[MemoryLevel.M1]:
        statuses["H01"] = "PARTIALLY_VALID"

    if any(float(row.prediction_error_sum) > 0.0 for row in levels[MemoryLevel.M1]):
        statuses["H02"] = "PARTIALLY_VALID"

    if recurring[MemoryLevel.M2]:
        statuses["H03"] = "PARTIALLY_VALID"

    if any(
        len(row.key_parts) >= 2 and int(row.key_parts[1]) != 0
        for row in recurring[MemoryLevel.M3]
    ):
        statuses["H04"] = "PARTIALLY_VALID"

    if recurring[MemoryLevel.M3]:
        statuses["H05"] = "PARTIALLY_VALID"

    if any(
        float(row.transfer_prior_sum) > 0.0
        for level in (MemoryLevel.M3, MemoryLevel.M4)
        for row in levels[level]
    ):
        statuses["H06"] = "PARTIALLY_VALID"

    if recurring[MemoryLevel.M4]:
        statuses["H07"] = "PARTIALLY_VALID"

    if recurring[MemoryLevel.M5]:
        statuses["H08"] = "PARTIALLY_VALID"

    option_rows = tuple(
        row
        for level in (MemoryLevel.M3, MemoryLevel.M4, MemoryLevel.M5, MemoryLevel.M6)
        for row in recurring[level]
    )
    if any(abs(float(row.future_option_delta)) > 1e-9 for row in option_rows):
        statuses["H09"] = "PARTIALLY_VALID"

    m1 = levels[MemoryLevel.M1]
    has_nonzero_option = any(abs(float(row.future_option_delta)) > 1e-9 for row in m1)
    has_zero_option = any(abs(float(row.future_option_delta)) <= 1e-9 for row in m1)
    if has_nonzero_option and has_zero_option:
        statuses["H10"] = "PARTIALLY_VALID"

    if any(
        int(row.validation_state) > 0 and float(row.transfer_prior_sum) > 0.0
        for row in levels[MemoryLevel.M4]
    ):
        statuses["H11"] = "PARTIALLY_VALID"

    if recurring[MemoryLevel.M7]:
        statuses["H12"] = "PARTIALLY_VALID"

    if recurring[MemoryLevel.M6]:
        statuses["H13"] = "PARTIALLY_VALID"

    strategies_by_outcome: dict[tuple[int, int], set[int]] = defaultdict(set)
    for row in levels[MemoryLevel.M7]:
        if len(row.key_parts) < 3:
            continue
        strategies_by_outcome[(int(row.key_parts[1]), int(row.key_parts[2]))].add(
            int(row.key_parts[0])
        )
    if any(len(strategies) >= 2 for strategies in strategies_by_outcome.values()):
        statuses["H14"] = "PARTIALLY_VALID"

    # H15 target-like preference requires explicit learned preference evidence.
    # The current v8 arenas do not yet store that relation, so it cannot be inferred
    # from wins, terminal labels, M6 outcomes, or M7 strategy counts.
    statuses["H15"] = "INSUFFICIENT_EVIDENCE"

    assert tuple(statuses) == HYPOTHESIS_IDS
    return statuses
