from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from threading import Lock
from typing import Iterable

from v8.arena import NodeRecord
from v8.model import CognitiveState, MemoryLevel, MemoryType, MemoryUid, ValidationState


@dataclass(frozen=True, slots=True)
class ISFScore:
    option_structure_impact: float
    prediction_error: float
    learning_value: float
    transfer_potential: float
    explanatory_potential: float
    # Preserved as separate evidence for planning/reporting.  It is not a sixth
    # independently weighted ISF channel in v8.2.
    future_option_value: float
    total: float
    developmental_stage: int


# Five paper channels: OSI, PE, LV_hat, TP_prior, EP_hat.
_STAGE_WEIGHTS: dict[int, tuple[float, float, float, float, float]] = {
    0: (0.80, 0.00, 0.15, 0.00, 0.05),
    1: (0.30, 0.30, 0.25, 0.05, 0.10),
    2: (0.15, 0.15, 0.25, 0.20, 0.25),
    3: (0.10, 0.10, 0.15, 0.30, 0.35),
    4: (0.25, 0.10, 0.15, 0.20, 0.30),
    5: (0.30, 0.05, 0.10, 0.20, 0.35),
    6: (0.30, 0.05, 0.10, 0.15, 0.40),
}

_ADMISSIBLE = {
    int(CognitiveState.ACTIVE),
    int(CognitiveState.VALIDATED),
    int(CognitiveState.REACTIVATED),
}

_STAGE_LOCK = Lock()
_CURRENT_DEVELOPMENTAL_STAGE = 0
_DEVELOPMENTAL_STAGE_REVISION = 0


def publish_developmental_stage(stage: int) -> int:
    """Publish the Stage_t fixed for the current peer update interval."""
    global _CURRENT_DEVELOPMENTAL_STAGE, _DEVELOPMENTAL_STAGE_REVISION
    value = max(0, min(6, int(stage)))
    with _STAGE_LOCK:
        _CURRENT_DEVELOPMENTAL_STAGE = value
        _DEVELOPMENTAL_STAGE_REVISION += 1
    return value


def developmental_stage_snapshot() -> tuple[int, int]:
    with _STAGE_LOCK:
        return int(_CURRENT_DEVELOPMENTAL_STAGE), int(_DEVELOPMENTAL_STAGE_REVISION)


def current_developmental_stage() -> int:
    return developmental_stage_snapshot()[0]


def _clip(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def infer_developmental_stage(rows: Iterable[NodeRecord]) -> int:
    """Infer Stage_t only from capabilities present before the scoring interval."""
    rows = tuple(rows)
    minimum_stage = 1 if any(float(row.prediction_error) > 0.0 for row in rows) else 0
    active = tuple(row for row in rows if int(row.cognitive_state) in _ADMISSIBLE)
    if any(
        int(row.level) == int(MemoryLevel.M7)
        and int(row.memory_type) == int(MemoryType.STRATEGY)
        and row.attempt_weight > 0
        and row.strategy_reliability > 0
        for row in active
    ):
        return 6
    if any(
        int(row.level) == int(MemoryLevel.M6)
        and int(row.memory_type) == int(MemoryType.OUTCOME)
        for row in active
    ):
        return 5
    if any(int(row.level) == int(MemoryLevel.M5) for row in active):
        return 4
    if any(
        int(row.level) == int(MemoryLevel.M4)
        and (
            int(row.validation_state) >= int(ValidationState.VALIDATED)
            or row.transfer_prior > 0
        )
        for row in active
    ):
        return 3
    if any(int(row.level) in {int(MemoryLevel.M2), int(MemoryLevel.M3)} for row in active):
        return 2
    if any(
        int(row.level) == int(MemoryLevel.M1) and int(row.support_count) >= 3
        for row in rows
    ):
        return 1
    return minimum_stage


def _fallback_stage(row: NodeRecord) -> int:
    level = int(row.level)
    if level <= int(MemoryLevel.M0):
        return 0
    if level == int(MemoryLevel.M1):
        return 1
    if level in {int(MemoryLevel.M2), int(MemoryLevel.M3)}:
        return 2
    if level == int(MemoryLevel.M4):
        return 3
    if level == int(MemoryLevel.M5):
        return 4
    if level == int(MemoryLevel.M6):
        return 5
    return 6


def raw_components(row: NodeRecord) -> tuple[float, float, float, float, float, float]:
    """Return bounded raw OSI/PE/LV/TP/EP plus separate FO evidence.

    After learned reachability exists, |delta FO| is the structural OSI estimate.
    Before that, the stored non-semantic structural significance is used.
    """
    future = _clip(abs(row.future_option_delta) / 4.0)
    option_impact = future if abs(float(row.future_option_delta)) > 1e-9 else _clip(row.significance)
    prediction = _clip(row.prediction_error)
    learning = _clip(row.learning_value)
    transfer = _clip(row.transfer_prior)
    explanatory = _clip(row.explanatory_reach / 4.0)
    return option_impact, prediction, learning, transfer, explanatory, future


def _score_from_components(
    components: tuple[float, float, float, float, float, float],
    stage: int,
) -> ISFScore:
    option_impact, prediction, learning, transfer, explanatory, future = components
    weighted = (option_impact, prediction, learning, transfer, explanatory)
    total = sum(
        weight * component
        for weight, component in zip(_STAGE_WEIGHTS[stage], weighted, strict=True)
    )
    return ISFScore(
        float(option_impact),
        float(prediction),
        float(learning),
        float(transfer),
        float(explanatory),
        float(future),
        float(total),
        int(stage),
    )


def score_memory(row: NodeRecord, *, developmental_stage: int | None = None) -> ISFScore:
    """Static bounded fallback when a comparison-class snapshot is unavailable."""
    stage = (
        _fallback_stage(row)
        if developmental_stage is None
        else max(0, min(6, int(developmental_stage)))
    )
    return _score_from_components(raw_components(row), stage)


def _rank_normalize(
    values: list[tuple[MemoryUid, float]],
) -> dict[MemoryUid, float]:
    if len(values) < 4:
        return {uid: _clip(value) for uid, value in values}
    ordered = sorted(values, key=lambda item: (float(item[1]), item[0]))
    denominator = max(1, len(ordered) - 1)
    result: dict[MemoryUid, float] = {}
    for rank, (uid, _value) in enumerate(ordered):
        result[uid] = rank / denominator
    return result


def score_memories(
    rows: Iterable[NodeRecord],
    *,
    developmental_stage: int,
) -> dict[MemoryUid, ISFScore]:
    """Deterministically normalize within (level,type,Stage_t) comparison classes."""
    stage = max(0, min(6, int(developmental_stage)))
    grouped: dict[tuple[int, int, int], list[NodeRecord]] = defaultdict(list)
    for row in rows:
        grouped[(int(row.level), int(row.memory_type), stage)].append(row)

    result: dict[MemoryUid, ISFScore] = {}
    for members in grouped.values():
        raw = {row.uid: raw_components(row) for row in members}
        normalized_channels: list[dict[MemoryUid, float]] = []
        for index in range(5):
            normalized_channels.append(
                _rank_normalize([(row.uid, raw[row.uid][index]) for row in members])
            )
        for row in members:
            future = raw[row.uid][5]
            components = (
                normalized_channels[0][row.uid],
                normalized_channels[1][row.uid],
                normalized_channels[2][row.uid],
                normalized_channels[3][row.uid],
                normalized_channels[4][row.uid],
                future,
            )
            result[row.uid] = _score_from_components(components, stage)
    return result
