from __future__ import annotations

from typing import Protocol, TypeVar

from v9.memory.identity import MemoryUid


class StrategyLike(Protocol):
    strategy_uid: MemoryUid
    target_outcome_uid: MemoryUid
    environment_id: int
    native_actions: tuple[int, ...]
    reliability: float
    relative_efficiency: float | None
    primary_valence: float


T = TypeVar("T", bound=StrategyLike)


def _rank(row: StrategyLike) -> tuple[float, float, float, float, MemoryUid]:
    grounding = float(getattr(row, "grounding_authority", 0.0))
    return (
        -grounding,
        -float(row.reliability),
        -float(row.primary_valence),
        -(float(row.relative_efficiency) if row.relative_efficiency is not None else -1.0),
        row.strategy_uid,
    )


def choose_strategy(strategies: tuple[T, ...], *, target_environment_id: int, available_actions: tuple[int, ...] = ()) -> T | None:
    eligible = tuple(
        row for row in strategies
        if int(row.environment_id) == int(target_environment_id)
        and (row.reliability > 0 or float(getattr(row, "grounding_authority", 0.0)) > 0.0)
        and row.native_actions
        and (not available_actions or int(row.native_actions[0]) in available_actions)
    )
    return min(eligible, key=_rank) if eligible else None


def replan(current: T, alternatives: tuple[T, ...], *, target_environment_id: int, available_actions: tuple[int, ...] = ()) -> T | None:
    same_outcome = tuple(
        row for row in alternatives
        if row.strategy_uid != current.strategy_uid
        and row.target_outcome_uid == current.target_outcome_uid
    )
    return choose_strategy(same_outcome, target_environment_id=target_environment_id, available_actions=available_actions)
