from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import random
from typing import Sequence, TypeVar


class H16Condition(str, Enum):
    C0_INTERACTION_ONLY = "C0"
    C1_SYMBOLS_ONLY = "C1"
    C2_ALIGNED = "C2"
    C3_SHUFFLED = "C3"


@dataclass(frozen=True, slots=True)
class H16Metrics:
    interaction_prediction: float = 0.0
    action_success: float = 0.0
    symbol_conditioned_transfer: float = 0.0
    world_to_symbol_generalization: float = 0.0
    composition_success: float = 0.0


T = TypeVar("T")


def condition_payload(condition: H16Condition, interactions: Sequence[T], symbols: Sequence[T], *, seed: int) -> tuple[tuple[T, ...], tuple[T, ...]]:
    world = tuple(interactions)
    symbolic = tuple(symbols)
    if condition is H16Condition.C0_INTERACTION_ONLY:
        return world, ()
    if condition is H16Condition.C1_SYMBOLS_ONLY:
        return (), symbolic
    if condition is H16Condition.C3_SHUFFLED:
        shuffled = list(symbolic)
        random.Random(seed).shuffle(shuffled)
        return world, tuple(shuffled)
    return world, symbolic
