from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from v8.environments.synthetic_symbolic import SyntheticSymbolicConfig


class GroundingCondition(str, Enum):
    C0_INTERACTION_ONLY = "C0"
    C1_SYMBOLS_ONLY = "C1"
    C2_ALIGNED = "C2"
    C3_SHUFFLED = "C3"


@dataclass(frozen=True, slots=True)
class GroundingControlSpec:
    condition: GroundingCondition
    publish_world: bool
    publish_symbols: bool
    symbol_condition: str


CONTROL_SPECS = {
    GroundingCondition.C0_INTERACTION_ONLY: GroundingControlSpec(GroundingCondition.C0_INTERACTION_ONLY, True, False, "none"),
    GroundingCondition.C1_SYMBOLS_ONLY: GroundingControlSpec(GroundingCondition.C1_SYMBOLS_ONLY, False, True, "aligned"),
    GroundingCondition.C2_ALIGNED: GroundingControlSpec(GroundingCondition.C2_ALIGNED, True, True, "aligned"),
    GroundingCondition.C3_SHUFFLED: GroundingControlSpec(GroundingCondition.C3_SHUFFLED, True, True, "shuffled"),
}


def synthetic_config(condition: GroundingCondition, *, seed: int, mechanic: str = "advance", appearance: str = "default", held_out: bool = False) -> SyntheticSymbolicConfig:
    spec = CONTROL_SPECS[condition]
    symbol_condition = "permuted" if held_out and condition is GroundingCondition.C2_ALIGNED else spec.symbol_condition
    return SyntheticSymbolicConfig(
        mechanic=str(mechanic),
        symbol_condition=symbol_condition,
        appearance=("heldout-appearance" if held_out else str(appearance)),
        seed=int(seed) + (100_003 if held_out else 0),
    )
