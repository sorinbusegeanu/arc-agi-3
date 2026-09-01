from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class TransferClass(str, Enum):
    ARC_TO_ARC = "ARC_TO_ARC"
    GYM_TO_GYM = "GYM_TO_GYM"
    SYNTHETIC_TO_SYNTHETIC = "SYNTHETIC_TO_SYNTHETIC"
    BABYAI_TO_BABYAI = "BABYAI_TO_BABYAI"
    CROSS_FAMILY = "CROSS_FAMILY"
    CROSS_MODAL_GROUNDED = "CROSS_MODAL_GROUNDED"


@dataclass(frozen=True, slots=True)
class TransferKey:
    source_environment: int
    target_environment: int
    structural_uid: int


@dataclass(frozen=True, slots=True)
class TransferDecision:
    admitted: bool
    target_action: int | None
    reason: str
    trust: int


class EnvironmentNeutralTransferGate:
    def __init__(self) -> None: self._trust: dict[TransferKey, int] = {}
    def trust(self, key: TransferKey) -> int: return int(self._trust.get(key, 0))
    def observe_result(self, key: TransferKey, *, held_out: bool, success: bool) -> int:
        current = self.trust(key)
        if held_out and success: current += 1
        elif not success: current -= 1
        self._trust[key] = current; return current
    def evaluate(self, key: TransferKey, *, structurally_admissible: bool, target_available_actions: tuple[int, ...] | list[int], target_grounded_action: int | None, held_out_intervention_required: bool = True) -> TransferDecision:
        if not structurally_admissible: return TransferDecision(False, None, "STRUCTURALLY_INADMISSIBLE", self.trust(key))
        if target_grounded_action is None: return TransferDecision(False, None, "NO_TARGET_LOCAL_GROUNDING", self.trust(key))
        action = int(target_grounded_action)
        if action not in {int(x) for x in target_available_actions}: return TransferDecision(False, None, "TARGET_ACTION_UNAVAILABLE", self.trust(key))
        trust = self.trust(key)
        if held_out_intervention_required and trust <= 0: return TransferDecision(False, None, "HELD_OUT_TRUST_REQUIRED", trust)
        return TransferDecision(True, action, "ADMITTED", trust)
    def state_dict(self) -> dict[str, object]:
        return {"trust": [{"source": key.source_environment, "target": key.target_environment, "structure": key.structural_uid, "value": value} for key, value in sorted(self._trust.items(), key=lambda item: (item[0].source_environment, item[0].target_environment, item[0].structural_uid))]}
    @classmethod
    def from_state_dict(cls, state: dict[str, object]) -> "EnvironmentNeutralTransferGate":
        obj = cls()
        for raw in state.get("trust", []):
            if isinstance(raw, dict): obj._trust[TransferKey(int(raw["source"]), int(raw["target"]), int(raw["structure"]))] = int(raw["value"])
        return obj
