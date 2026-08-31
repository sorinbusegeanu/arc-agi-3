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


@dataclass(frozen=True, slots=True, order=True)
class TransferKey:
    source_environment: int
    target_environment: int
    structural_uid: int
    context_scope_id: int = 0


@dataclass(frozen=True, slots=True)
class TransferDecision:
    admitted: bool
    target_action: int | None
    reason: str
    trust: int


class EnvironmentNeutralTransferGate:
    STATE_VERSION = 1

    def __init__(self, *, held_out_minimum: int = 1) -> None:
        self.held_out_minimum = int(held_out_minimum)
        self._trust: dict[TransferKey, int] = {}
        self._held_out_passes: dict[TransferKey, int] = {}
        self.false_transfer_count = 0

    def trust(self, key: TransferKey) -> int:
        return int(self._trust.get(key, 0))

    def observe_result(self, key: TransferKey, *, held_out: bool, success: bool) -> int:
        value = self.trust(key)
        if held_out and success:
            value += 1
            self._held_out_passes[key] = int(self._held_out_passes.get(key, 0)) + 1
        elif not success:
            value -= 1
            self.false_transfer_count += 1
        self._trust[key] = value
        return value

    def evaluate(self, key: TransferKey, *, structurally_admissible: bool, target_available_actions: tuple[int, ...] | list[int], target_grounded_action: int | None) -> TransferDecision:
        trust = self.trust(key)
        if not structurally_admissible:
            return TransferDecision(False, None, "STRUCTURALLY_INADMISSIBLE", trust)
        if target_grounded_action is None:
            return TransferDecision(False, None, "NO_TARGET_LOCAL_GROUNDING", trust)
        action = int(target_grounded_action)
        if action not in {int(v) for v in target_available_actions}:
            return TransferDecision(False, None, "TARGET_ACTION_UNAVAILABLE", trust)
        if int(self._held_out_passes.get(key, 0)) < self.held_out_minimum or trust <= 0:
            return TransferDecision(False, None, "HELD_OUT_TRUST_REQUIRED", trust)
        return TransferDecision(True, action, "ADMITTED", trust)

    def state_dict(self) -> dict[str, object]:
        return {
            "version": self.STATE_VERSION,
            "held_out_minimum": self.held_out_minimum,
            "false_transfer_count": self.false_transfer_count,
            "rows": [
                {"source": k.source_environment, "target": k.target_environment, "structural_uid": k.structural_uid, "context_scope_id": k.context_scope_id, "trust": v, "held_out_passes": int(self._held_out_passes.get(k, 0))}
                for k, v in sorted(self._trust.items())
            ],
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, object]) -> "EnvironmentNeutralTransferGate":
        if int(state.get("version", 0)) != cls.STATE_VERSION:
            raise ValueError("unsupported environment-neutral transfer state")
        obj = cls(held_out_minimum=int(state.get("held_out_minimum", 1)))
        obj.false_transfer_count = int(state.get("false_transfer_count", 0))
        for raw in state.get("rows", []):
            if not isinstance(raw, dict): continue
            key = TransferKey(int(raw["source"]), int(raw["target"]), int(raw["structural_uid"]), int(raw.get("context_scope_id", 0)))
            obj._trust[key] = int(raw.get("trust", 0)); obj._held_out_passes[key] = int(raw.get("held_out_passes", 0))
        return obj
