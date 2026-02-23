from __future__ import annotations

from dataclasses import dataclass

from .types import TransitionDelta


@dataclass
class MechanicBelief:
    movement_bias: float = 0.0
    interaction_bias: float = 0.0
    click_bias: float = 0.0


class MechanicInference:
    def __init__(self) -> None:
        self.belief = MechanicBelief()

    def update(self, action_name: str, delta: TransitionDelta) -> None:
        gain = 0.1 if not delta.no_op else -0.05
        if action_name in {"ACTION1", "ACTION2", "ACTION3", "ACTION4"}:
            self.belief.movement_bias += gain
        elif action_name == "ACTION5":
            self.belief.interaction_bias += gain
        elif action_name == "ACTION6":
            self.belief.click_bias += gain

    def bias_for(self, action_name: str) -> float:
        if action_name in {"ACTION1", "ACTION2", "ACTION3", "ACTION4"}:
            return self.belief.movement_bias
        if action_name == "ACTION5":
            return self.belief.interaction_bias
        if action_name == "ACTION6":
            return self.belief.click_bias
        return 0.0
