from __future__ import annotations

from .memory_episodic import EpisodicMemory
from .types import ActionProposal, NormalizedAction
from .world_model import EmpiricalWorldModel


class Explorer:
    def __init__(self, memory: EpisodicMemory, model: EmpiricalWorldModel) -> None:
        self.memory = memory
        self.model = model

    def propose(self, state_hash: str, available_actions: tuple[str, ...]) -> list[ActionProposal]:
        proposals: list[ActionProposal] = []
        for name in available_actions:
            action = NormalizedAction(name=name)
            unknown_bonus = 1.0 if self.model.is_unknown_edge(state_hash, action) else 0.0
            visits = self.memory.action_visits(state_hash, action)
            score = unknown_bonus + (1.0 / (1.0 + visits))
            proposals.append(ActionProposal(action=action, source="explorer", score=score))
        proposals.sort(key=lambda p: p.score, reverse=True)
        return proposals
