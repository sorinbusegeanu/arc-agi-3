from __future__ import annotations

from .types import ActionProposal


class Critic:
    """Rule-based risk penalties for candidate actions."""

    def score_penalty(self, proposal: ActionProposal) -> tuple[float, tuple[str, ...]]:
        p = 0.0
        tags: list[str] = []
        # ACTION6 can be irreversible in some environments, apply conservative bias.
        if proposal.action.name == "ACTION6":
            p += 0.1
            tags.append("complex-risk")
        if proposal.action.name == "RESET":
            p += 0.3
            tags.append("reset-cost")
        return p, tuple(tags)
