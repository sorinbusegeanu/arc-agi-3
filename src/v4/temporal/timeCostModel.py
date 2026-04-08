from __future__ import annotations

from v4.agentContract.types import V4Action


class TimeCostModelV4:
    def cost_for_action(self, action: V4Action) -> float:
        if action.action_name in {"inspect", "inspect_local"}:
            return 0.5
        if action.action_name == "click_at":
            return 1.0
        return 1.0
