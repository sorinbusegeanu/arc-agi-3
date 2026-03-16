from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class StopPolicyState:
    no_progress_rounds: int = 0
    planner_starvation_rounds: int = 0
    repeated_route_failure_rounds: int = 0
    no_new_evidence_rounds: int = 0


@dataclass
class StopPolicy:
    config: object
    state: StopPolicyState = field(default_factory=StopPolicyState)

    def update_and_decide(
        self,
        *,
        round_progress: float,
        won: bool,
        selected_candidate: dict | None,
        outcome: dict,
        analysis_summary: dict,
    ) -> str | None:
        candidate = dict(selected_candidate or {})
        candidate_class = str(candidate.get("candidate_class") or "")
        termination_reason = str(outcome.get("termination_reason") or dict(outcome.get("outcome", {})).get("termination_reason") or "")
        changed_steps = int(analysis_summary.get("changed_steps_count", 0) or 0)

        if won and self.config.runtime.stop_on_win:
            return "win"

        if round_progress <= 0.0 and not won:
            self.state.no_progress_rounds += 1
        else:
            self.state.no_progress_rounds = 0

        if candidate_class.startswith("fallback"):
            self.state.planner_starvation_rounds += 1
        else:
            self.state.planner_starvation_rounds = 0

        if termination_reason.startswith("route") or termination_reason in {"blocked", "stalled"}:
            self.state.repeated_route_failure_rounds += 1
        else:
            self.state.repeated_route_failure_rounds = 0

        if changed_steps <= 0 and round_progress <= 0.0 and not won:
            self.state.no_new_evidence_rounds += 1
        else:
            self.state.no_new_evidence_rounds = 0

        budget = int(self.config.runtime.no_progress_budget or 1)
        if self.state.no_progress_rounds >= budget:
            return "no_progress_budget"
        if self.state.planner_starvation_rounds >= budget:
            return "planner_starvation"
        if self.state.repeated_route_failure_rounds >= budget:
            return "repeated_route_failure_budget"
        if self.state.no_new_evidence_rounds >= budget:
            return "no_new_world_evidence_budget"
        return None
