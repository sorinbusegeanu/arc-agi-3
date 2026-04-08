from __future__ import annotations

from v4.planning.planContracts import CandidatePlanV4
from v4.state.parsedState import ParsedStateV4


class TemporalVerifierV4:
    def assess(self, parsed_state: ParsedStateV4, candidate: CandidatePlanV4) -> tuple[bool, tuple[str, ...]]:
        if not candidate.action_prefix:
            return False, ("empty_action_prefix",)
        if parsed_state.temporal_reference is None:
            return True, ()
        safe_horizon_steps = parsed_state.temporal_reference.safe_horizon_steps
        hazard_window_remaining = parsed_state.temporal_reference.hazard_window_remaining
        prefix_length = len(candidate.action_prefix)
        rejection_reasons: list[str] = []
        if safe_horizon_steps < prefix_length:
            rejection_reasons.append("insufficient_safe_horizon")
        if hazard_window_remaining is not None and hazard_window_remaining <= 0:
            rejection_reasons.append("hazard_window_expired")
        return (not rejection_reasons), tuple(rejection_reasons)
