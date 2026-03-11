from __future__ import annotations

from typing import Dict, List, Optional

from codex_baseline_v2.shared.schemas import LatentStateHypothesisV1


def latent_state_map(states: List[LatentStateHypothesisV1]) -> Dict[str, LatentStateHypothesisV1]:
    return {state.latent_state_id: state for state in states}


def active_latent_states(states: List[LatentStateHypothesisV1], min_confidence: float = 0.5) -> List[LatentStateHypothesisV1]:
    return [state for state in states if state.current_value is not None and state.confidence >= min_confidence]


def find_state_for_scope(states: List[LatentStateHypothesisV1], scope_type: str, scope_id: str) -> Optional[LatentStateHypothesisV1]:
    for state in states:
        if state.scope_type == scope_type and state.scope_id == scope_id:
            return state
    return None
