from __future__ import annotations

from typing import TYPE_CHECKING

from .beliefState import BeliefStateV4
from .safeInference import _state_key, extract_observed_and_unknown_cells, infer_local_facts
from v4.agentContract.environmentMetadata import V4EnvironmentMetadata
from v4.agentContract.types import V4Observation

if TYPE_CHECKING:
    from v4.state.parsedState import ParsedStateV4


class BeliefUpdaterV4:
    def initialize_from_observation(
        self,
        current_observation: V4Observation,
        environment_metadata: V4EnvironmentMetadata | None,
        step_index: int,
        parsed_state: ParsedStateV4 | None = None,
    ) -> BeliefStateV4:
        state_key = _state_key(current_observation)
        observed_cells, unknown_cells = extract_observed_and_unknown_cells(
            current_observation,
            environment_metadata,
            step_index,
            parsed_state,
        )
        inferred_facts = infer_local_facts(observed_cells, unknown_cells, step_index, state_key)
        return BeliefStateV4(
            revision=0,
            state_key=state_key,
            observed_cells=observed_cells,
            unknown_cells=unknown_cells,
            inferred_facts=inferred_facts,
            evidence_refs=(f"step:{step_index}", f"state:{state_key}"),
        )

    def update_from_observation(
        self,
        previous_belief: BeliefStateV4 | None,
        current_observation: V4Observation,
        environment_metadata: V4EnvironmentMetadata | None,
        step_index: int,
        parsed_state: ParsedStateV4 | None = None,
    ) -> BeliefStateV4:
        state_key = _state_key(current_observation)
        observed_cells, unknown_cells = extract_observed_and_unknown_cells(
            current_observation,
            environment_metadata,
            step_index,
            parsed_state,
        )
        inferred_facts = infer_local_facts(observed_cells, unknown_cells, step_index, state_key)
        current_refs = [f"step:{step_index}", f"state:{state_key}"]
        previous_refs = [] if previous_belief is None else list(previous_belief.evidence_refs)
        merged = current_refs + previous_refs
        deduped_reversed: list[str] = []
        seen: set[str] = set()
        for ref in reversed(merged):
            if ref in seen:
                continue
            seen.add(ref)
            deduped_reversed.append(ref)
        evidence_refs = tuple(reversed(deduped_reversed))
        return BeliefStateV4(
            revision=0 if previous_belief is None else previous_belief.revision + 1,
            state_key=state_key,
            observed_cells=observed_cells,
            unknown_cells=unknown_cells,
            inferred_facts=inferred_facts,
            evidence_refs=evidence_refs,
        )
