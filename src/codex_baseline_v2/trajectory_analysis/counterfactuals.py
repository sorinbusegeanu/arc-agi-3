from __future__ import annotations

from typing import List

from codex_baseline_v2.shared.config import ProbeModeConfigV2
from codex_baseline_v2.shared.schemas import CounterfactualTraceV2, CausalChainHypothesisV2, ChangeEventV2, HiddenTriggerHypothesisV2, InterventionRecordV2, SCHEMA_VERSION


def build_counterfactual_traces(
    interventions: List[InterventionRecordV2],
    events: List[ChangeEventV2],
    hidden_hypotheses: List[HiddenTriggerHypothesisV2],
    chain_hypotheses: List[CausalChainHypothesisV2],
    cfg: ProbeModeConfigV2,
) -> list[CounterfactualTraceV2]:
    traces: List[CounterfactualTraceV2] = []
    for idx, intervention in enumerate(interventions):
        matched = [event.event_id for event in events if event.trigger_instruction_id == intervention.instruction_id]
        trace_type = "target_reached_without_effect" if intervention.reached and not matched else "same_effect_without_contact" if matched and not intervention.contact else "same_zone_without_action"
        supports = bool(matched) and not intervention.null_effect
        traces.append(
            CounterfactualTraceV2(
                schema_version=SCHEMA_VERSION,
                game_id=intervention.game_id,
                counterfactual_id=f"counterfactual:{idx:03d}",
                reference_intervention_id=intervention.instruction_id,
                trace_type=trace_type,
                target_poi_id=intervention.target_poi_id,
                target_trigger_zone_id=intervention.target_trigger_zone_id,
                matched_event_ids=matched,
                supports_reference=supports,
                confidence=0.75 if supports else 0.4,
            )
        )
    return traces
