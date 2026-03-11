from __future__ import annotations

from typing import Dict, List, Optional

from codex_baseline_v2.shared.schemas import (
    AreaStateV2,
    ChangeEventV2,
    InterventionRecordV2,
    LatentStateHypothesisV1,
    SCHEMA_VERSION,
    TriggerZoneV2,
)


def induce_latent_states(
    event_table: List[ChangeEventV2],
    intervention_traces: List[InterventionRecordV2],
    trigger_zone_table: List[TriggerZoneV2],
    causal_edge_candidates: List[object],
    area_states: List[AreaStateV2],
    existing: Optional[List[LatentStateHypothesisV1]] = None,
) -> list[LatentStateHypothesisV1]:
    by_id: Dict[str, LatentStateHypothesisV1] = {state.latent_state_id: state for state in (existing or [])}
    remote_events = [event for event in event_table if event.locality in {"remote_same_area", "cross_area_transition", "global_same_area"}]
    for zone in trigger_zone_table:
        if zone.activation_count >= 2 and zone.anchor_poi_id is None:
            latent_id = f"latent:{zone.trigger_zone_id}:binary_gate"
            prev = by_id.get(latent_id)
            contradiction = zone.null_count + zone.contradiction_count
            support_event_ids = sorted(set((prev.support_event_ids if prev else []) + [event.event_id for event in remote_events[: zone.activation_count]]))
            contradiction_event_ids = sorted(set((prev.contradiction_event_ids if prev else []) + [event.event_id for event in remote_events[zone.activation_count : zone.activation_count + contradiction]]))
            by_id[latent_id] = LatentStateHypothesisV1(
                schema_version="v2.3.1",
                latent_state_id=latent_id,
                state_type="binary_gate",
                scope_type="trigger_zone",
                scope_id=zone.trigger_zone_id,
                candidate_values=["inactive", "active"],
                current_value="active" if zone.activation_count > contradiction else "inactive",
                confidence=min(1.0, 0.35 + 0.15 * zone.activation_count - 0.1 * contradiction),
                support_event_ids=support_event_ids,
                contradiction_event_ids=contradiction_event_ids,
                source_intervention_ids=list(prev.source_intervention_ids if prev else []),
                first_seen_step=prev.first_seen_step if prev else 0,
                last_updated_step=max((event.end_step_idx for event in remote_events), default=prev.last_updated_step if prev else None),
                notes="repeated hidden trigger precedes remote structured effect",
            )
    for area in area_states:
        if area.visit_count >= 2 and len(area.dynamic_object_ids) >= 1:
            latent_id = f"latent:{area.area_id}:area_phase"
            prev = by_id.get(latent_id)
            by_id[latent_id] = LatentStateHypothesisV1(
                schema_version="v2.3.1",
                latent_state_id=latent_id,
                state_type="area_phase",
                scope_type="area",
                scope_id=area.area_id,
                candidate_values=["phase_a", "phase_b"],
                current_value="phase_b" if area.visit_count % 2 == 0 else "phase_a",
                confidence=max(prev.confidence if prev else 0.0, min(1.0, 0.25 + 0.1 * area.visit_count)),
                support_event_ids=list(prev.support_event_ids if prev else []),
                contradiction_event_ids=list(prev.contradiction_event_ids if prev else []),
                source_intervention_ids=list(prev.source_intervention_ids if prev else []),
                first_seen_step=prev.first_seen_step if prev else 0,
                last_updated_step=prev.last_updated_step if prev else 0,
                notes="area alternates between repeated stable observations",
            )
    for intervention in intervention_traces:
        if intervention.reached and intervention.effect_event_ids:
            latent_id = f"latent:{intervention.target_poi_id or intervention.target_trigger_zone_id or intervention.instruction_id}:route_enabled"
            prev = by_id.get(latent_id)
            contradiction_ids = list(prev.contradiction_event_ids if prev else [])
            if intervention.null_effect:
                contradiction_ids.extend(intervention.effect_event_ids)
            by_id[latent_id] = LatentStateHypothesisV1(
                schema_version="v2.3.1",
                latent_state_id=latent_id,
                state_type="route_enabled",
                scope_type="poi" if intervention.target_poi_id else "trigger_zone",
                scope_id=intervention.target_poi_id or intervention.target_trigger_zone_id or intervention.instruction_id,
                candidate_values=["disabled", "enabled"],
                current_value="enabled",
                confidence=min(1.0, 0.4 + 0.1 * len(intervention.effect_event_ids)),
                support_event_ids=sorted(set((prev.support_event_ids if prev else []) + intervention.effect_event_ids)),
                contradiction_event_ids=sorted(set(contradiction_ids)),
                source_intervention_ids=sorted(set((prev.source_intervention_ids if prev else []) + [intervention.instruction_id])),
                first_seen_step=prev.first_seen_step if prev else intervention.start_step_idx,
                last_updated_step=intervention.end_step_idx,
                notes="route becomes useful only after structured prior intervention",
            )
    return list(by_id.values())
