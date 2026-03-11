from __future__ import annotations

from typing import Dict, List

from codex_baseline_v2.shared.config import HiddenTriggerConfigV2
from codex_baseline_v2.shared.schemas import ChangeEventV2, HiddenTriggerHypothesisV2, InterventionRecordV2, SpatialInterventionCellV2, TriggerZoneV2, SCHEMA_VERSION


def induce_hidden_trigger_hypotheses(
    trigger_zones: List[TriggerZoneV2],
    interventions: List[InterventionRecordV2],
    spatial_field: List[SpatialInterventionCellV2],
    events: List[ChangeEventV2],
    cfg: HiddenTriggerConfigV2,
) -> list[HiddenTriggerHypothesisV2]:
    field_by_cell: Dict[tuple[object, object], SpatialInterventionCellV2] = {(cell.area_id, tuple(cell.cell)): cell for cell in spatial_field}
    by_zone = {zone.trigger_zone_id: zone for zone in trigger_zones}
    hypotheses: List[HiddenTriggerHypothesisV2] = []
    for idx, zone in enumerate(trigger_zones):
        if zone.anchor_poi_id is not None and zone.source_kind == "visible_poi":
            continue
        support = [record for record in interventions if record.target_trigger_zone_id == zone.trigger_zone_id and not record.null_effect]
        nulls = [record for record in interventions if record.target_trigger_zone_id == zone.trigger_zone_id and record.null_effect]
        contradictions = [record for record in interventions if record.target_trigger_zone_id == zone.trigger_zone_id and record.blocked]
        best_cell = next((field_by_cell.get((zone.area_id, tuple(cell))) for cell in zone.cells if (zone.area_id, tuple(cell)) in field_by_cell), None)
        required_action_id = zone.per_action_counts[0][0] if zone.per_action_counts else None
        required_dwell = max(cfg.dwell_step_thresholds) if zone.condition_type == "dwell" else None
        confidence = zone.hidden_trigger_confidence + 0.15 * len(support) - cfg.null_penalty * len(nulls) - cfg.contradiction_penalty * len(contradictions)
        hypotheses.append(
            HiddenTriggerHypothesisV2(
                schema_version=SCHEMA_VERSION,
                game_id=zone.game_id,
                hidden_hypothesis_id=f"hidden_trigger:{idx:03d}",
                trigger_zone_id=zone.trigger_zone_id,
                condition_type=zone.condition_type,
                required_action_id=required_action_id,
                required_dwell_steps=required_dwell,
                required_entry_side=None,
                required_preceding_zone_ids=[],
                effect_signature_id=events[0].effect_signature_id if events else None,
                support_intervention_ids=[record.instruction_id for record in support],
                null_intervention_ids=[record.instruction_id for record in nulls],
                contradiction_intervention_ids=[record.instruction_id for record in contradictions],
                confidence=max(0.0, min(1.0, confidence + (best_cell.hidden_trigger_score if best_cell is not None else 0.0) * 0.1)),
                status="promoted" if len(support) >= cfg.min_activation_support else "candidate",
            )
        )
    return hypotheses
