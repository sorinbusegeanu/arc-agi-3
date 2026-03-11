from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

from codex_baseline_v2.shared.config import ProbeModeConfigV2
from codex_baseline_v2.shared.schemas import ChangeEventV2, ProbeOutcomeV2, TrajectoryStepV2, InterventionRecordV2, TriggerZoneV2, SCHEMA_VERSION


def _cell_from_step(step: TrajectoryStepV2) -> Optional[Tuple[int, int]]:
    if step.actual_avatar_centroid is not None:
        return (int(round(step.actual_avatar_centroid[0])), int(round(step.actual_avatar_centroid[1])))
    if step.predicted_avatar_centroid is not None:
        return (int(round(step.predicted_avatar_centroid[0])), int(round(step.predicted_avatar_centroid[1])))
    return None


def derive_probe_outcomes(
    interventions: List[InterventionRecordV2],
    steps: List[TrajectoryStepV2],
    events: List[ChangeEventV2],
    cfg: ProbeModeConfigV2,
    trigger_zones: Optional[List[TriggerZoneV2]] = None,
) -> list[ProbeOutcomeV2]:
    zones = {zone.trigger_zone_id: set(tuple(cell) for cell in zone.cells) for zone in (trigger_zones or [])}
    steps_by_intervention: Dict[str, List[TrajectoryStepV2]] = {}
    for step in steps:
        if step.intervention_id:
            steps_by_intervention.setdefault(step.intervention_id, []).append(step)
    event_ids_by_instruction: Dict[str, List[str]] = {}
    for event in events:
        if event.trigger_instruction_id is None:
            continue
        event_ids_by_instruction.setdefault(event.trigger_instruction_id, []).append(event.event_id)
    outcomes: List[ProbeOutcomeV2] = []
    for idx, intervention in enumerate(interventions):
        intervention_steps = sorted(steps_by_intervention.get(intervention.instruction_id, []) or steps_by_intervention.get(f"intervention:{intervention.instruction_id}", []), key=lambda step: step.step_idx)
        zone_cells = zones.get(intervention.target_trigger_zone_id or "", set())
        entered = False
        dwelled = False
        executed_action = False
        crossed = False
        prev_inside = None
        dwell_run = 0
        for step in intervention_steps:
            cell = _cell_from_step(step)
            inside = cell in zone_cells if cell is not None and zone_cells else False
            entered = entered or inside
            executed_action = executed_action or (inside and step.action.action_id is not None)
            if inside:
                dwell_run += 1
            else:
                dwell_run = 0
            dwelled = dwelled or dwell_run >= 2
            if prev_inside is not None and prev_inside != inside:
                crossed = True
            prev_inside = inside
        event_ids = list(intervention.effect_event_ids or [])
        if not event_ids:
            event_ids = list(event_ids_by_instruction.get(intervention.instruction_id, []))
        outcomes.append(
            ProbeOutcomeV2(
                schema_version=SCHEMA_VERSION,
                game_id=intervention.game_id,
                probe_outcome_id=f"probe_outcome:{idx:03d}",
                intervention_id=intervention.instruction_id,
                probe_mode=intervention.probe_mode or intervention.intent_class,
                target_trigger_zone_id=intervention.target_trigger_zone_id,
                target_poi_id=intervention.target_poi_id,
                start_step_idx=intervention.start_step_idx,
                end_step_idx=intervention.end_step_idx,
                entered_target_region=entered,
                dwelled_in_target_region=dwelled,
                executed_action_in_target_region=executed_action,
                crossed_target_boundary=crossed,
                side_contact_label=None,
                null_effect=intervention.null_effect or not event_ids,
                event_ids=event_ids,
                confidence=0.8 if entered else 0.4,
            )
        )
    return outcomes
