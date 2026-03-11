from __future__ import annotations

from typing import Dict, List, Tuple

from codex_baseline_v2.shared.config import CausalityConfigV2
from codex_baseline_v2.shared.schemas import ChangeEventV2, ContrastCaseV2, InterventionRecordV2, SCHEMA_VERSION


def build_contrast_cases(
    interventions: List[InterventionRecordV2],
    events: List[ChangeEventV2],
    history: List[ContrastCaseV2],
    cfg: CausalityConfigV2,
) -> list[ContrastCaseV2]:
    del cfg
    existing: Dict[str, ContrastCaseV2] = {row.contrast_id: row for row in history}
    events_by_episode: Dict[str, List[ChangeEventV2]] = {}
    for event in events:
        events_by_episode.setdefault(event.episode_id, []).append(event)

    for intervention in interventions:
        linked = [event for event in events_by_episode.get(intervention.start_episode_id, []) if event.start_step_idx >= intervention.start_step_idx]
        contrast_type = "supports_effect"
        supports = True
        if not intervention.contact:
            contrast_type = "no_contact"
            supports = False
        elif not linked:
            contrast_type = "no_effect"
            supports = False
        elif intervention.target_area_id and any(event.post_area_id not in {None, intervention.target_area_id} for event in linked):
            contrast_type = "mismatched_effect"
            supports = False
        contrast_id = "contrast:%s:%s" % (intervention.instruction_id, contrast_type)
        existing[contrast_id] = ContrastCaseV2(
            schema_version=SCHEMA_VERSION,
            game_id=intervention.game_id,
            contrast_id=contrast_id,
            intervention_id=intervention.instruction_id,
            contrast_type=contrast_type,
            matched_target_poi_id=intervention.target_poi_id,
            matched_area_id=intervention.target_area_id,
            event_ids=sorted({event.event_id for event in linked}),
            supports_causality=supports,
            confidence=0.7 if supports else 0.55,
        )
    return sorted(existing.values(), key=lambda row: row.contrast_id)
